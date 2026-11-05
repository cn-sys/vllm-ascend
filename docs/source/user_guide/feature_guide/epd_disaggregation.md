# Disaggregated Encoder (EPD)

Disaggregated Encoder (EPD) splits the multimodal path of a vision-language
model into up to three stages, each running in its own vLLM instance group:

- **E (encoder)** runs only the multimodal encoder (for example the ViT of
  Qwen3-VL) and writes the encoder cache (EC) to a shared backend.
- **P (prefill)** is optional. It consumes the encoder cache, computes the
  prompt KV cache, and transfers that KV cache to D through a prefill-decode
  connection such as `MooncakeLayerwiseConnector`.
- **D (decode)** generates tokens. In the E+PD topology the PD instance also
  runs prefill and is the encoder-cache consumer; in E+P+D the P instance is
  the encoder-cache consumer and D only consumes the KV cache.

A load-balancing proxy in front of the instances routes every request to one
encoder instance, optionally to one prefill instance, and finally to one decode
instance.

EPD builds on Prefill-Decode (PD) disaggregation. The E stage extracts the
vision work from the prompt path, and the remaining text path can still be
split into P and D. A text-only request skips the encoder stage entirely, so
decoding does not wait for vision encoding.

Running the stages in separate processes gives three practical benefits:

- **Independent scaling**: encoders are small compared with the language model,
  so E instances can be added or removed without touching the P/D fleet.
- **Lower time-to-first-token (TTFT)**: the encoder no longer competes with
  prefill and decode for the same device, and the result can be produced ahead
  of the prompt.
- **Cross-process encoder-cache reuse**: a request whose image was already
  encoded on any E instance can load the embedding from the shared cache
  instead of running the vision encoder again.

For the architecture, data/control plane, and interface specification, see the
[design document](../../developer_guide/Design_Documents/epd_disaggregation.md).

## Architecture and topologies

The example proxy exposes a single OpenAI-compatible endpoint and forwards the
three stages as follows. Text-only requests bypass E and, in E+P+D, still go
through P and D.

### Role and configuration matrix

| Topology | Instance | `--ec-transfer-config` | Other key parameters |
| --- | --- | --- | --- |
| E+PD | E | `ec_connector`; `ec_role: ec_producer`; extra config of the chosen connector | `--gpu-memory-utilization 0.01 --enforce-eager --no-enable-prefix-caching` |
| E+PD | PD | `ec_connector`; `ec_role: ec_consumer`; extra config of the chosen connector | `--gpu-memory-utilization 0.9` |
| E+P+D | E | same as the E+PD E instance | same as the E+PD E instance |
| E+P+D | P | `ec_role: ec_consumer`; `--kv-transfer-config` with `MooncakeLayerwiseConnector` and `kv_role: kv_producer` | `--gpu-memory-utilization 0.7` |
| E+P+D | D | no EC configuration | `--kv-transfer-config` with `MooncakeLayerwiseConnector` and `kv_role: kv_consumer` |
| Non-EPD | single instance | none | baseline deployment |

An E instance is marked as encoder-only (`mm_encoder_only`) by the
`ec_role: ec_producer` configuration: it runs the multimodal encoder and the
EC save path, and does not allocate a KV cache, hence the small
`--gpu-memory-utilization` value.

### E + PD topology

Two instances and two cards are enough:

1. The encoder instance on `CARD_E` with `ec_role: ec_producer`.
2. The PD instance on `CARD_PD` with `ec_role: ec_consumer`.
3. The proxy, pointed at the encoder URL and the PD URL, with
   `--prefill-servers-urls disable` to mark the E+PD topology.

Every multimodal request is first primed on the encoder instance, which stores
the encoder cache; the PD instance then loads the cache during prefill.

### E + P + D topology

Three instances are used when prefill and decode must scale separately:

1. The encoder instance with `ec_role: ec_producer`.
2. The prefill instance with `ec_role: ec_consumer` and
   `--kv-transfer-config` in the `kv_producer` role.
3. The decode instance with `--kv-transfer-config` in the `kv_consumer`
   role and without any EC configuration.
4. The proxy, with all three URL groups configured.

The P instance loads the encoder cache while prefilling the prompt and pushes
the KV cache layer-wise to D, which then decodes.

## Model Runner support

EPD runs on **Model Runner V2 (MRV2) by default**. MRV2 gained EPD and
encoder-only support in the encoder-cache and model-runner work that removed
`ec_transfer_config` and `mm_encoder_only` from the V2 blacklist, so no extra
flag is needed:

```bash
vllm serve <model> ... --ec-transfer-config '{...}'
```

To force the legacy Model Runner V1, export:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

The selection logic lives in `vllm_ascend/mrv2_utils.py`:

- **MRV2 (default)**: E nodes use the upstream `MMEncoderModelRunner` for the
  encoder-only instance; P/D nodes use the Ascend V2 model runner and the
  upstream encoder-cache contract.
- **MRV1**: EPD is served by the V1 Ascend model runner, which contains the
  complete encoder-cache implementation.

Some features still force V1 even when MRV2 is the default. In particular,
**the VL encoder graph is not supported on MRV2**: setting
`--compilation-config '{"cudagraph_mm_encoder": true}'` makes the runner fall
back to MRV1 with a warning. Other V2 blacklist entries (draft-window and
several speculative-decoding modes) behave the same way. Do not combine the VL
encoder graph with EPD if MRV2 behavior is required.

### Encoder-only pooling models are a different entry point

Models such as BERT, BGE, and SigLIP2 are served as **encoder-only pooling
models** with `--runner pooling` and the Ascend encoder-only model state. They
are not EPD deployments: there is no separate E/P/D split, no
`--ec-transfer-config`, and no encoder-cache transfer between instances. In
particular, SigLIP2 is a pooling embedding model and does not support PD
separation; see [SigLIP2](../../tutorials/models/SigLIP2.md#5-online-service-deployment).

## Encoder-cache connectors

Both E instances and consumer instances must use the same connector and the same
connector configuration. The EC data plane is selected by the connector in
`--ec-transfer-config`; Ascend-specific connectors are registered in
`vllm_ascend/distributed/ec_transfer/__init__.py`.

| Connector | Data plane | Extra configuration keys | When to use |
| --- | --- | --- | --- |
| `ECExampleConnector` | Shared filesystem, one safetensors file per `mm_hash` | `shared_storage_path` | E and P/D instances can see the same directory; simplest reference setup |
| `ECMooncakeConnector` | Mooncake with Ascend direct transfer and a bounce arena | `mooncake_protocol` (must be `ascend` when set; defaults to it), `ascend_mooncake_bounce_arena_size` | A Mooncake cluster with shared storage is deployed; multi-node EC reuse |
| `ECCPUConnector` | Pinned CPU mmap under `/dev/shm` | `ec_cpu_bytes` | Single host, repeated multimodal inputs; no cross-node transfer |
| `ECMemcacheConnector` | MemCache cluster through `memcache_hybrid` | `ec_connector_module_path`, `recv_buffer_tokens` | A MemCache deployment is available and EC entries must survive across hosts |

Notes:

- `ECExampleConnector` and `ECMooncakeConnector` are the cross-process
  connectors for EPD. The E instance acts as producer and P/D as consumer.
- `ECCPUConnector` is a single-host cache, supports `ec_both` only, and does
  not perform cross-node transfer; see
  [Encoder Cache CPU Offload](encoder_cache_cpu_offload.md).
- `ECMemcacheConnector` is configured with an explicit
  `ec_connector_module_path` (the upstream factory cannot find it otherwise)
  and requires the `memcache_hybrid` package plus `MMC_LOCAL_CONFIG_PATH`.
- The Mooncake connector requires a working Mooncake deployment; see the
  [PD disaggregation multi-node deployment guide](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md)
  for the Mooncake prerequisites.

## Quick start

### Prerequisites

- Two Ascend cards for E+PD (`CARD_E` and `CARD_PD`), or more for E+P+D.
- A local multimodal model directory, for example Qwen3-VL-8B-Instruct, visible
  from every instance.
- A shared encoder-cache path that all instances can read and write; with
  `ECExampleConnector` this must be a directory on a shared filesystem and the
  same absolute path must be used by E and P/D.
- The E+PD script launches its proxy with a relative path, so run it from
  `examples/disaggregated_encoder/`.

### E + PD on two cards

`examples/disaggregated_encoder/disagg_1e1pd_example.sh` starts the encoder,
the PD instance, the proxy, sends one image request, and then runs a benchmark.
It accepts the following environment overrides:

| Variable | Meaning | Default |
| --- | --- | --- |
| `MODEL` | Served model path or name | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `EC_SHARED_STORAGE_PATH` | Shared filesystem directory for the encoder cache | `/tmp/ec_cache` |
| `MM_IMAGE_PATH` | Local image used for the single request; required | none |
| `CARD_E` / `CARD_PD` | Cards used by the E and PD instances | `0` / `1` |
| `ENCODE_PORT` / `PREFILL_DECODE_PORT` / `PROXY_PORT` | Service ports | `19534` / `19535` / `10001` |
| `NUM_PROMPTS` | Prompts used by `vllm bench serve` | `100` |

```bash
cd examples/disaggregated_encoder

export MODEL=Qwen/Qwen3-VL-8B-Instruct
export EC_SHARED_STORAGE_PATH=/path/visible/to/all/instances/ec_cache
export MM_IMAGE_PATH=/path/to/an/image.jpg

bash disagg_1e1pd_example.sh
```

The local request is sent as a base64 image. To send an image request by URL
instead, call the proxy directly:

```bash
curl http://localhost:10001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
            {"type": "text", "text": "What is in this image?"}
        ]}
    ]
}'
```

An image for `MM_IMAGE_PATH` can be obtained from the
`vllm-ascend/mm_request` dataset used by `tools/send_mm_request.py`.

### E + P + D on three cards

`examples/epd_disaggregated/epd_disaggregated_guide.md` contains the complete
1E1P1D walkthrough for Qwen3-VL-8B-Instruct: three `vllm serve` commands with
ExampleConnector EC transfer and MooncakeLayerwise KV transfer, the proxy
command, and an example request. It shares the connector model with the E+PD
script but adds the P stage; the two guides are complementary and do not
duplicate each other.

The walkthrough places E, P, and D on `ASCEND_RT_VISIBLE_DEVICES` 0, 1, and 4,
so three cards are enough for the example. Production deployments can add more
P or D instances and scale the card count accordingly.

Set the following before running its commands:

```bash
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct    # same path on every node
export EC_SHARED_STORAGE_PATH=/path/visible/to/all/instances/ec_cache
```

### Proxy

The reference proxy is `examples/disaggregated_encoder/disagg_epd_proxy.py`.
Its three URL groups select the topology:

```bash
python examples/disaggregated_encoder/disagg_epd_proxy.py \
    --host 0.0.0.0 \
    --port 10001 \
    --encode-servers-urls "http://localhost:19534" \
    --prefill-servers-urls "disable" \
    --decode-servers-urls "http://localhost:19535"
```

- `--encode-servers-urls`: comma-separated E instance URLs.
- `--prefill-servers-urls`: comma-separated P instance URLs; use `disable`
  (or `none`) for the E+PD topology.
- `--decode-servers-urls`: comma-separated D (or PD) instance URLs.
- `--encoder-dispatch-mode`: `single` (default, one encoder request per
  request) or `fanout` (one encoder request per multimodal item).

Optional parameters are opt-in and keep the previous behavior by default:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--encode-timeout-seconds` | `0` | Per-request timeout for the encode stage; `0` keeps the shared overall client timeout of 100000 seconds |
| `--prefill-timeout-seconds` | `0` | Same for the prefill stage |
| `--decode-timeout-seconds` | `0` | Same for the decode stage, including streaming decode |
| `--encoder-failure-cooldown-seconds` | `30` | Seconds a failed encoder URL is deprioritized when several encoders are configured; `0` or negative disables health-aware ordering |
| `--probe-encoders-on-startup` | off | Probe every encoder `GET /health` on startup (5 s per probe) and fail fast when an instance is unreachable |
| `--shutdown-drain-seconds` | `0` | On shutdown, wait up to this many seconds for in-flight requests before closing the client sessions; `0` closes immediately |

## Health checks and startup order

Every vLLM instance exposes `GET /health`, which returns HTTP 200 once the
engine is ready. The reference proxy aggregates the three groups:

```bash
curl -s http://localhost:10001/health
```

The response is a JSON object with `proxy`, `encode_cluster`,
`prefill_cluster`, and `decode_cluster` fields. Each cluster field is
`healthy`, `unhealthy`, or `empty` (no instance configured); the endpoint
returns 200 when no cluster is unhealthy and 503 otherwise.

This aggregated `/health` endpoint belongs to the reference proxy
`examples/disaggregated_encoder/disagg_epd_proxy.py`. The layerwise proxy used
by the E+P+D guide
(`examples/epd_disaggregated/epd_load_balance_proxy_layerwise_server_example.py`)
only exposes `GET /healthcheck`, which reports its own status and the number of
connected encoder, prefill, decode, and PD instances without aggregating
instance health.

With `--probe-encoders-on-startup`, the reference proxy checks every encoder
`/health` once while starting (5 s timeout) and refuses to start when an
encoder is unreachable. The reference proxy also exposes Prometheus metrics at
`GET /metrics`: `epd_requests_total{stage,status}`,
`epd_stage_seconds_bucket/sum/count{stage}`, `epd_retries_total{stage}`,
`epd_encoder_requests_total`, `epd_mm_items_total`, and
`epd_requests_inflight`. The metrics endpoint is additive and does not change
the `/health` response.

Start the services in this order and wait for each instance's `/health` before
starting the next stage:

1. E instance(s).
2. P and D instance(s) (in E+P+D, P before D).
3. The proxy.
4. Wait for the proxy health endpoint before sending traffic (`/health` for
   the reference proxy, `/healthcheck` for the layerwise proxy).

The E+PD example script implements this wait with `curl -sf <url>/health` and
a timeout. On the NPU side, `npu-smi info -i <rank> -t health` can be used to
check device health while the service is starting.

## Graceful shutdown and cleanup

The reference proxy supports an optional drain window via
`--shutdown-drain-seconds` (default `0`, which closes immediately as before).
When it is greater than zero, shutdown waits until no request is in flight or
the window expires, then closes the client sessions. Stop in this order:

1. Stop the proxy; it drains and closes its client sessions in its shutdown
   hook.
2. Stop the P and D instances with SIGTERM, then the E instance(s). On SIGTERM
   the worker calls `ensure_kv_transfer_shutdown()` and then
   `ensure_ec_transfer_shutdown()`, which calls the connector `shutdown()`
   (see `vllm_ascend/worker/worker.py`).
3. Clean the encoder-cache backend:
   - `ECExampleConnector`: files under `shared_storage_path` are not removed
     automatically; remove them explicitly when the cache should not be reused.
   - `ECCPUConnector`: the mmap under `/dev/shm` is unregistered and unlinked
     during shutdown.
   - `ECMooncakeConnector`: the Mooncake deployment owns the storage
     lifecycle; follow the Mooncake guide for cleanup.

## Failure and retry semantics

- **Stage attempts**: each stage request gets up to three attempts in total
  (the initial attempt plus up to two retries).
- **Retryable failures**: connection errors are mapped to HTTP 502, stage
  timeouts to HTTP 504, and downstream 5xx responses to HTTP 502; all of them
  are retried. Every 2xx response is treated as success.
- **Downstream 4xx**: the 4xx status is preserved and the request is **not**
  retried.
- **Multi-encoder routing**: when several encoder URLs are configured, a
  failed encoder is deprioritized for `--encoder-failure-cooldown-seconds`.
  Only connection errors, timeouts and 5xx count as encoder failures; 4xx is a
  request-side error and does not mark the encoder unhealthy. Retries
  therefore prefer healthy encoders, and when all encoders are failing the
  proxy falls back to the full list. With a single encoder the selection is
  unchanged.
- **Exhaustion**: when all attempts fail, the client receives HTTP 500.
- **P stage failure**: a prefill failure returns its mapped status (4xx
  preserved, otherwise 502/504) and D is not started, so no partial output is
  produced.
- **D stage**: non-streaming requests use the same attempt rule. Streaming
  requests are retried only before the first chunk is forwarded; once
  streaming has started, an error is surfaced to the client instead of being
  retried.
- **Encoder-cache miss on the consumer**: behavior depends on the connector.
  `ECMemcacheConnector` falls back to local encoding and logs an
  `EC LOAD miss` warning. The shared-filesystem (`ECExampleConnector`) load
  path requires the EC entry to exist; if it is missing, the request fails with
  an error, so make sure the E instance and the P/D instances use the same
  `shared_storage_path`.
- **Timeouts**: the three `--*-timeout-seconds` options configure per-stage
  timeouts. With the default `0` the proxy omits the per-request timeout, so
  the shared 100000-second overall client timeout applies; a value greater
  than zero bounds the corresponding stage request (including streaming
  decode).

## Limitations

- **VL encoder graph**: `--compilation-config '{"cudagraph_mm_encoder": true}'`
  is not supported on MRV2 and falls back to MRV1. EPD with the encoder graph
  should be treated as a V1 deployment.
- **Encoder-only pooling models** (`--runner pooling`) are not EPD; they have
  no PD/EPD split. SigLIP2 does not support PD separation.
- **EC + Pipeline Parallel** is not verified. Treat the combination as unknown
  until coverage exists.
- **ECCPUConnector** works on a single host only (pinned `/dev/shm` mapping).
- **ECMooncakeConnector** requires a Mooncake cluster and shared storage, which
  must be provisioned before starting EPD.
- **ECExampleConnector** requires the same `shared_storage_path` on the
  encoder and consumer instances; a mismatch shows up as missing cache entries.
- **Encoder-cache keys are content hashes**: the key of an image is
  `mm_hash` (the multimodal feature identifier). The same image with different
  preprocessing or resolution can produce a different key, so cross-preprocessing
  reuse is not guaranteed. Duplicate images inside one request are deduplicated.
- **Processor cache**: setting `--mm-processor-cache-gb 0` disables the
  multimodal processor cache, so processed multimodal inputs are not reused.
  This cache is independent of the encoder-cache transfer described here.
- **310P**: multi-process EPD is not committed on 310P.
- **E+P+D** is documented as a supported deployment design, but end-to-end
  validation currently covers E+PD only; MRV1 EPD and E+P+D end-to-end tests are
  still being added.
