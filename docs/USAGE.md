# agent-sim-platform 使用说明

本文档面向需要在本地快速运行仿真、校准模型、验证集群容量的用户。假设你已经完成仓库克隆并能在仓库根目录运行 Python 模块。

## 1. 环境检查

```bash
cd agent-sim-platform
python3 -m pytest tests/ -q
```

预期输出末尾为 `121 passed`。

## 2. 查看平台预设

平台内置了大量模型、硬件、集群、网络拓扑、基准 fixture 预设。

```bash
python3 -m agent_sim_platform list-hardware
python3 -m agent_sim_platform list-models
python3 -m agent_sim_platform list-clusters
python3 -m agent_sim_platform list-topologies
python3 -m agent_sim_platform list-benchmarks
python3 -m agent_sim_platform list-offload-tiers
python3 -m agent_sim_platform list-algorithms
```

## 3. 推理服务仿真（Inference Serving）

### 3.1 基础同地部署

```bash
python3 -m agent_sim_platform serve \
  --hardware H100-SXM5 \
  --model 70B-Dense \
  --algorithm dense \
  --request-length-mean 4096 \
  --output-length-mean 512 \
  --max-batch-size 4 \
  --simulation-duration 60 \
  --gpu-count 8 \
  --optimization layer3 \
  --precision FP8 \
  --kv-precision FP8 \
  --output /tmp/serve_baseline.json
```

输出 `/tmp/serve_baseline.json` 包含 TTFT/TPOT、吞吐、瓶颈、网络利用率等字段。

### 3.2 Prefill-Decode 分离

```bash
python3 -m agent_sim_platform serve \
  --hardware H100-SXM5 \
  --model 70B-Dense \
  --algorithm dense \
  --request-length-mean 4096 \
  --output-length-mean 512 \
  --max-batch-size 4 \
  --simulation-duration 60 \
  --pd-enabled \
  --prefill-gpus 4 \
  --decode-gpus 4 \
  --kv-transfer-bw 200 \
  --optimization layer3 \
  --precision FP8 \
  --kv-precision FP8
```

### 3.3 AFD 三池分离（适合 MoE）

```bash
python3 -m agent_sim_platform serve \
  --hardware H100-SXM5 \
  --model 1T-MoE \
  --algorithm moe \
  --request-length-mean 32768 \
  --output-length-mean 2048 \
  --max-batch-size 32 \
  --simulation-duration 60 \
  --afd-enabled \
  --attention-gpus 32 \
  --ffn-gpus 64 \
  --afd-decode-gpus 32 \
  --activation-transfer-bw 200 \
  --optimization layer3 \
  --precision FP8 \
  --kv-precision FP8
```

### 3.4 分层 KV Offload

```bash
python3 -m agent_sim_platform serve \
  --hardware H100-SXM5 \
  --model 70B-Dense \
  --algorithm dense \
  --request-length-mean 4096 \
  --output-length-mean 512 \
  --max-batch-size 4 \
  --simulation-duration 60 \
  --pd-enabled \
  --prefill-gpus 4 \
  --decode-gpus 4 \
  --kv-offload-tiers mooncake-like \
  --optimization layer3 \
  --precision FP8 \
  --kv-precision FP8
```

可选 `--kv-offload-tiers`：`none`, `hbm-dram`, `hbm-dram-ssd`, `hbm-icms`, `hbm-cxl`, `mooncake-like`, `lmcache-like`。

### 3.5 集群拓扑感知

指定 `--cluster` 后，引擎会自动按集群拓扑计算跨节点通信、网络利用率和瓶颈。

```bash
python3 -m agent_sim_platform serve \
  --hardware B200 \
  --model 1T-Dense \
  --algorithm dense \
  --cluster fat-tree-1024-h100 \
  --request-length-mean 10000000 \
  --output-length-mean 1000 \
  --max-batch-size 1 \
  --simulation-duration 120 \
  --tp 8 --pp 2 --cp 64 \
  --optimization layer3 \
  --precision FP8 \
  --kv-precision FP8
```

## 4. 训练仿真

```bash
python3 -m agent_sim_platform train \
  --model 70B-Dense \
  --hardware H100-SXM5 \
  --dataset-tokens 1B \
  --global-batch-size 1024 \
  --micro-batch-size 1 \
  --sequence-length 4096 \
  --parallelism dp=4,tp=8,pp=2 \
  --precision FP8 \
  --cluster fat-tree-256-h100
```

## 5. 容量估算

### 5.1 单机容量

```bash
python3 -m agent_sim_platform capacity \
  --hardware B200 \
  --model 1T-Dense \
  --context 10M \
  --optimization layer3 \
  --precision FP8 \
  --kv-precision FP8
```

### 5.2 集群分布式容量

```bash
python3 -m agent_sim_platform cluster-capacity \
  --hardware B200 \
  --model 1T-Dense \
  --cluster fat-tree-1024-h100 \
  --context 10000000 \
  --optimization layer3 \
  --precision FP8 \
  --kv-precision FP8 \
  --output /tmp/capacity.json
```

输出会给出可行/不可行、最小 GPU 数、每卡显存、建议 TP/PP/CP。

## 6. 基准测试

运行单个 fixture 并查看与观测值的误差：

```bash
python3 -m agent_sim_platform benchmark --name vllm_llama70b_serving
python3 -m agent_sim_platform benchmark --name mooncake_kimi_k2
python3 -m agent_sim_platform benchmark --name afd_megascale_moe
```

## 7. 校准

### 7.1 常量校准

```bash
python3 -m agent_sim_platform calibrate \
  --domain serving \
  --fit-params default_prefill_utilization,default_prefill_saturation_tokens,default_prefill_attention_hbm_passes,default_prefill_latency_floor_ms,default_decode_utilization \
  --max-iterations 3 \
  --tolerance 0.001 \
  --output /tmp/calibrate.json
```

### 7.2 混合模型校准（解析 + ML residual）

```bash
python3 -m agent_sim_platform calibrate \
  --domain serving \
  --fit-params default_prefill_utilization,default_prefill_saturation_tokens,default_prefill_attention_hbm_passes,default_prefill_latency_floor_ms,default_decode_utilization \
  --max-iterations 3 \
  --tolerance 0.001 \
  --residual \
  --residual-output agent_sim_platform/calibration/data/residual_model_serving.json \
  --output /tmp/calibrate_hybrid.json
```

校准完成后，`serve` / `benchmark` 会自动加载默认路径下的 residual model。

## 8. 生成报告

```bash
# 先保存 JSON 结果
python3 -m agent_sim_platform serve ... --output /tmp/result.json

# 再转 markdown 报告
python3 -m agent_sim_platform report \
  --input /tmp/result.json \
  --output /tmp/report.md
```

## 9. 关键概念速查

| 术语 | 含义 |
|------|------|
| TP | Tensor Parallelism，张量并行 |
| PP | Pipeline Parallelism，流水线并行 |
| CP | Context / Sequence Parallelism，上下文/序列并行 |
| PD 分离 | Prefill-Decode Disaggregation，分离 prefill 与 decode GPU 池 |
| AFD 分离 | Attention-FFN-Decode 三池分离，适合 MoE 大模型 |
| KV Offload | 将 KV Cache 卸载到 DRAM/SSD/ICMS/CXL 等多级存储 |
| Residual Model | 在解析模型 baseline 上叠加线性残差修正，降低 MAPE |

## 10. 典型工作流

1. **可行性判断**：用 `cluster-capacity` 确认目标模型/上下文/集群是否装得下。
2. **部署形态对比**：用 `serve` 对比同地 / PD / PD+KV / AFD 四种部署。
3. **模型校准**：用 `calibrate --residual` 对解析模型做混合校准。
4. **基准验证**：用 `benchmark` 验证关键 fixture 的误差是否可接受。
5. **报告输出**：用 `report` 将 JSON 结果转为 markdown 文档。

## 11. 注意事项

- `serve` 命令默认使用仓库内置的 residual model（`agent_sim_platform/calibration/data/residual_model_serving.json`）。如要关闭，可临时删除/重命名该文件。
- residual model 是用当前 6 个 serving fixture 训练的，对新 workload 的外推能力取决于特征分布是否被覆盖。
- 10M 以上超长上下文建议显式指定 `--cp`，否则容量估算器可能只按内存选 CP，导致 attention 计算瓶颈被低估。
