"""
基准对比脚本
============
对比 vnet-asr1 与 WeNet AISHELL-1 标准基线的性能。

用法：
    python scripts/benchmark_baselines.py

输出：docs/benchmark_report.md — 格式化的对比报告
"""

import json
import os
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# Reference: WeNet official AISHELL-1 results (conformer U2++)
REFERENCE_BASELINES = {
    "WeNet Conformer (official)":
        "WeNet U2++ Conformer, 12 layers, d_model=256, 4 heads, "
        "Attention Rescoring, ~46M params",
    "WeNet Conformer (official) CTC greedy":
        "Same model, CTC greedy decoding",
    "ContextNet (official)":
        "ContextNet with ~10.6M params",
}


def get_vnet_results():
    """
    Read vnet-asr1 results from the latest training.
    Returns benchmark dict.
    """
    results = {
        "模型": "vnet-asr1 Conformer + CTC/Attn/Trans",
        "参数量": None,
        "数据集": "AISHELL-1",
        "CTC Greedy CER": None,
        "Attention CER": None,
        "Transducer CER": None,
        "流式 (chunk=16) CER": None,
        "训练数据": "AISHELL-1 (178小时)",
        "配置": "6层,d_model=144,4头",
    }

    # Try to parse from experiment results
    for exp_dir in ["exp/aishell_conformer", "exp/ablations"]:
        result_file = os.path.join(exp_dir, "results.json")
        if os.path.exists(result_file):
            with open(result_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in results:
                    if key in data and data[key] is not None:
                        results[key] = data[key]
            elif isinstance(data, list) and len(data) > 0:
                # Take last/default experiment
                last = data[-1]
                mapping = {
                    "CTC Greedy CER": "ctc_cer",
                    "Attention CER": "attention_cer",
                    "Transducer CER": "transducer_cer",
                    "流式 (chunk=16) CER": "streaming_cer",
                }
                for k, v in mapping.items():
                    if v in last and last[v] is not None:
                        results[k] = last[v]

    # Also try ablation results
    for mode in ["model_size", "num_layers", "loss_weights", "streaming"]:
        result_file = os.path.join("exp/ablations", f"results_{mode}.json")
        if os.path.exists(result_file):
            with open(result_file, encoding="utf-8") as f:
                data = json.load(f)
            if data and len(data) > 0:
                # Pick the "Default" or "Medium" config if available
                for item in data:
                    name = item.get("experiment", "")
                    if "Medium" in name or "Default" in name or "6 layers" in name:
                        mapping = {
                            "CTC Greedy CER": "ctc_cer",
                            "Attention CER": "attention_cer",
                            "Transducer CER": "transducer_cer",
                            "流式 (chunk=16) CER": "streaming_cer",
                        }
                        for k, v in mapping.items():
                            if v in item and item[v] is not None:
                                results[k] = item[v]
                        break

    # Count model params from checkpoint files
    for exp_dir in ["exp/aishell_conformer", "exp/ablations"]:
        for root, dirs, files in os.walk(exp_dir):
            for fname in ["final.pt", "best.pt"]:
                fpath = os.path.join(root, fname)
                if os.path.exists(fpath):
                    try:
                        size_mb = os.path.getsize(fpath) / 1024 / 1024
                        results["参数量"] = f"{size_mb:.1f}MB checkpoint"
                    except Exception:
                        pass
                    break

    return results


def generate_report(vnet_results=None):
    """Generate benchmark comparison report."""
    if vnet_results is None:
        vnet_results = get_vnet_results()

    cer_ctc = vnet_results.get("CTC Greedy CER", "TBD")
    cer_attn = vnet_results.get("Attention CER", "TBD")
    cer_trans = vnet_results.get("Transducer CER", "TBD")
    cer_stream = vnet_results.get("流式 (chunk=16) CER", "TBD")

    report = f"""# vnet-asr1 基准对比报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 实验配置

- **数据集**: AISHELL-1 (178小时，中文普通话)
- **训练集**: 120,098 条音频 (S0001-S0040)
- **开发集**: 14,326 条音频 (S0041-S0047)
- **测试集**: 7,176 条音频 (S0048-S0054)
- **特征**: 80维 Fbank + CMVN, 帧长25ms, 帧移10ms
- **配置**: {vnet_results.get('配置', '6层,d_model=144,4头')}

## 对比结果

| 模型 | 参数量 | 解码模式 | AISHELL-1 CER (%) | 说明 |
|------|--------|---------|-------------------|------|
| vnet-asr1 (ours) | ~6.6M | CTC Greedy | {cer_ctc} | 无语言模型 |
| vnet-asr1 (ours) | ~6.6M | Attention | {cer_attn} | 标准自回归 |
| vnet-asr1 (ours) | ~6.6M | Transducer | {cer_trans} | 流式友好 |
| vnet-asr1 (ours) | ~6.6M | Transducer (chunk=16) | {cer_stream} | 流式模式 |
| vnet-asr1 (ours) + LM | ~6.6M | Attn + LM Rescore | TBD | 外部4-gram LM |
| WeNet Conformer | ~46M | Attention Rescoring | ~4.6% | Wenet SOTA |
| WeNet Conformer | ~46M | CTC Greedy | ~5.8% | 基线 |
| WeNet Conformer | ~46M | CTC + Attn Rescoring | ~4.3% | 最佳结果 |

## 消融研究

### 模型大小影响

| d_model | 参数量 | Attn CER | Trans CER | 训练时间 |
|---------|--------|---------|-----------|---------|
| 96      | ~3.0M  | TBD     | TBD       | TBD     |
| 144     | ~6.6M  | TBD     | TBD       | TBD     |
| 256     | ~19M   | TBD     | TBD       | TBD     |

### 编码器层数影响

| 层数 | 参数量 | Attn CER | Trans CER |
|------|--------|---------|-----------|
| 3    | ~3.5M  | TBD     | TBD       |
| 6    | ~6.6M  | TBD     | TBD       |
| 12   | ~13M   | TBD     | TBD       |

### 损失权重影响

| CTC 权重 | Attn 权重 | Trans 权重 | CTC CER | Attn CER | Trans CER |
|----------|-----------|------------|---------|----------|-----------|
| 1.0      | 0.0       | 0.0        | TBD     | TBD      | TBD       |
| 0.0      | 1.0       | 0.0        | TBD     | TBD      | TBD       |
| 0.0      | 0.0       | 1.0        | TBD     | TBD      | TBD       |
| 0.33     | 0.33      | 0.34       | TBD     | TBD      | TBD       |
| 0.3      | 0.3       | 0.4        | TBD     | TBD      | TBD       |

## 流式与延迟分析

| 解码配置 | CER (%) | RTF | 延迟(ms) |
|---------|--------|-----|---------|
| 非流式 (chunk=0) | TBD | TBD | TBD |
| 流式 chunk=16, RC=4 | TBD | TBD | TBD |
| 流式 chunk=32, RC=4 | TBD | TBD | TBD |
| 流式 chunk=16, RC=8 | TBD | TBD | TBD |

> RTF = Real-Time Factor, 值越小越快
> 延迟 = 首次语音帧到识别结果的时间

## 结论

1. **参数量优势**: vnet-asr1 (~6.6M) 远小于 WeNet (~46M)，更适合端侧部署
2. **性能对比**: 需在 AISHELL-1 上训练后填充 CER 数据
3. **三解码头优势**: 支持 CTC/Attention/Transducer 三种解码，灵活适应不同场景
4. **流式支持**: chunk-wise 推理，适合实时语音场景

---

**注意**: 带 TBD 标记的数值需在完成 AISHELL-1 训练后更新。
运行 `scripts/ablation_study.py` 自动填充实验数据。
"""

    report_path = "docs/benchmark_report.md"
    os.makedirs("docs", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"基准报告已生成: {report_path}")
    return report_path


def main():
    results = get_vnet_results()
    report_path = generate_report(results)

    print(f"\n{'='*60}")
    print("基准对比报告生成完成!")
    print(f"查看报告: {report_path}")
    print(f"\n本地 WeNet AISHELL-1 基线:")
    print("  cd wenet-aishell-u2pp && .\\run_aishell.ps1")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
