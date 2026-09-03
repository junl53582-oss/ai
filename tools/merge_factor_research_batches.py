"""
分批因子研究报告合并器 (tools/merge_factor_research_batches.py)

把 reports/production_research_v2/batch*/ 下的各批报告合并为全因子视图:
    merged/factor_ic.csv           # 各因子 RankIC (主视界) 全表
    merged/factor_summary.csv      # 各因子汇总指标
    merged/factor_selection.csv    # 分级结果
    merged/factor_correlation.csv  # 批内相关矩阵合并 (仅可用对, 非完整 97x97)
    merged/MERGE_REPORT.md         # 合并说明与分级统计

注意 (已知限制): 各批的全局 BH-FDR 在批内校正, 跨批 FDR p 值不可直接比较;
分级 (STRONG/USEFUL/REJECT) 以批内判定为准, 合并后以 factor_ic/summary 的
RankIC/ICIR 数值排序为准。

用法:
    python tools/merge_factor_research_batches.py [--out merged]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches-root", type=str,
                        default=str(ROOT / "reports" / "production_research_v2"))
    parser.add_argument("--out", type=str, default="merged")
    args = parser.parse_args()

    root = Path(args.batches_root)
    batch_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("batch")])
    if not batch_dirs:
        print("FATAL: 未找到任何 batch* 目录")
        sys.exit(1)

    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_ic, frames_sum, frames_sel, frames_corr = [], [], [], []
    for bd in batch_dirs:
        for name, frames in [
            ("factor_ic.csv", frames_ic),
            ("factor_summary.csv", frames_sum),
            ("factor_selection.csv", frames_sel),
            ("factor_correlation.csv", frames_corr),
        ]:
            f = bd / name
            if f.exists():
                frames.append(pd.read_csv(f))
        print(f"{bd.name}: factor_ic={len(frames_ic)} factor_summary={len(frames_sum)} "
              f"factor_selection={len(frames_sel)} factor_corr={len(frames_corr)}")

    if frames_ic:
        pd.concat(frames_ic, ignore_index=True).drop_duplicates().to_csv(out_dir / "factor_ic.csv", index=False)
    if frames_sum:
        pd.concat(frames_sum, ignore_index=True).drop_duplicates().to_csv(out_dir / "factor_summary.csv", index=False)
    if frames_sel:
        pd.concat(frames_sel, ignore_index=True).drop_duplicates().to_csv(out_dir / "factor_selection.csv", index=False)
    if frames_corr:
        pd.concat(frames_corr, ignore_index=True).drop_duplicates().to_csv(out_dir / "factor_correlation.csv", index=False)

    # 合并报告
    lines = [
        "# 分批因子研究报告合并 (v2)",
        "",
        f"来源批次: {len(batch_dirs)} 个目录 ({', '.join(d.name for d in batch_dirs)})",
        "",
        "## 分级统计 (批内判定合并)",
    ]
    if frames_sel:
        sel = pd.concat(frames_sel, ignore_index=True).drop_duplicates()
        for col in [c for c in sel.columns if "grade" in c.lower() or "level" in c.lower() or "classification" in c.lower()]:
            counts = sel[col].value_counts().to_dict()
            lines.append(f"- {col}: {counts}")
        lines.append("")
        lines.append("## 注意")
        lines.append("- 全局 BH-FDR 为批内校正, 跨批 FDR p 值不可直接比较; 数值比较以 factor_ic/factor_summary 的 RankIC/ICIR 为准。")
        lines.append("- 完整 97x97 相关矩阵需单批全因子重跑或分批后仅作参考。")
    (out_dir / "MERGE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"合并完成 -> {out_dir} (ic={len(frames_ic)} sum={len(frames_sum)} sel={len(frames_sel)} corr={len(frames_corr)})")


if __name__ == "__main__":
    main()
