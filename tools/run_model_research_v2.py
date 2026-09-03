"""
v2 数据集模型研究重认证运行器 (tools/run_model_research_v2.py)

对数据集 v2 (factor_matrix_300_v2.parquet) 执行正式模型研究与科学完整性认证。

前置条件 (fail-closed 预检):
    1. v2 因子矩阵存在 (data_storage/research/factor_matrix_300_v2.parquet)
    2. git 工作区干净 —— certified 模式要求 (用户未提交改动会在此显式报错, 而非
       等 5 小时后才失败)
    3. 声明代码冻结 SHA = 当前 HEAD

用法:
    python tools/run_model_research_v2.py [--output-root reports/model_research_v2] [--dataset <path>]
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_model_research import run_research, get_git_worktree_clean, get_git_commit_sha  # noqa: E402

V2_MATRIX = PROJECT_ROOT / "data_storage" / "research" / "factor_matrix_300_v2.parquet"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="v2 数据集模型研究重认证")
    parser.add_argument("--dataset", type=str, default=str(V2_MATRIX), help="因子矩阵路径")
    parser.add_argument("--output-root", type=str, default=str(PROJECT_ROOT / "reports" / "model_research_v2"))
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"FATAL: 数据集不存在: {dataset}")
        sys.exit(1)

    # 预检: 工作区必须干净 (certified 模式硬要求, 提前失败而非数小时后失败)
    clean, dirty = get_git_worktree_clean()
    sha = get_git_commit_sha()
    if not clean:
        print("=" * 64)
        print(f"FATAL: certified 模型研究要求干净的 git 工作区, 当前有 {len(dirty)} 个未提交改动:")
        for d in dirty[:10]:
            print(f"  - {d}")
        print("请先提交或暂存这些改动 (git stash) 后重跑。")
        print("=" * 64)
        sys.exit(1)

    run_config = {
        "run_mode": "certified",
        "expected_code_freeze_sha": sha,
        "seed_set": [42, 100, 2024],
    }

    print(f"启动 v2 模型研究认证: dataset={dataset} | code_freeze={sha[:8]} | output={args.output_root}")
    res = run_research(
        dataset_path=dataset,
        output_root=Path(args.output_root),
        run_config=run_config,
    )
    print(f"完成: run_id={res['run_id']} | overall={res['gate_matrix']['OVERALL_STATUS']}")
    sys.exit(0 if res["gate_matrix"]["OVERALL_STATUS"] in ("VERIFIED", "PASS", "CERTIFIED") else 1)


if __name__ == "__main__":
    main()
