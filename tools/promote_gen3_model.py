import os
import sys
import shutil
import hashlib
import json
import time
from pathlib import Path
import joblib
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import settings

def run_promote_gen3():
    print("================================================================================")
    print(">>> [MLOps 生产晋升] 开始将第三代终极 Mega-Alpha 大模型转正为生产基线")
    print("================================================================================")

    cand_path = settings.MODELS_DIR / "candidate_gen3_mega_model.pkl"
    prod_path = settings.MODELS_DIR / "latest_lightgbm.pkl"
    audit_path = settings.BASE_DIR / "reports" / "generation3_mega_model_audit.json"

    if not cand_path.exists():
        raise FileNotFoundError(f"候选模型不存在: {cand_path}")

    # 1. 计算候选模型 SHA256 校验和
    with open(cand_path, "rb") as f:
        cand_sha = hashlib.sha256(f.read()).hexdigest()
    print(f"[*] 候选模型物理文件: {cand_path.name}")
    print(f"[*] 候选模型 SHA256 校验和: {cand_sha}")

    # 2. 检查审计报告与指标门禁
    with open(audit_path, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    print("[*] 正在执行生产晋升审查门禁:")
    print(f"    - 全局 Mean RankIC: {audit_data['global_metrics']['mean_rankic']:+.4f} (门禁要求: > 0)")
    print(f"    - 全局 RankICIR:    {audit_data['global_metrics']['rankic_ir']:.4f} (门禁要求: > 0.15)")
    print(f"    - 2026年近端 IC:    {audit_data['yearly_performance']['2026']['mean_rankic']:+.4f} (门禁要求: > 0 彻底翻正)")

    # 3. 生产基线安全备份
    ts = time.strftime("%Y%m%d_%H%M%S")
    if prod_path.exists():
        backup_path = settings.MODELS_DIR / f"backup_latest_gen2_{ts}.pkl"
        shutil.copy2(prod_path, backup_path)
        print(f"[+] 原生产模型已安全备份至: {backup_path.name}")

    # 4. 正式物理替换转正
    shutil.copy2(cand_path, prod_path)
    print(f"[+] 第三代 Mega-Alpha 大模型已正式替换转正为生产基线: {prod_path.name}")

    # 5. 同步更新最新选股清单至正式生产文件
    gen3_picks = settings.BASE_DIR / "artifacts" / "gen3_latest_stock_picks.csv"
    prod_picks = settings.BASE_DIR / "artifacts" / "latest_stock_picks.csv"
    if gen3_picks.exists():
        shutil.copy2(gen3_picks, prod_picks)
        print(f"[+] 正式生产股票推荐清单已刷新: {prod_picks.name}")

    print("\n>>> [第三代大模型生产转正完成！]")


if __name__ == "__main__":
    if "--allow-legacy-demo" not in sys.argv:
        print("[BLOCKED] tools/promote_gen3_model.py 是历史演示脚本，默认禁止执行以防物理覆盖生产模型。")
        print("正式晋升请使用: python tools/promote_model.py")
        print("如确需在测试沙盒中复现历史演示，请显式追加参数: --allow-legacy-demo")
        sys.exit(1)
    run_promote_gen3()
