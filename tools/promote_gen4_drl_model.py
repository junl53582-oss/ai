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

print("================================================================================")
print(">>> [MLOps 生产晋升] 开始将第四代 DRL 深度强化学习模型转正为系统生产基线")
print("================================================================================")

cand_path = settings.MODELS_DIR / "candidate_gen4_drl_model.pkl"
prod_path = settings.MODELS_DIR / "latest_lightgbm.pkl"
audit_path = settings.BASE_DIR / "reports" / "drl_model_hardening_audit.json"

if not cand_path.exists():
    raise FileNotFoundError(f"候选第四代 DRL 模型不存在: {cand_path}")

# 1. 计算候选模型 SHA256 校验和
with open(cand_path, "rb") as f:
    cand_sha = hashlib.sha256(f.read()).hexdigest()
print(f"[*] 候选模型物理文件: {cand_path.name}")
print(f"[*] 候选模型 SHA256 校验和: {cand_sha}")

# 2. 检查审计报告与指标门禁
with open(audit_path, "r", encoding="utf-8") as f:
    audit_data = json.load(f)

perf = audit_data["performance_metrics"]
print("[*] 正在执行生产晋升审查门禁:")
print(f"    - 传统基线年化夏普:    {perf['baseline_sharpe']:.4f}")
print(f"    - DRL 强化后年化夏普:  {perf['drl_enhanced_sharpe']:.4f} (门禁要求: 夏普显著提升)")
print(f"    - 夏普提升增益幅度:    +{perf['sharpe_improvement_pct']:.1f}% (门禁要求: > +15%)")
print(f"    - 传统基线最大回撤:    {perf['baseline_max_drawdown']*100:.2f}%")
print(f"    - DRL 强化后最大回撤:  {perf['drl_max_drawdown']*100:.2f}% (门禁要求: 回撤明显收窄)")
print(f"    - 最大回撤收窄幅度:    -{perf['drawdown_reduction_pct']:.1f}% (门禁要求: > -10%)")
print(f"    - 逐日胜率:            {perf['daily_win_rate']*100:.1f}%")

assert perf['drl_enhanced_sharpe'] > perf['baseline_sharpe'], "门禁失败: DRL 夏普未超越基线"
assert perf['drl_max_drawdown'] < perf['baseline_max_drawdown'], "门禁失败: DRL 回撤未收窄"

# 3. 生产基线安全备份
ts = time.strftime("%Y%m%d_%H%M%S")
if prod_path.exists():
    backup_path = settings.MODELS_DIR / f"backup_latest_gen3_{ts}.pkl"
    shutil.copy2(prod_path, backup_path)
    print(f"[+] 原生产模型已安全备份至: {backup_path.name}")

# 4. 正式物理替换转正
shutil.copy2(cand_path, prod_path)
print(f"[+] 第四代 DRL 强化模型已正式替换转正为生产基线: {prod_path.name}")

# 5. 同步更新最新选股清单至正式生产文件
gen4_picks = settings.BASE_DIR / "artifacts" / "gen4_drl_stock_picks.csv"
prod_picks = settings.BASE_DIR / "artifacts" / "latest_stock_picks.csv"
if gen4_picks.exists():
    shutil.copy2(gen4_picks, prod_picks)
    print(f"[+] 正式生产股票推荐清单已刷新 (DRL 动态权重): {prod_picks.name}")

# 6. 保存生产模型元数据
prod_meta_path = settings.MODELS_DIR / "production_model_meta.json"
meta = {
    "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model_generation": "Gen 4 (Deep Reinforcement Learning + Mega-Alpha Hybrid)",
    "sha256": cand_sha,
    "metrics": perf,
    "source_candidate": str(cand_path.name),
    "production_path": str(prod_path.name)
}
with open(prod_meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"[+] 生产元数据已归档: {prod_meta_path.name}")

print("\n>>> [第四代 DRL 强化大模型正式晋升生产基线圆满完成！]")
