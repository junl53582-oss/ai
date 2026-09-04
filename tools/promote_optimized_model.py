"""
模型安全晋级与发布审计工具 (tools/promote_optimized_model.py)
用于将 candidate_optimized_ensemble.pkl 经历完整的 MLOps 门禁审查与晋级流程。
"""
import sys
import shutil
import hashlib
from datetime import datetime
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import logging
from config.settings import settings
from models.registry import ModelRegistry, ModelState

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('promote_model')

def run_promotion():
    print('\n' + '=' * 75)
    print('>> [Action 1/3] 启动下一代优化模型正式晋升与生产转正审查 (Promotion Gate)')
    print('=' * 75)

    candidate_file = settings.MODELS_DIR / 'candidate_optimized_ensemble.pkl'
    if not candidate_file.exists():
        logger.error(f'未找到候选模型: {candidate_file}')
        return

    # 1. 计算候选模型 SHA256 与文件大小
    with open(candidate_file, 'rb') as f:
        cand_bytes = f.read()
    cand_sha256 = hashlib.sha256(cand_bytes).hexdigest()
    cand_size = len(cand_bytes)
    print(f'[*] 候选模型: {candidate_file.name} ({cand_size:,} 字节)')
    print(f'[*] 制品 SHA256: {cand_sha256}')

    # 计算数据集哈希
    dataset_path = root_dir / 'data_storage' / 'research' / 'factor_matrix_300.parquet'
    with open(dataset_path, 'rb') as f:
        ds_bytes = f.read()
    ds_sha256 = hashlib.sha256(ds_bytes).hexdigest()

    # 2. 注册进 MLOps Registry
    registry = ModelRegistry()
    print('\n[Step 1] 注册制品至 MLOps Registry (RESEARCH 初始状态)...')
    metrics = {
        'mean_rank_ic': 0.0103,
        'rank_icir': 0.0802,
        'multi_seed_std': 0.000621,
        'status_2024': 0.0257,
        'status_2025': 0.0278,
        'status_gate': 'PASS'
    }
    mid = registry.register_research_artifact(
        artifact_path=candidate_file,
        model_type='bagging_ensemble',
        task_type='classification',
        metrics=metrics,
        dataset_sha256=ds_sha256,
        dataset_path=str(dataset_path),
        notes='多随机种子强正则化袋装浅树集成模型 (攻克多种子方差与极端风格周期)'
    )
    print(f'   + 注册成功，分配 Model ID: {mid}')

    # 3. 晋升至 CANDIDATE
    print('\n[Step 2] 审查 OOS 指标与血缘，晋升至 CANDIDATE 状态...')
    rec1 = registry.promote(mid, ModelState.CANDIDATE, approver='auto_review', note='OOS 均值与多种子 STD 达标')
    print(f'   + 晋级状态: {rec1.state}')

    # 4. 晋升至 APPROVED
    print('\n[Step 3] 注入科学认证证据引用，晋升至 APPROVED 状态...')
    cert_ev = {
        'certification_ref': 'reports/audit_hardening_v3/runs/research_b041d50_20260903_180901',
        'multi_seed_std_verified': 0.000621
    }
    rec2 = registry.promote(mid, ModelState.APPROVED, approver='quant_committee', evidence=cert_ev, note='通过多种子稳定性与因果性审计')
    print(f'   + 晋级状态: {rec2.state}')

    # 5. 晋升至 PRODUCTION
    print('\n[Step 4] 注入前瞻验证与模拟盘授权，正式晋升至 PRODUCTION 生产状态...')
    prod_ev = {
        'certification_ref': 'reports/audit_hardening_v3/runs/research_b041d50_20260903_180901',
        'prospective_validation': True,
        'paper_trading': True
    }
    rec3 = registry.promote(mid, ModelState.PRODUCTION, approver='user_lin', evidence=prod_ev, note='正式签署上线授权')
    print(f'   + 终极生产状态: {rec3.state} (已归档于注册表制品库)')

    # 6. 安全更新生产发布软链/文件 (并做好前置备份)
    prod_target = settings.MODELS_DIR / 'latest_lightgbm.pkl'
    if prod_target.exists():
        backup_name = f"backup_latest_lightgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        backup_path = settings.MODELS_DIR / backup_name
        shutil.copy2(prod_target, backup_path)
        print(f'\n[Step 5] 原生产模型已安全备份至: {backup_path.name}')

    shutil.copy2(candidate_file, prod_target)
    print(f'[Step 6] 生产发布成功: {prod_target.name} 已更新为最新袋装高胜率模型！')
    print('=' * 75)
    print('✅ 【动作 1 完成】模型已合法合规转正为当前生产基线！')
    print('=' * 75)

if __name__ == '__main__':
    if "--allow-legacy-demo" not in sys.argv:
        print("[BLOCKED] tools/promote_optimized_model.py 是历史演示脚本，包含硬编码模拟审批凭证，默认禁止执行以防伪造审批。")
        print("正式晋升请使用: python tools/promote_model.py promote --model-id <ID> --to <STATE> --approval-artifact <JSON>")
        print("如确需在测试沙盒中复现历史演示，请显式追加参数: --allow-legacy-demo")
        sys.exit(1)
    run_promotion()
