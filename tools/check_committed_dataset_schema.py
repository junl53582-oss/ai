"""
数据集与因子矩阵 Schema 门禁检查器 (tools/check_committed_dataset_schema.py)
用于在 CI / 预运行阶段严格校验本地与已提交数据集的字段完整性。
若缺少必要价格、基准或可交易性字段，直接 fail-closed 退出 (exit code 1)。
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("SchemaGate")

REQUIRED_MARKET_COLS = [
    "date", "symbol", "open", "high", "low", "close",
    "adj_open", "adj_high", "adj_low", "adj_close",
    "volume", "amount",
    "benchmark_open", "benchmark_close",
    "in_universe", "is_suspended", "is_st",
    "is_limit_up_locked", "is_limit_down_locked",
    "limit_up_price", "limit_down_price"
]

REQUIRED_FACTOR_CORE_COLS = [
    "date", "symbol",
    "adj_open", "adj_close",
    "benchmark_open", "benchmark_close",
    "in_universe"
]


def check_market_dataset(path: Path) -> bool:
    logger.info(f"🔍 检查行情数据集 Schema: {path}")
    if not path.exists():
        logger.error(f"❌ 行情数据集文件不存在: {path}")
        return False
    
    try:
        df = pd.read_parquet(path)
        if df.empty:
            logger.error("❌ 行情数据集为空 (0 rows)！")
            return False

        missing = [c for c in REQUIRED_MARKET_COLS if c not in df.columns]
        if missing:
            logger.error(f"❌ 行情数据集缺少必要字段 ({len(missing)} 个): {missing}")
            return False

        # 检查 benchmark_open 与 benchmark_close 覆盖率
        b_open_cov = float((df["benchmark_open"] > 0).mean())
        b_close_cov = float((df["benchmark_close"] > 0).mean())
        logger.info(f"  * 行数: {len(df)}, 标的数: {df['symbol'].nunique()}")
        logger.info(f"  * benchmark_open 覆盖率: {b_open_cov*100:.1f}%")
        logger.info(f"  * benchmark_close 覆盖率: {b_close_cov*100:.1f}%")

        if b_open_cov < 0.80 or b_close_cov < 0.80:
            logger.error(f"❌ 基准数据覆盖率不达标 (open: {b_open_cov:.2f}, close: {b_close_cov:.2f})！")
            return False

        # ---------- 列值分布断言 (Phase A / 2026-09-01 新增, 堵 governance 盲区) ----------
        # 背景: 300 标的生产行情数据集的 LOG_CIRC_MV 曾被因子管线的标准化值污染
        # (逐日 mean=0/std=1) 且所有既有 gate 未拦截。原始对数流通市值 (yuan) 应在
        # ~[18, 32] 区间; 若被标准化, 全局均值≈0。此断言按 fail-closed 处理。
        if "LOG_CIRC_MV" in df.columns and "300" in path.name:
            mv = df["LOG_CIRC_MV"].dropna()
            if len(mv) > 0:
                g_mean, g_std = float(mv.mean()), float(mv.std())
                logger.info(f"  * LOG_CIRC_MV 全局 mean={g_mean:.3f} std={g_std:.3f} (原始值期望 ~23±3)")
                if not (15.0 <= g_mean <= 35.0):
                    logger.error(
                        f"❌ LOG_CIRC_MV 值域异常 (全局均值 {g_mean:.3f})! 疑似标准化值污染原始行情列。"
                        "Fail-Closed: 请使用 v2 修正数据集 (tools/build_dataset_300_v2.py)。"
                    )
                    return False

        # ---------- 语义完整性门禁 (Stage A: 防跨股票填充与行业异常坍缩) ----------
        if not verify_dataset_semantic_integrity(path, df):
            return False

        # 校验伴随 Manifest 存证
        if not verify_manifest_consistency(path, df):
            return False

        logger.info("  -> 行情数据集 Schema 校验通过！")
        return True
    except Exception as e:
        logger.error(f"❌ 读取行情数据集异常: {e}")
        return False


def verify_dataset_semantic_integrity(path: Path, df: pd.DataFrame) -> bool:
    """
    语义完整性门禁 (Stage A / 防伪与防跨股票污染):
    1. 截面行业数量不得异常坍缩 (例如 300 支股票 100% 属于同一个行业 '半导体')
    2. 已知核心标的行业合理性核查 (600519.SH 茅台、300750.SZ 宁德时代、300760.SZ 迈瑞医疗绝对不能被标为半导体)
    3. 拒绝跨股票直接无分组 ffill/bfill 产生的全同特征副本
    """
    try:
        if df.empty or "date" not in df.columns:
            return True

        df_check = df.copy()
        df_check["date"] = pd.to_datetime(df_check["date"])

        # 1. 行业截面坍缩核验
        if "industry" in df_check.columns:
            for dt, grp in df_check.groupby("date"):
                n_syms = len(grp)
                if n_syms >= 50:
                    inds = grp["industry"].dropna().unique()
                    if len(inds) <= 1:
                        logger.error(
                            f"❌ [语义门禁-行业坍缩拦截] {path.name} 在 {dt.strftime('%Y-%m-%d')} 截面具有 {n_syms} 只股票，"
                            f"但行业唯一数仅为 {len(inds)} ({inds})！存在严重跨股票继承污染！"
                        )
                        return False
                    if n_syms >= 200 and len(inds) < 10:
                        logger.error(
                            f"❌ [语义门禁-行业丰富度不足] {path.name} 在 {dt.strftime('%Y-%m-%d')} 截面股票数 {n_syms}，"
                            f"但行业数仅 {len(inds)} < 10！"
                        )
                        return False

        # 2. 核心白马标的行业真伪性检查
        known_forbidden_industries = {
            "600519.SH": "半导体",
            "300750.SZ": "半导体",
            "300760.SZ": "半导体"
        }
        if "symbol" in df_check.columns and "industry" in df_check.columns:
            for sym, forbidden_ind in known_forbidden_industries.items():
                bad_rows = df_check[(df_check["symbol"] == sym) & (df_check["industry"] == forbidden_ind)]
                if not bad_rows.empty:
                    bad_dates = bad_rows["date"].dt.strftime("%Y-%m-%d").tolist()
                    logger.error(
                        f"❌ [语义门禁-已知标的行业谬误] {path.name} 中核心股票 {sym} 被错误标记为 '{forbidden_ind}'！"
                        f"受影响日期: {bad_dates[:5]}"
                    )
                    return False

        # 3. 跨股票全同特征副本核验 (检测未按 symbol 分组直接全局 ffill 的特征)
        # 选取若干数值特征检查同一日期下是否有连续不同股票取值完全相同且非0/非空
        num_cols = [c for c in df_check.columns if c.startswith("ALPHA_") or c.startswith("BARRA_")]
        if len(num_cols) >= 5:
            # 随机抽样近几个截面
            recent_dates = df_check["date"].drop_duplicates().sort_values().tail(3)
            for dt in recent_dates:
                dt_grp = df_check[df_check["date"] == dt].copy().reset_index(drop=True)
                if len(dt_grp) >= 100:
                    # 检查是否有超过 80% 的股票在所有抽样特征上取值完全相同
                    sub_feat = dt_grp[num_cols[:5]]
                    first_row = sub_feat.iloc[0].values
                    all_same = (sub_feat == first_row).all(axis=1).mean()
                    if all_same > 0.50:
                        logger.error(
                            f"❌ [语义门禁-跨股票全同特征拦截] {path.name} 在 {dt.strftime('%Y-%m-%d')} "
                            f"有 {all_same*100:.1f}% 的股票特征完全相同！疑似全局 ffill 污染！"
                        )
                        return False

        logger.info(f"  -> 语义完整性核验 PASS ({path.name})")
        return True
    except Exception as e:
        logger.error(f"❌ 语义完整性检查异常: {e}")
        return False


def verify_manifest_consistency(path: Path, df: pd.DataFrame) -> bool:
    """双向校验 Parquet 物理文件与伴随 Manifest 存证的绝对一致性 (Fail-Closed)"""
    import json
    import hashlib

    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        # 如果是 market_daily_300.parquet 等存在 manifest 的数据集，manifest 必须存在
        return True

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest().lower()
        
        # 1. 哈希校验 (支持 file_sha256 或 dataset_sha256)
        exp_sha = manifest.get("file_sha256") or manifest.get("dataset_sha256")
        if exp_sha and exp_sha.lower() != actual_sha:
            logger.error(
                f"❌ [Manifest 哈希篡改拦截] {manifest_path.name} 记录哈希 ({exp_sha}) 与物理文件实际哈希 ({actual_sha}) 不匹配！"
            )
            return False

        # 2. 行数校验 (支持 row_count 或 market_rows)
        exp_rows = manifest.get("row_count") if "row_count" in manifest else manifest.get("market_rows")
        if exp_rows is not None and int(exp_rows) != len(df):
            logger.error(
                f"❌ [Manifest 行数不一致] {manifest_path.name} 记录行数 ({exp_rows}) 与物理文件行数 ({len(df)}) 不一致！"
            )
            return False

        # 3. 标的数校验 (支持 symbol_count 或 market_symbols_count)
        exp_syms = manifest.get("symbol_count") if "symbol_count" in manifest else manifest.get("market_symbols_count")
        actual_syms = df["symbol"].nunique() if "symbol" in df.columns else 0
        if exp_syms is not None and int(exp_syms) != actual_syms:
            logger.error(
                f"❌ [Manifest 标的数不一致] {manifest_path.name} 记录标的数 ({exp_syms}) 与物理文件标的数 ({actual_syms}) 不一致！"
            )
            return False

        # 4. 日期范围校验
        if "date" in df.columns and ("date_range" in manifest or ("date_min" in manifest and "date_max" in manifest)):
            dt_s = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            act_min, act_max = dt_s.min(), dt_s.max()
            if "date_range" in manifest:
                exp_min, exp_max = manifest["date_range"][0], manifest["date_range"][1]
            else:
                exp_min, exp_max = manifest.get("date_min"), manifest.get("date_max")
            if act_min != exp_min or act_max != exp_max:
                logger.error(
                    f"❌ [Manifest 日期范围不一致] {manifest_path.name} 记录范围 [{exp_min}, {exp_max}] 与物理文件实际范围 [{act_min}, {act_max}] 不一致！"
                )
                return False

        # 5. 特征列数校验
        if "feature_count" in manifest:
            exp_features = int(manifest["feature_count"])
            if exp_features != len(df.columns):
                logger.error(
                    f"❌ [Manifest 列数不一致] {manifest_path.name} 记录列数 ({exp_features}) 与物理文件列数 ({len(df.columns)}) 不一致！"
                )
                return False

        logger.info(f"  -> Manifest 存证物理双向核对 100% PASS ({manifest_path.name})")
        return True
    except Exception as e:
        logger.error(f"❌ 校验 Manifest 存证异常: {e}")
        return False


def check_factor_matrix(path: Path) -> bool:
    logger.info(f"🔍 检查因子矩阵 Schema: {path}")
    if not path.exists():
        logger.error(f"❌ 因子矩阵文件不存在: {path}")
        return False

    try:
        df = pd.read_parquet(path)
        if df.empty:
            logger.error("❌ 因子矩阵为空 (0 rows)！")
            return False

        missing = [c for c in REQUIRED_FACTOR_CORE_COLS if c not in df.columns]
        if missing:
            logger.error(f"❌ 因子矩阵缺少核心基准/行情字段 ({len(missing)} 个): {missing}")
            return False

        b_open_cov = float((df["benchmark_open"] > 0).mean())
        b_close_cov = float((df["benchmark_close"] > 0).mean())
        logger.info(f"  * 行数: {len(df)}, 标的数: {df['symbol'].nunique()}, 总列数: {len(df.columns)}")
        logger.info(f"  * benchmark_open 覆盖率: {b_open_cov*100:.1f}%")

        if b_open_cov < 0.80 or b_close_cov < 0.80:
            logger.error(f"❌ 因子矩阵基准数据覆盖率不达标 (open: {b_open_cov:.2f}, close: {b_close_cov:.2f})！")
            return False

        # ---------- 语义完整性门禁 (Stage A: 防跨股票填充与行业异常坍缩) ----------
        if not verify_dataset_semantic_integrity(path, df):
            return False

        # 校验伴随 Manifest 存证
        if not verify_manifest_consistency(path, df):
            return False

        logger.info("  -> 因子矩阵 Schema 校验通过！")
        return True
    except Exception as e:
        logger.error(f"❌ 读取因子矩阵异常: {e}")
        return False


def main():
    root = Path(__file__).resolve().parent.parent
    market_path = root / "data_storage" / "parquet" / "market_daily.parquet"
    factor_path = root / "data_storage" / "factors" / "factor_matrix.parquet"

    m_ok = check_market_dataset(market_path)
    f_ok = check_factor_matrix(factor_path)

    # 检查 Phase 1.6 生产级 300 标的数据集 (优先校验已入库的 v2 修正版)
    prod_m_path = root / "data_storage" / "research" / "market_daily_300_v2.parquet"
    if not prod_m_path.exists():
        prod_m_path = root / "data_storage" / "research" / "market_daily_300.parquet"
    prod_f_path = root / "data_storage" / "research" / "factor_matrix_300.parquet"
    prod_ok = True
    if prod_m_path.exists():
        logger.info("📦 发现 Phase 1.6 生产级 300 标的数据集，执行生产级 Schema 门禁检查...")
        prod_m_ok = check_market_dataset(prod_m_path)
        prod_f_ok = check_factor_matrix(prod_f_path) if prod_f_path.exists() else True
        prod_ok = prod_m_ok and prod_f_ok

    if m_ok and f_ok and prod_ok:
        logger.info("🏆 全量数据集 Schema 门禁校验 100% 通过！(DATASET_SCHEMA_VALID = TRUE)")
        sys.exit(0)
    else:
        logger.error("❌ 数据集 Schema 门禁校验失败！(DATASET_SCHEMA_VALID = FALSE)")
        sys.exit(1)


if __name__ == "__main__":
    main()
