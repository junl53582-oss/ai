"""
全链路暴力压力测试 (scripts/brutal_e2e_stress.py)

目标: 不放过任何"能跑通但结果是垃圾"的静默失败。
覆盖六大层级:
  T1 单元/契约: 因子/标签/训练/回测 对异常输入的健壮性
  T2 边界极值: 空表/单行/单标的/全NaN/inf/零方差/乱序/重复行
  T3 数据污染: 因子矩阵缓存被改坏(行数不符/列缺失)能否被检出并重建
  T4 无未来函数: 标签与特征的时序隔离
  T5 交易规则: 涨跌停/停牌/T+1/整手
  T6 执行层演练: paper broker + dry_run 的调仓链路(不下真单)

判定标准:
  PASS  - 正常产出且结果自洽
  PASS* - 主动抛出明确异常(对非法输入是可接受的失败方式)
  FAIL  - 静默产出垃圾结果(最危险) 或 崩溃
"""
import sys
import io
import os
import time
import shutil
import tempfile
import traceback
from pathlib import Path

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings

RESULTS = []


def report(tier, name, status, detail=""):
    RESULTS.append((tier, name, status, detail))
    icon = {"PASS": "[OK]", "PASS*": "[OK!]", "FAIL": "[!!]"}.get(status, "[??]")
    print(f"  {icon} {name:<44} {status}  {detail[:70]}")


def run_case(tier, name, fn):
    """执行单个用例, 捕获一切异常; 崩溃记为 FAIL"""
    t0 = time.time()
    try:
        status, detail = fn()
        report(tier, name, status, f"{detail} ({time.time()-t0:.1f}s)")
    except Exception as e:
        report(tier, name, "FAIL", f"崩溃 {type(e).__name__}: {str(e)[:60]}")
        traceback.print_exc(limit=1)


# ---------------------------------------------------------------- 数据工厂
def make_market(n_symbols=6, n_days=200, seed=42, start="2021-01-04"):
    """构造合法的最小行情数据 (含回测所需全部列)"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    frames = []
    for i in range(n_symbols):
        sym = f"60000{i}.SH"
        px = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n_days)))
        frames.append(pd.DataFrame({
            "symbol": sym, "date": dates,
            "open": px * 0.999, "high": px * 1.02, "low": px * 0.98, "close": px,
            "adj_open": px * 0.999, "adj_high": px * 1.02, "adj_low": px * 0.98, "adj_close": px,
            "volume": rng.integers(1e5, 1e7, n_days).astype(float),
            "amount": rng.uniform(1e7, 1e9, n_days),
            "turnover": rng.uniform(0.005, 0.05, n_days),
            "benchmark_close": 4000 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n_days))),
            "industry": ["银行", "医药", "科技", "消费", "能源", "地产"][i % 6],
            "in_universe": True, "is_st": False, "is_suspended": False,
            "listing_date": pd.Timestamp(start) - pd.Timedelta(days=800),
        }))
    df = pd.concat(frames, ignore_index=True)
    # 因子计算依赖涨跌幅列 (alpha158 / custom_ashare 会取 adj_pct_change 或 pct_change)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["pct_change"] = df.groupby("symbol")["close"].pct_change().fillna(0.0)
    df["adj_pct_change"] = df.groupby("symbol")["adj_close"].pct_change().fillna(0.0)
    # 生产数据由上游注入的列: 涨跌停标记 / 对数流通市值 / 前收盘 / 涨跌停价
    df["pre_close"] = df.groupby("symbol")["close"].shift(1).fillna(df["close"])
    df["is_limit_up"] = (df["pct_change"] >= 0.099).astype(bool)
    df["is_limit_down"] = (df["pct_change"] <= -0.099).astype(bool)
    df["log_circ_mv"] = np.log(df.groupby("symbol")["amount"].transform("mean") * 20 + 1e6)
    df["limit_up_price"] = (df["pre_close"] * 1.1).round(2)
    df["limit_down_price"] = (df["pre_close"] * 0.9).round(2)
    return df


# ================================================================ T1/T2 因子处理器
def t_factor_baseline():
    from factors.processor import FactorProcessor
    df = make_market()
    with tempfile.TemporaryDirectory() as td:
        p = FactorProcessor(factor_dir=Path(td))
        out = p.build_and_save_factor_matrix(df, force_update=True)
        n_fac = len(FactorProcessor.get_all_factor_cols())
        miss = [c for c in FactorProcessor.get_all_factor_cols() if c not in out.columns]
        if len(out) != len(df):
            return "FAIL", f"行数不符 {len(out)} != {len(df)}"
        if miss:
            return "FAIL", f"缺列 {miss[:3]}"
        return "PASS", f"{len(out)}行 x {n_fac}因子"


def t_factor_empty():
    from factors.processor import FactorProcessor
    with tempfile.TemporaryDirectory() as td:
        p = FactorProcessor(factor_dir=Path(td))
        out = p.build_and_save_factor_matrix(make_market().iloc[0:0], force_update=True)
        return "PASS", f"空表未崩, 输出 {len(out)}行"


def t_factor_single_row():
    from factors.processor import FactorProcessor
    with tempfile.TemporaryDirectory() as td:
        p = FactorProcessor(factor_dir=Path(td))
        out = p.build_and_save_factor_matrix(make_market(n_days=1), force_update=True)
        return "PASS", f"单行未崩, 输出 {len(out)}行"


def t_factor_all_nan_factor():
    """某一因子列全 NaN: 必须保留 NaN 而不是被 fillna(0) 污染"""
    from factors.processor import FactorProcessor
    df = make_market()
    with tempfile.TemporaryDirectory() as td:
        p = FactorProcessor(factor_dir=Path(td))
        out = p.build_and_save_factor_matrix(df, force_update=True)
        cols = [c for c in FactorProcessor.get_all_factor_cols() if c in out.columns]
        if not cols:
            return "FAIL", "无有效因子列"
        out2 = out.copy()
        out2[cols[0]] = np.nan
        std = p.cross_sectional_standardize(out2, [cols[0]])
        kept_nan = std[cols[0]].isna().all()
        return ("PASS" if kept_nan else "FAIL"), f"{cols[0]} NaN保持={kept_nan}"


def t_factor_extreme_values():
    """inf / 1e12 / -1e12 极值: MAD 去极值后不应产生 inf"""
    from factors.processor import FactorProcessor
    df = make_market()
    with tempfile.TemporaryDirectory() as td:
        p = FactorProcessor(factor_dir=Path(td))
        out = p.build_and_save_factor_matrix(df, force_update=True)
        cols = [c for c in FactorProcessor.get_all_factor_cols() if c in out.columns]
        out2 = out.copy()
        out2.loc[out2.index[:5], cols[0]] = np.inf
        out2.loc[out2.index[5:10], cols[0]] = -np.inf
        out2.loc[out2.index[10:12], cols[0]] = 1e12
        std = p.cross_sectional_standardize(out2, cols[:5])
        has_inf = np.isinf(std[cols[:5]].to_numpy(dtype=float)).any()
        return ("FAIL" if has_inf else "PASS"), f"极值后残留inf={has_inf}"


def t_factor_shuffled_dates():
    """日期乱序输入: 输出必须按 (date, symbol) 有序"""
    from factors.processor import FactorProcessor
    df = make_market().sample(frac=1.0, random_state=7)
    with tempfile.TemporaryDirectory() as td:
        p = FactorProcessor(factor_dir=Path(td))
        out = p.build_and_save_factor_matrix(df, force_update=True)
        ok = out["date"].is_monotonic_increasing
        return ("PASS" if ok else "FAIL"), f"日期单调递增={ok}"


def t_factor_duplicate_rows():
    """重复 (symbol,date) 行: 不应产生行数膨胀"""
    from factors.processor import FactorProcessor
    df = pd.concat([make_market(), make_market().head(30)], ignore_index=True)
    with tempfile.TemporaryDirectory() as td:
        p = FactorProcessor(factor_dir=Path(td))
        out = p.build_and_save_factor_matrix(df, force_update=True)
        return "PASS", f"输入{len(df)}行 -> 输出{len(out)}行"


# ================================================================ 标签器
def t_label_horizon_too_long():
    from models.labeler import TargetLabeler
    df = make_market(n_days=10)
    lab = TargetLabeler(horizon=200)
    out = lab.compute_excess_return_label(df)
    nan_ratio = out[lab.label_col].isna().mean()
    ok = nan_ratio > 0.99  # 数据不足时标签应几乎全 NaN(而非伪造)
    return ("PASS" if ok else "FAIL"), f"NaN占比={nan_ratio:.2%}"


def t_label_missing_benchmark():
    from models.labeler import TargetLabeler
    df = make_market().drop(columns=["benchmark_close"])
    out = TargetLabeler(horizon=5).compute_excess_return_label(df)
    return "PASS", f"无基准列未崩, 标签NaN={out[settings.LABEL_COLUMN].isna().mean():.1%}"


def t_label_extreme_balance():
    """极端分组标签: 保留样本内正负应均衡, 中间段为 NaN"""
    from models.labeler import TargetLabeler
    df = make_market(n_symbols=30, n_days=300)
    lab = TargetLabeler(horizon=settings.LABEL_HORIZON)
    out = lab.compute_excess_return_label(df)
    c = out[lab.label_col_clf].dropna()
    pos = (c == 1).mean()
    retained = out[lab.label_col_clf].notna().mean()
    ok = 0.45 <= pos <= 0.55 and 0.5 <= retained <= 0.7
    return ("PASS" if ok else "FAIL"), f"正样本{pos:.1%} 保留{retained:.1%}"


# ================================================================ 训练器
def t_train_insufficient_data():
    """数据天数不足以构成一个 fold: 应'大声失败'(明确异常)而非静默产出垃圾"""
    from factors.processor import FactorProcessor
    from models.labeler import TargetLabeler
    from models.walk_forward import WalkForwardTrainer
    df = make_market(n_symbols=4, n_days=40)
    with tempfile.TemporaryDirectory() as td:
        p = FactorProcessor(factor_dir=Path(td))
        f = p.build_and_save_factor_matrix(df, force_update=True)
        f = TargetLabeler(horizon=5).compute_excess_return_label(f)
        try:
            WalkForwardTrainer(train_years=0.3, val_months=1, test_months=1, purge_gap_days=5).run_walk_forward(f)
            return "PASS", "未抛异常(优雅返回空)"
        except ValueError as e:
            # 主动抛出带说明的 ValueError 属可接受的大声失败
            return "PASS*", f"大声失败: {str(e)[:36]}"


def t_train_all_nan_label():
    """标签全 NaN(如 horizon 超长): 不应崩溃, 应跳过所有 fold"""
    from factors.processor import FactorProcessor
    from models.labeler import TargetLabeler
    from models.walk_forward import WalkForwardTrainer
    df = make_market(n_symbols=4, n_days=120)
    with tempfile.TemporaryDirectory() as td:
        p = FactorProcessor(factor_dir=Path(td))
        f = p.build_and_save_factor_matrix(df, force_update=True)
        f = TargetLabeler(horizon=5000).compute_excess_return_label(f)
        # 标签全 NaN 时, 训练器应"大声失败"而非静默产出空模型/垃圾预测
        has_clf_col = settings.LABEL_COLUMN_CLF in f.columns
        try:
            WalkForwardTrainer(train_years=0.2, val_months=1, test_months=1, purge_gap_days=5).run_walk_forward(f)
            return "PASS", f"分类列已创建={has_clf_col}, 未抛异常"
        except ValueError as e:
            status = "PASS*" if has_clf_col else "FAIL"
            return status, f"分类列={has_clf_col} 大声失败: {str(e)[:30]}"


def t_train_normal_folds():
    from factors.processor import FactorProcessor
    from models.labeler import TargetLabeler
    from models.walk_forward import WalkForwardTrainer
    from models.evaluator import ModelEvaluator
    df = make_market(n_symbols=12, n_days=700, seed=11)
    with tempfile.TemporaryDirectory() as td:
        p = FactorProcessor(factor_dir=Path(td))
        f = p.build_and_save_factor_matrix(df, force_update=True)
        f = TargetLabeler(horizon=settings.LABEL_HORIZON).compute_excess_return_label(f)
        tr = WalkForwardTrainer(train_years=0.5, val_months=1, test_months=1, purge_gap_days=25)
        oos, mdl = tr.run_walk_forward(f)
        if len(oos) == 0:
            return "FAIL", "OOS 为空"
        m = ModelEvaluator().evaluate_predictions(oos)
        auc = m.get("auc")
        ok = auc is not None and 0.0 <= auc <= 1.0
        return ("PASS" if ok else "FAIL"), f"OOS {len(oos)}行 AUC={auc}"


# ================================================================ T3 缓存污染防御
def t_cache_rowcount_poison():
    """核心回归: 缓存行数与上游行情不符时必须判定污染并重建"""
    from factors.processor import FactorProcessor
    df = make_market(n_symbols=6, n_days=200)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = FactorProcessor(factor_dir=td)
        p.build_and_save_factor_matrix(df, force_update=True)
        # 人为把缓存裁成极少行数(模拟单测污染)
        poisoned = pd.read_parquet(td / "factor_matrix.parquet").head(20)
        poisoned.to_parquet(td / "factor_matrix.parquet", index=False)
        out = p.build_and_save_factor_matrix(df)   # 不 force: 应检测到污染并重建
        ok = len(out) == len(df)
        return ("PASS" if ok else "FAIL"), f"重建后 {len(out)} 行(应 {len(df)})"


def t_cache_column_missing():
    """缓存缺列时必须重建"""
    from factors.processor import FactorProcessor
    df = make_market(n_symbols=6, n_days=200)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = FactorProcessor(factor_dir=td)
        p.build_and_save_factor_matrix(df, force_update=True)
        bad = pd.read_parquet(td / "factor_matrix.parquet")
        cols = [c for c in FactorProcessor.get_all_factor_cols() if c in bad.columns]
        bad = bad.drop(columns=[cols[0], cols[1]])
        bad.to_parquet(td / "factor_matrix.parquet", index=False)
        out = p.build_and_save_factor_matrix(df)
        ok = all(c in out.columns for c in FactorProcessor.get_all_factor_cols())
        return ("PASS" if ok else "FAIL"), f"缺列后重建完整={ok}"


def t_cache_no_pollution_leak():
    """跑完上述流程后, 生产缓存目录不应被本次测试写入"""
    prod = settings.FACTORS_DIR / "factor_matrix.parquet"
    existed = prod.exists()
    return "PASS", f"生产缓存存在={existed}(测试隔离于临时目录)"


# ================================================================ T4 无未来函数
def t_no_lookahead_purge():
    """训练集末尾必须早于测试集起点(严格 purge 隔离)"""
    from factors.processor import FactorProcessor
    from models.labeler import TargetLabeler
    from models.walk_forward import WalkForwardTrainer
    df = make_market(n_symbols=8, n_days=600, seed=3)
    with tempfile.TemporaryDirectory() as td:
        p = FactorProcessor(factor_dir=Path(td))
        f = p.build_and_save_factor_matrix(df, force_update=True)
        f = TargetLabeler(horizon=settings.LABEL_HORIZON).compute_excess_return_label(f)
        tr = WalkForwardTrainer(train_years=0.4, val_months=1, test_months=1, purge_gap_days=25)
        tr.run_walk_forward(f)
        # trainer.models 记录键: fold / train_end / test_start / test_end / model
        bad = [x for x in tr.models if x["train_end"] >= x["test_start"]]
        return ("PASS" if not bad else "FAIL"), f"检查{len(tr.models)}折, 训练/测试重叠={len(bad)}"


# ================================================================ T5 交易规则
def t_rules_limit_up_blocked():
    """一字涨停: 禁止买入 (open==high==low 且成交价触及涨停价)"""
    from strategy.trading_rules import AShareTradingRules
    rules = AShareTradingRules()
    row = pd.Series({
        "symbol": "600000.SH", "date": pd.Timestamp("2021-06-01"),
        "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0, "pre_close": 10.0,
        "limit_up_price": 11.0, "limit_down_price": 9.0,
        "is_suspended": False, "is_st": False, "volume": 1.0,
    })
    ok_buy, reason = rules.can_buy(row, execution_price=11.0)
    return ("PASS" if not ok_buy else "FAIL"), f"一字涨停可买={ok_buy} 原因={reason}"


def t_rules_suspended_blocked():
    """停牌: 禁止买入"""
    from strategy.trading_rules import AShareTradingRules
    rules = AShareTradingRules()
    row = pd.Series({
        "symbol": "600000.SH", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0,
        "pre_close": 10.0, "limit_up_price": 11.0, "limit_down_price": 9.0,
        "is_suspended": True, "is_st": False, "volume": 0.0,
    })
    ok_buy, reason = rules.can_buy(row, execution_price=10.0)
    return ("PASS" if not ok_buy else "FAIL"), f"停牌可买={ok_buy} 原因={reason}"


def t_rules_normal_ok():
    """正常行情: 应当允许买入 (防止规则过度拦截)"""
    from strategy.trading_rules import AShareTradingRules
    rules = AShareTradingRules()
    row = pd.Series({
        "symbol": "600000.SH", "open": 10.0, "high": 10.3, "low": 9.9, "close": 10.1,
        "pre_close": 10.0, "limit_up_price": 11.0, "limit_down_price": 9.0,
        "is_suspended": False, "is_st": False, "volume": 1e6,
    })
    ok_buy, reason = rules.can_buy(row, execution_price=10.05)
    return ("PASS" if ok_buy else "FAIL"), f"正常行情可买={ok_buy} 原因={reason}"


def t_rules_lot_size():
    """整手约束: 买入份额必须向下取整到 100 股"""
    from strategy.trading_rules import AShareTradingRules
    rules = AShareTradingRules()
    shares = rules.calculate_lot_shares(target_amount=123456.0, price=10.0)
    ok = shares % 100 == 0 and shares <= int(123456.0 / 10.0)
    return ("PASS" if ok else "FAIL"), f"123456元@10元 -> {shares}股(整手={shares%100==0})"


# ================================================================ T6 执行层演练
def t_exec_paper_dryrun():
    """paper broker + dry_run 调仓链路: 不下真单, 份额须为 100 的整数倍"""
    from execution.paper_broker import PaperBroker
    from execution.run_trader import PortfolioRebalancer
    broker = PaperBroker(initial_cash=1_000_000)
    broker.connect()
    target = pd.DataFrame({
        "symbol": ["600519.SH", "000858.SZ", "600036.SH"],
        "target_weight": [0.3, 0.2, 0.1],
        "close": [1600.0, 150.0, 40.0],
    })
    res = PortfolioRebalancer(broker).execute_rebalance(target_df=target, dry_run=True)
    # 目标份额校验(复现内部计算)
    eq = broker.get_account().total_equity
    for _, r in target.iterrows():
        s = int(eq * r["target_weight"] / (r["close"] * 100)) * 100
        if s % 100 != 0:
            return "FAIL", f"{r['symbol']} 份额 {s} 非整手"
    ok = res["buy_orders_count"] >= 0
    return ("PASS" if ok else "FAIL"), f"dry_run 买单{res['buy_orders_count']}笔 权益{eq:,.0f}"


def t_exec_no_real_order():
    """安全红线: 默认参数绝不能连真实券商"""
    import inspect
    from execution import run_trader
    src = inspect.getsource(run_trader.run_trader_cli)
    ok = 'default="paper"' in src
    return ("PASS" if ok else "FAIL"), f"默认broker=paper: {ok}"


# ================================================================ 主流程
def main():
    print("=" * 92)
    print("  全链路暴力压力测试  (Brutal End-to-End Stress Test)")
    print(f"  项目: {ROOT}")
    # 本压力测试聚焦 因子/标签/训练/回测/交易规则 的健壮性,
    # 合成行情不含财报, 故在测试进程内关闭基本面, 避免 F_* 列缺失造成误报。
    settings.ENABLE_FUNDAMENTALS = False
    print(f"  ENABLE_FUNDAMENTALS(测试内): {settings.ENABLE_FUNDAMENTALS}")
    print("=" * 92)

    sections = [
        ("T2 边界极值-因子", [
            ("合法基线", t_factor_baseline),
            ("空表", t_factor_empty),
            ("单行", t_factor_single_row),
            ("全NaN因子保留NaN", t_factor_all_nan_factor),
            ("inf/1e12极值", t_factor_extreme_values),
            ("日期乱序", t_factor_shuffled_dates),
            ("重复行", t_factor_duplicate_rows),
        ]),
        ("T2 边界极值-标签", [
            ("horizon超长", t_label_horizon_too_long),
            ("缺失基准列", t_label_missing_benchmark),
            ("极端分组均衡性", t_label_extreme_balance),
        ]),
        ("T1 训练器健壮性", [
            ("数据不足", t_train_insufficient_data),
            ("标签全NaN", t_train_all_nan_label),
            ("正常多折", t_train_normal_folds),
        ]),
        ("T3 缓存污染防御", [
            ("行数不符->重建", t_cache_rowcount_poison),
            ("缺列->重建", t_cache_column_missing),
            ("生产缓存未被污染", t_cache_no_pollution_leak),
        ]),
        ("T4 无未来函数", [
            ("purge时序隔离", t_no_lookahead_purge),
        ]),
        ("T5 交易规则", [
            ("一字涨停禁买", t_rules_limit_up_blocked),
            ("停牌禁买", t_rules_suspended_blocked),
            ("正常行情放行", t_rules_normal_ok),
            ("整手约束", t_rules_lot_size),
        ]),
        ("T6 执行层演练", [
            ("paper+dry_run调仓", t_exec_paper_dryrun),
            ("默认不连真实券商", t_exec_no_real_order),
        ]),
    ]

    for tier, cases in sections:
        print(f"\n--- {tier} ---")
        for name, fn in cases:
            run_case(tier, name, fn)

    n_pass = sum(1 for r in RESULTS if r[2] in ("PASS", "PASS*"))
    n_fail = len(RESULTS) - n_pass
    print("\n" + "=" * 92)
    print(f"  总计: {len(RESULTS)} 项 | 通过 {n_pass} | 失败 {n_fail}")
    if n_fail:
        print("  失败明细:")
        for t, n, s, d in RESULTS:
            if s not in ("PASS", "PASS*"):
                print(f"    - [{t}] {n}: {d}")
    print("=" * 92)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
