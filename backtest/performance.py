"""
量化绩效与风险评价指标计算引擎 (backtest/performance.py)
严格遵循现代投资组合理论标准金融定义与数学一致性：
1. Sharpe: 基于日度超额收益均值与日度波动率的标准年化
2. Sortino: 基于日度超额收益均值与下行半方差的标准年化
3. Alpha / Beta: 标准日度 CAPM OLS 回归截距年化 (alpha_capm_regression) 与 CAGR 近似 (alpha_cagr_approx)
4. 平均持仓周期: 基于交易日历的真实交易日天数 (无平仓单时严格返回 0.0)
5. 统一审计输出: 接入 Fail-Closed AuditMetadata 实体
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import pandas as pd
import numpy as np

from config.settings import settings
from .audit import AuditMetadata, AuditCollector

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """策略绩效评估与报表生成器 (含精确数学公式与审计元数据)"""

    def __init__(self, risk_free_rate: float = 0.02):
        self.rf = risk_free_rate
        self.daily_rf = (1.0 + risk_free_rate) ** (1.0 / 242.0) - 1.0

    def calculate_metrics(
        self,
        equity_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        closed_trades: Optional[List[Dict[str, Any]]] = None,
        audit_info: Optional[Union[Dict[str, Any], AuditMetadata]] = None
    ) -> Dict[str, Any]:
        """计算全套量化绩效指标与审计报告"""
        df = equity_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)

        n_days = len(df)
        if n_days < 2:
            return {}

        years = max(n_days / 242.0, 1.0 / 242.0)

        # ---------------- 1. 收益指标 ----------------
        strat_ret = df["strategy_return"].iloc[1:] if len(df) > 1 else df["strategy_return"]
        bench_ret = df["benchmark_return"].iloc[1:] if len(df) > 1 else df["benchmark_return"]
        
        cum_strat_ret = (df["total_equity"].iloc[-1] / df["total_equity"].iloc[0]) - 1.0
        cum_bench_ret = (df["benchmark_equity"].iloc[-1] / df["benchmark_equity"].iloc[0]) - 1.0
        excess_cum_ret = cum_strat_ret - cum_bench_ret

        # 年化复合增长率 (CAGR)
        cagr = ((1.0 + max(cum_strat_ret, -0.9999)) ** (1.0 / years)) - 1.0
        bench_cagr = ((1.0 + max(cum_bench_ret, -0.9999)) ** (1.0 / years)) - 1.0

        # ---------------- 2. 风险与波动指标 (标准日度超额计算) ----------------
        strat_excess_daily = strat_ret - self.daily_rf
        bench_excess_daily = bench_ret - self.daily_rf

        ann_vol = float(strat_ret.std(ddof=1) * np.sqrt(242.0)) if len(strat_ret) > 1 else 0.0
        bench_vol = float(bench_ret.std(ddof=1) * np.sqrt(242.0)) if len(bench_ret) > 1 else 0.0

        # 动态水下回撤
        equity_series = df["total_equity"]
        cummax = equity_series.cummax()
        drawdowns = (equity_series - cummax) / cummax
        max_drawdown = float(drawdowns.min())

        # 索提诺下行标准差 (Downside Semi-Deviation)
        downside_diff = np.minimum(strat_excess_daily, 0.0)
        downside_std = float(np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(242.0)) if len(downside_diff) > 0 else 1e-6
        if downside_std < 1e-6:
            downside_std = 1e-6

        # 标准年化夏普、索提诺与卡玛比率
        mean_excess_ann = float(strat_excess_daily.mean() * 242.0)
        sharpe = mean_excess_ann / (ann_vol + 1e-8) if ann_vol > 0 else 0.0
        sortino = mean_excess_ann / (downside_std + 1e-8)
        calmar = cagr / (abs(max_drawdown) + 1e-8)

        # ---------------- 3. Alpha & Beta (CAPM 日度回归) ----------------
        if len(strat_excess_daily) > 1:
            cov_matrix = np.cov(strat_excess_daily, bench_excess_daily, ddof=1)
            bench_variance = float(np.var(bench_excess_daily, ddof=1))
            beta = float(cov_matrix[0, 1] / (bench_variance + 1e-8)) if bench_variance > 0 else 1.0
            alpha_daily = float(strat_excess_daily.mean() - beta * bench_excess_daily.mean())
            alpha_capm_regression = float(alpha_daily * 242.0)
        else:
            beta = 1.0
            alpha_capm_regression = 0.0

        alpha_cagr_approx = float(cagr - (self.rf + beta * (bench_cagr - self.rf)))

        # ---------------- 4. 订单与交易摩擦统计 ----------------
        filled_orders = orders_df[orders_df["status"].isin(["FILLED", "PARTIALLY_FILLED"])].copy() if not orders_df.empty else pd.DataFrame()
        total_filled_trades = len(filled_orders)

        total_commission = float(filled_orders["commission"].sum()) if not filled_orders.empty else 0.0
        total_stamp_tax = float(filled_orders["stamp_tax"].sum()) if not filled_orders.empty else 0.0
        total_transfer_fee = float(filled_orders["transfer_fee"].sum()) if not filled_orders.empty and "transfer_fee" in filled_orders.columns else 0.0
        total_slippage = float(filled_orders["slippage"].sum()) if not filled_orders.empty else 0.0
        total_trading_cost = total_commission + total_stamp_tax + total_transfer_fee + total_slippage

        # 年化换手率
        if not filled_orders.empty and "filled_shares" in filled_orders.columns:
            trade_turnover_amount = float((filled_orders["filled_shares"] * filled_orders["execution_price"]).sum())
            avg_nav = float(df["total_equity"].mean())
            annualized_turnover = float(trade_turnover_amount / (2.0 * avg_nav * years))
        else:
            annualized_turnover = 0.0

        # ---------------- 5. FIFO 真实批次买卖配对评价 ----------------
        pairs = closed_trades or orders_df.attrs.get("closed_trades", [])
        
        net_win_rate = 0.0
        gross_win_rate = 0.0
        profit_loss_ratio = 0.0
        gross_profit_loss_ratio = 0.0
        avg_holding_trading_days = 0.0

        if pairs:
            net_pnl_list = [p.get("net_realized_pnl_pct", p.get("realized_pnl_pct", 0.0)) for p in pairs]
            gross_pnl_list = [p.get("gross_realized_pnl_pct", p.get("realized_pnl_pct", 0.0)) for p in pairs]
            days_list = [p.get("holding_trading_days", p.get("holding_days", 1)) for p in pairs]

            net_pnl_s = pd.Series(net_pnl_list)
            gross_pnl_s = pd.Series(gross_pnl_list)

            # 净胜率 (扣除全部税费、佣金与分红后的 Total Return)
            net_wins = net_pnl_s[net_pnl_s > 0]
            net_losses = net_pnl_s[net_pnl_s < 0]
            net_win_rate = float((len(net_wins) / len(net_pnl_s)) * 100.0) if len(net_pnl_s) > 0 else 0.0
            avg_net_win = float(net_wins.mean()) if len(net_wins) > 0 else 0.0
            avg_net_loss = float(abs(net_losses.mean())) if len(net_losses) > 0 else 1e-6
            profit_loss_ratio = float(avg_net_win / avg_net_loss) if avg_net_loss > 0 else 0.0

            # 毛胜率 (仅原始价差)
            gross_wins = gross_pnl_s[gross_pnl_s > 0]
            gross_losses = gross_pnl_s[gross_pnl_s < 0]
            gross_win_rate = float((len(gross_wins) / len(gross_pnl_s)) * 100.0) if len(gross_pnl_s) > 0 else 0.0
            avg_gross_win = float(gross_wins.mean()) if len(gross_wins) > 0 else 0.0
            avg_gross_loss = float(abs(gross_losses.mean())) if len(gross_losses) > 0 else 1e-6
            gross_profit_loss_ratio = float(avg_gross_win / avg_gross_loss) if avg_gross_loss > 0 else 0.0

            avg_holding_trading_days = float(np.mean(days_list)) if days_list else 0.0
        else:
            avg_holding_trading_days = 0.0

        # ---------------- 6. 月度收益热力图矩阵 ----------------
        monthly_table = self._build_monthly_table(df)

        # ---------------- 7. 审计元数据 (Fail-Closed) ----------------
        if isinstance(audit_info, AuditMetadata):
            audit_meta = audit_info.to_dict()
        elif isinstance(audit_info, dict):
            audit_meta = audit_info
        else:
            audit_meta = AuditCollector.collect().to_dict()

        metrics = {
            "total_days": n_days,
            "cum_strategy_return": round(cum_strat_ret * 100.0, 2),
            "cum_benchmark_return": round(cum_bench_ret * 100.0, 2),
            "excess_return": round(excess_cum_ret * 100.0, 2),
            "cagr": round(cagr * 100.0, 2),
            "benchmark_cagr": round(bench_cagr * 100.0, 2),
            "alpha": round(alpha_capm_regression * 100.0, 2),
            "alpha_capm_regression": round(alpha_capm_regression * 100.0, 2),
            "alpha_cagr_approx": round(alpha_cagr_approx * 100.0, 2),
            "beta": round(beta, 2),
            "annualized_volatility": round(ann_vol * 100.0, 2),
            "max_drawdown": round(max_drawdown * 100.0, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "sortino_ratio": round(float(sortino), 2),
            "calmar_ratio": round(float(calmar), 2),
            "win_rate": round(net_win_rate, 2),
            "net_win_rate": round(net_win_rate, 2),
            "gross_win_rate": round(gross_win_rate, 2),
            "profit_loss_ratio": round(profit_loss_ratio, 2),
            "gross_profit_loss_ratio": round(gross_profit_loss_ratio, 2),
            "annualized_turnover": round(annualized_turnover, 2),
            "average_holding_days": round(avg_holding_trading_days, 1),
            "average_holding_trading_days": round(avg_holding_trading_days, 1),
            "total_trades": total_filled_trades,
            "closed_pair_trades": len(pairs),
            "total_commission": round(total_commission, 2),
            "total_stamp_tax": round(total_stamp_tax, 2),
            "total_transfer_fee": round(total_transfer_fee, 2),
            "total_slippage": round(total_slippage, 2),
            "total_costs": round(total_trading_cost, 2),
            "monthly_table": monthly_table,
            "audit_metadata": audit_meta
        }
        return metrics

    def _build_monthly_table(self, df: pd.DataFrame) -> Dict[str, Any]:
        """构建月度收益矩阵"""
        df = df.copy()
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        
        monthly_returns = df.groupby(["year", "month"])["strategy_return"].apply(
            lambda x: (np.prod(1.0 + x) - 1.0) * 100.0
        ).unstack()

        monthly_returns = monthly_returns.round(2)
        return monthly_returns.to_dict()

    @staticmethod
    def _monthly_table_to_pivot(table: Any) -> Optional[pd.DataFrame]:
        """
        将 monthly_table 归一化为「行=年份、列=1~12月」的透视表。
        兼容两种历史结构: 嵌套 {月: {年: 收益}} 与扁平 {(年, 月): 收益} / "YYYY-MM" 字符串键。
        无有效数据时返回 None。
        """
        import re as _re
        records = []
        if isinstance(table, dict):
            for k, v in table.items():
                try:
                    if isinstance(v, dict):          # 嵌套: 外层=月, 内层={年: 值}
                        month = int(str(k))
                        for yk, vv in v.items():
                            records.append((int(str(yk)), month, float(vv)))
                    elif isinstance(k, (tuple, list)) and len(k) >= 2:   # 扁平元组键
                        records.append((int(str(k[0])), int(str(k[1])), float(v)))
                    else:                            # 扁平字符串键 "(2024, 3)" 或 "2024-03"
                        m = _re.search(r"(\d{4})\D+(\d{1,2})", str(k))
                        if m:
                            records.append((int(m.group(1)), int(m.group(2)), float(v)))
                except (TypeError, ValueError):
                    continue
        if not records:
            return None
        pdf = pd.DataFrame(records, columns=["year", "month", "ret"])
        pivot = pdf.pivot_table(index="year", columns="month", values="ret", aggfunc="first")
        return pivot.reindex(columns=range(1, 13))

    # ==================== 回测产物落盘 (reports/) ====================

    def save_reports(
        self,
        equity_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        metrics: Dict[str, Any],
        output_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """
        将本次回测完整产物持久化到 reports/ 目录 (时间戳 + latest 双份):
        1. performance_*.json   全量绩效指标 + 审计元数据
        2. equity_curve_*.csv   逐日净值曲线 (含 NAV 归一化)
        3. orders_*.csv         订单流水明细
        4. monthly_heatmap_*.png 月度收益热力图
        单文件失败仅告警不中断，返回成功写入的 {名称: 路径} 字典。
        """
        out_dir = Path(output_dir) if output_dir else settings.REPORTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        saved: Dict[str, str] = {}

        def _dual_write(label: str, writer) -> None:
            for suffix in [f"{stamp}", "latest"]:
                target = out_dir / f"{label}_{suffix}{path_suffix[label]}"
                try:
                    writer(target)
                    saved[f"{label}({suffix})"] = str(target)
                except Exception as e:
                    logger.warning(f"写入报告文件失败 ({target.name}): {e}")

        path_suffix = {"performance": ".json", "equity_curve": ".csv", "orders": ".csv", "monthly_heatmap": ".png"}

        if not metrics:
            logger.warning("绩效指标为空，跳过 reports/ 落盘。")
            return saved

        # ---- 1. 绩效指标 JSON (monthly_table 归一化为 YYYY-MM 字符串键) ----
        json_metrics: Dict[str, Any] = {}
        monthly_pivot = self._monthly_table_to_pivot(metrics.get("monthly_table"))
        monthly_flat: Dict[str, float] = {}
        if monthly_pivot is not None:
            for y, row in monthly_pivot.iterrows():
                for m, val in row.items():
                    if pd.notna(val):  # 缺失月份不输出，避免 JSON 出现 NaN
                        monthly_flat[f"{int(y)}-{int(m):02d}"] = round(float(val), 4)
        for k, v in metrics.items():
            if k == "monthly_table":
                json_metrics[k] = monthly_flat
                continue
            if k == "audit_metadata" and not isinstance(v, dict):
                continue  # AuditMetadata 实体由 audit 层单独序列化
            json_metrics[k] = v

        def _write_performance(p: Path) -> None:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(json_metrics, f, ensure_ascii=False, indent=2, default=str)

        _dual_write("performance", _write_performance)

        # ---- 2. 净值曲线 CSV (附加 NAV 归一化与回撤列) ----
        def _write_equity(p: Path) -> None:
            eq = equity_df.copy()
            eq["date"] = pd.to_datetime(eq["date"]).dt.strftime("%Y-%m-%d")
            base_eq = float(eq["total_equity"].iloc[0]) or 1.0
            base_bm = float(eq["benchmark_equity"].iloc[0]) or 1.0
            eq["nav_strategy"] = eq["total_equity"] / base_eq
            eq["nav_benchmark"] = eq["benchmark_equity"] / base_bm
            cummax = eq["total_equity"].cummax()
            eq["drawdown_pct"] = ((eq["total_equity"] - cummax) / cummax * 100.0).round(4)
            eq.to_csv(p, index=False, encoding="utf-8-sig")

        _dual_write("equity_curve", _write_equity)

        # ---- 3. 订单流水 CSV ----
        if orders_df is not None and len(orders_df) > 0:
            _dual_write("orders", lambda p: orders_df.to_csv(p, index=False, encoding="utf-8-sig"))

        # ---- 4. 月度收益热力图 PNG (非交互 Agg 后端) ----
        def _write_heatmap(p: Path) -> None:
            pivot = monthly_pivot
            if pivot is None or pivot.empty:
                raise ValueError("无月度收益数据")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            # Windows/Linux 常见中文字体回退链，避免标题出现方框
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]

            rows, cols = pivot.shape
            fig_w, fig_h = max(6.0, cols * 0.75 + 3.5), max(3.0, rows * 0.5 + 2.0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            vmin, vmax = -8.0, 8.0
            data = pivot.values.astype(float)
            im = ax.imshow(np.clip(data, vmin, vmax), cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(cols), labels=[f"{int(c)}月" for c in pivot.columns], fontsize=8)
            ax.set_yticks(range(rows), labels=[str(int(r)) for r in pivot.index], fontsize=8)
            for r in range(rows):
                for c in range(cols):
                    v = data[r, c]
                    if np.isfinite(v):
                        ax.text(c, r, f"{v:+.1f}", ha="center", va="center",
                                fontsize=7, color="black")
            ax.set_title(f"月度收益率热力图 (%) | 累计超额 {metrics.get('excess_return', 0):+.2f}%",
                         fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(p, dpi=150)
            plt.close(fig)

        _dual_write("monthly_heatmap", _write_heatmap)

        if saved:
            logger.info(f"回测产物已落盘 reports/: 共 {len(saved)} 个文件 (含 latest 副本)")
        return saved
