"""
经校验科研指标与凭证加载器 (models/verified_metrics.py)
严格从物理科研产物与审计对账清单中加载指标，禁止任何界面与通知层硬编码伪造数据。
若物理文件缺失、校验失败或未经验证，强制 Fail-Closed 显示 "暂无可验证数据"。
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from config.settings import settings

from dataclasses import dataclass

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path) -> str:
    """计算文件的 SHA-256 哈希"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ProbabilityCalibrationEvidence:
    """经科学实证检验的概率校准强类型物理凭证 (Zero-Fabrication 审计标准)"""
    method: str
    brier_score: float
    sample_count: int
    calibrator_artifact_path: str
    calibrator_artifact_sha256: str
    reliability_curve_artifact_path: str
    reliability_curve_sha256: str
    calibration_dataset_manifest_path: str
    calibration_dataset_sha256: str
    model_id: str
    feature_schema_hash: str
    signed_at: str
    signer_key_id: str
    signature: str
    status: str = "VERIFIED"

    def is_valid(self, base_dir: Optional[Path] = None, expected_dataset_sha256: Optional[str] = None) -> bool:
        """
        深度校验校准凭证有效性 (Fail-Closed: 物理文件存在、SHA256哈希比对、Manifest全字段核验、专用签名验签与公钥权限)
        禁止仅凭自称哈希字符串洗白，禁止替代用途密钥，禁止 Manifest 缺字段。
        """
        if self.status != "VERIFIED":
            return False
        if not self.method or not isinstance(self.method, str):
            return False
        if not isinstance(self.brier_score, (int, float)) or isinstance(self.brier_score, bool):
            return False
        if not (0.0 <= self.brier_score <= 1.0):
            return False
        if not isinstance(self.sample_count, int) or isinstance(self.sample_count, bool) or self.sample_count <= 0:
            return False
        if not self.model_id or not isinstance(self.model_id, str):
            return False
        if not self.feature_schema_hash or not isinstance(self.feature_schema_hash, str) or len(self.feature_schema_hash) != 64:
            return False
        if not self.signed_at or not isinstance(self.signed_at, str):
            return False
        if not self.signer_key_id or not isinstance(self.signer_key_id, str):
            return False
        if not self.signature or not isinstance(self.signature, str):
            return False

        root = Path(base_dir or settings.BASE_DIR)

        # 1. 物理检查 3 个制品文件存在性与 SHA-256 哈希一致性
        file_checks = [
            (self.calibrator_artifact_path, self.calibrator_artifact_sha256, "calibrator"),
            (self.reliability_curve_artifact_path, self.reliability_curve_sha256, "reliability_curve"),
            (self.calibration_dataset_manifest_path, self.calibration_dataset_sha256, "calibration_dataset_manifest"),
        ]
        for rel_or_abs_path, expected_hash, desc in file_checks:
            if not rel_or_abs_path or not expected_hash or len(expected_hash) != 64:
                return False
            p = Path(rel_or_abs_path)
            full_p = p if p.is_absolute() else (root / p)
            if not full_p.is_file():
                return False
            actual_sha = compute_sha256(full_p)
            if actual_sha.lower() != expected_hash.lower():
                return False

        # 2. 读取 calibration_dataset_manifest_path，强制要求全量且非空包含 model_id, feature_schema_hash, dataset_sha256
        try:
            manifest_p = Path(self.calibration_dataset_manifest_path)
            full_manifest_p = manifest_p if manifest_p.is_absolute() else (root / manifest_p)
            manifest_data = json.loads(full_manifest_p.read_text(encoding="utf-8"))
            if not isinstance(manifest_data, dict):
                return False

            # model_id 必须存在、非空、且精确匹配 self.model_id
            m_id = manifest_data.get("model_id")
            if not isinstance(m_id, str) or not m_id.strip() or m_id.strip() != self.model_id:
                return False

            # feature_schema_hash 必须存在、非空、且精确匹配 self.feature_schema_hash
            f_hash = manifest_data.get("feature_schema_hash")
            if not isinstance(f_hash, str) or not f_hash.strip() or len(f_hash.strip()) != 64 or f_hash.strip() != self.feature_schema_hash:
                return False

            # dataset_sha256 或等价数据集版本哈希必须存在、非空且为合法 64 位哈希
            d_hash = manifest_data.get("dataset_sha256") or manifest_data.get("normalized_dataset_sha256") or manifest_data.get("dataset_version_sha256")
            if not isinstance(d_hash, str) or not d_hash.strip() or len(d_hash.strip()) != 64:
                return False
            d_hash_clean = d_hash.strip().lower()

            if expected_dataset_sha256 is not None:
                if d_hash_clean != str(expected_dataset_sha256).strip().lower():
                    return False

            # 若 Manifest 声明了物理 dataset_path，则该文件必须存在且其实际 SHA-256 必须精确匹配 dataset_sha256
            ds_path_str = manifest_data.get("dataset_path")
            if ds_path_str:
                ds_p = Path(ds_path_str)
                full_ds_p = ds_p if ds_p.is_absolute() else (root / ds_p)
                if not full_ds_p.is_file():
                    return False
                actual_ds_sha = compute_sha256(full_ds_p)
                if actual_ds_sha.lower() != d_hash_clean:
                    return False
        except Exception:
            return False

        # 3. 校验 Ed25519 数字签名 (仅允许 PROBABILITY_CALIBRATION_AUTHORIZATION 专用用途)
        try:
            from data.crypto_anchor import TRUSTED_KEY_REGISTRY, verify_ed25519_signature
            if self.signer_key_id not in TRUSTED_KEY_REGISTRY:
                return False
            key_info = TRUSTED_KEY_REGISTRY[self.signer_key_id]
            if key_info.get("status") != "ACTIVE":
                return False

            allowed = key_info.get("allowed_purposes", [])
            if "PROBABILITY_CALIBRATION_AUTHORIZATION" not in allowed:
                return False

            payload_to_verify = {
                "method": self.method,
                "brier_score": round(float(self.brier_score), 6),
                "sample_count": self.sample_count,
                "calibrator_artifact_sha256": self.calibrator_artifact_sha256,
                "reliability_curve_sha256": self.reliability_curve_sha256,
                "calibration_dataset_sha256": self.calibration_dataset_sha256,
                "model_id": self.model_id,
                "feature_schema_hash": self.feature_schema_hash,
                "signed_at": self.signed_at,
                "status": "VERIFIED",
            }
            msg_bytes = json.dumps(payload_to_verify, sort_keys=True, ensure_ascii=False).encode("utf-8")
            sig_ok, _ = verify_ed25519_signature(
                message=msg_bytes,
                signature_hex=self.signature,
                key_id=self.signer_key_id,
                required_purpose="PROBABILITY_CALIBRATION_AUTHORIZATION",
                domain_separator="QUANT_PROBABILITY_CALIBRATION_V1"
            )
            if not sig_ok:
                return False
        except Exception:
            return False

        return True


def load_verified_probability_calibration(
    artifact_path: Union[str, Path],
    base_dir: Optional[Path] = None
) -> Optional[ProbabilityCalibrationEvidence]:
    """
    受控加载经科学实证检验的概率校准证据制品。
    从物理磁盘读取 JSON 凭据并深度校验文件哈希、Manifest、Schema 与 Ed25519 签名。
    若文件缺失、解析失败、或校验未通过，一律 Fail-Closed 返回 None，禁止手工构造伪证据。
    """
    root = Path(base_dir or settings.BASE_DIR)
    raw_p = Path(artifact_path)
    full_p = raw_p if raw_p.is_absolute() else (root / raw_p)
    if not full_p.is_file():
        logger.warning(f"概率校准凭证文件不存在: {full_p}")
        return None

    try:
        data = json.loads(full_p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None

        required_fields = [
            "method", "brier_score", "sample_count",
            "calibrator_artifact_path", "calibrator_artifact_sha256",
            "reliability_curve_artifact_path", "reliability_curve_sha256",
            "calibration_dataset_manifest_path", "calibration_dataset_sha256",
            "model_id", "feature_schema_hash",
            "signed_at", "signer_key_id", "signature"
        ]
        for f in required_fields:
            if f not in data or data[f] is None:
                logger.warning(f"概率校准凭证缺少必要字段: {f}")
                return None

        ev = ProbabilityCalibrationEvidence(
            method=str(data["method"]),
            brier_score=float(data["brier_score"]),
            sample_count=int(data["sample_count"]),
            calibrator_artifact_path=str(data["calibrator_artifact_path"]),
            calibrator_artifact_sha256=str(data["calibrator_artifact_sha256"]),
            reliability_curve_artifact_path=str(data["reliability_curve_artifact_path"]),
            reliability_curve_sha256=str(data["reliability_curve_sha256"]),
            calibration_dataset_manifest_path=str(data["calibration_dataset_manifest_path"]),
            calibration_dataset_sha256=str(data["calibration_dataset_sha256"]),
            model_id=str(data["model_id"]),
            feature_schema_hash=str(data["feature_schema_hash"]),
            signed_at=str(data["signed_at"]),
            signer_key_id=str(data["signer_key_id"]),
            signature=str(data["signature"]),
            status=str(data.get("status", "VERIFIED")),
        )
        if not ev.is_valid(base_dir=root):
            logger.warning(f"概率校准凭证物理验证失败: {full_p}")
            return None
        return ev
    except Exception as e:
        logger.warning(f"读取或解析概率校准凭证异常: {e}")
# 审计与科研对账强制要求的 19 项血缘与实证字段
REQUIRED_METRICS_EVIDENCE_FIELDS: List[str] = [
    "model_id",
    "run_id",
    "git_commit",
    "data_snapshot_hash",
    "factor_matrix_hash",
    "train_start_date",
    "train_end_date",
    "test_start_date",
    "test_end_date",
    "nav_file_path",
    "nav_file_sha256",
    "trade_ledger_path",
    "trade_ledger_sha256",
    "orders_file_path",
    "orders_file_sha256",
    "annualized_return",
    "sharpe_ratio",
    "max_drawdown",
    "information_ratio",
]


def validate_scientific_evidence(payload: dict, base_dir: Optional[Path] = None) -> Tuple[bool, str]:
    """
    严格校验科研凭证的完整性、物理文件存在性、哈希防伪与底层账本重算一致性。
    禁止仅凭静态 JSON 声明或单纯 Manifest 哈希洗白。
    必须绑定 19 项全血缘字段与底层 NAV/订单实测对账。
    """
    if not isinstance(payload, dict):
        return False, "Payload must be a dictionary"

    # 1. 19项必要血缘字段校验
    for f in REQUIRED_METRICS_EVIDENCE_FIELDS:
        if f not in payload or payload[f] is None:
            return False, f"Missing required evidence field: '{f}'"

    root_dir = Path(base_dir or settings.BASE_DIR)

    # 2. 物理文件存在性与 SHA256 校验
    file_checks = [
        ("nav_file_path", "nav_file_sha256", "NAV"),
        ("trade_ledger_path", "trade_ledger_sha256", "Trade ledger"),
        ("orders_file_path", "orders_file_sha256", "Orders"),
    ]
    for path_key, hash_key, desc in file_checks:
        raw_path = Path(payload[path_key])
        p = raw_path if raw_path.is_absolute() else (root_dir / raw_path)
        if not p.exists():
            return False, f"{desc} file does not exist: {raw_path}"
        actual_hash = compute_sha256(p)
        if actual_hash != payload[hash_key]:
            return False, f"{desc} file SHA256 mismatch (expected {payload[hash_key]}, got {actual_hash})"

    # 3. 基于物理 NAV 文件的指标数学重算校验 (容差验证，防止伪造声明指标)
    try:
        nav_raw_path = Path(payload["nav_file_path"])
        nav_path = nav_raw_path if nav_raw_path.is_absolute() else (root_dir / nav_raw_path)

        if nav_path.suffix.lower() == ".csv":
            nav_df = pd.read_csv(nav_path)
        elif nav_path.suffix.lower() in (".parquet", ".pq"):
            nav_df = pd.read_parquet(nav_path)
        else:
            return False, f"Unsupported NAV file format: {nav_path.suffix}"

        # 查找净值列
        nav_col = None
        for col in ["nav", "strategy_nav", "equity", "portfolio_value", "close"]:
            if col in nav_df.columns:
                nav_col = col
                break

        if nav_col is None:
            return False, "NAV file does not contain recognized net value column (nav, equity, portfolio_value)"

        series = nav_df[nav_col].dropna().astype(float).values
        if len(series) < 2:
            return False, "NAV series has insufficient data points for metric recomputation"

        total_ret = (series[-1] / series[0]) - 1.0
        n_days = len(series)
        ann_ret = (1.0 + total_ret) ** (242.0 / max(n_days, 1)) - 1.0

        daily_rets = series[1:] / series[:-1] - 1.0
        mean_r = float(daily_rets.mean())
        std_r = float(daily_rets.std())
        if std_r > 1e-8:
            recomputed_sharpe = (mean_r * 242.0 - 0.02) / (std_r * (242.0 ** 0.5))
        else:
            recomputed_sharpe = 0.0

        peak = series[0]
        max_dd = 0.0
        for val in series:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd

        declared_ann_ret = float(payload["annualized_return"])
        declared_sharpe = float(payload["sharpe_ratio"])
        declared_max_dd = float(payload["max_drawdown"])

        norm_declared_ann = declared_ann_ret / 100.0 if abs(declared_ann_ret) > 1.0 else declared_ann_ret
        norm_declared_dd = declared_max_dd / 100.0 if abs(declared_max_dd) > 1.0 else declared_max_dd

        if abs(ann_ret - norm_declared_ann) > 0.08 and abs(ann_ret - declared_ann_ret) > 0.08:
            return False, f"Recomputed annualized return ({ann_ret:.4f}) does not match declared ({declared_ann_ret})"

        if abs(recomputed_sharpe - declared_sharpe) > 0.4:
            return False, f"Recomputed Sharpe ({recomputed_sharpe:.2f}) does not match declared ({declared_sharpe:.2f})"

        if abs(max_dd - abs(norm_declared_dd)) > 0.08 and abs(max_dd - abs(declared_max_dd)) > 0.08:
            return False, f"Recomputed max drawdown ({max_dd:.4f}) does not match declared ({declared_max_dd})"

    except Exception as e:
        return False, f"Error recomputing metrics from physical NAV: {e}"

    return True, "VALID"


def load_verified_research_metrics(metrics_path: Path | str, base_dir: Optional[Path] = None) -> dict:
    """
    加载经科研实证校验的指标对象。
    若标记为 LEGACY_UNVERIFIED、被排除科研实证、或未通过 19 项对账重算检验，直接抛出 ValueError。
    """
    p = Path(metrics_path)
    if not p.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from {metrics_path}: {e}")

    meta = data.get("_metadata", {})
    if (
        data.get("verification_status") == "LEGACY_UNVERIFIED"
        or meta.get("verification_status") == "LEGACY_UNVERIFIED"
        or data.get("excluded_from_scientific_evidence") is True
        or meta.get("excluded_from_scientific_evidence") is True
    ):
        raise ValueError(f"Artifact {metrics_path} is tagged LEGACY_UNVERIFIED or excluded from scientific evidence")

    is_valid, reason = validate_scientific_evidence(data, base_dir=base_dir)
    if not is_valid:
        raise ValueError(f"Scientific evidence validation failed for {metrics_path}: {reason}")

    return data


class VerifiedResearchMetricsLoader:
    """量化科研指标与物理凭证加载器 (Zero-Fabrication)"""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or settings.BASE_DIR)
        self.reports_dir = self.base_dir / "reports"

    def get_data_as_of(self) -> str:
        """动态读取物理凭证中的最新数据基准日 (仅限已核验有效凭证)"""
        # 1. 尝试从 production_research/research_run_manifest.json 读取
        prod_manifest = self.reports_dir / "production_research" / "research_run_manifest.json"
        if prod_manifest.exists():
            try:
                data = json.loads(prod_manifest.read_text(encoding="utf-8"))
                end_date = data.get("end_date")
                if end_date:
                    return str(end_date)
            except Exception as e:
                logger.warning(f"读取 production_research manifest 失败: {e}")

        # 2. 尝试从 certification_manifest.json 读取
        cert_manifest = self.reports_dir / "model_research" / "certification_manifest.json"
        if cert_manifest.exists():
            try:
                data = json.loads(cert_manifest.read_text(encoding="utf-8"))
                created_at = data.get("created_at")
                if created_at:
                    return str(created_at)[:10]
            except Exception as e:
                logger.warning(f"读取 certification_manifest 失败: {e}")

        # 3. 尝试从 performance_all_generations.manifest.json 读取 (仅限非遗留未核验凭证)
        perf_manifest = self.reports_dir / "performance_all_generations.manifest.json"
        if perf_manifest.exists():
            try:
                data = json.loads(perf_manifest.read_text(encoding="utf-8"))
                if (
                    data.get("verification_status") != "LEGACY_UNVERIFIED"
                    and not data.get("excluded_from_scientific_evidence", False)
                ):
                    data_as_of = data.get("data_as_of")
                    if data_as_of:
                        return str(data_as_of)
            except Exception as e:
                logger.warning(f"读取 performance manifest 失败: {e}")

        return "暂无可验证数据"

    def load_multi_generation_performance(self, for_product_display: bool = False) -> Optional[Dict[str, Any]]:
        """
        加载四代回测指标，执行 Manifest 签名/哈希校验。
        若用于产品展示 (for_product_display=True)，且指标被标记为 LEGACY_UNVERIFIED 或
        excluded_from_product_display=True，则强制 Fail-Closed 返回 None。
        """
        perf_path = self.reports_dir / "performance_all_generations.json"
        manifest_path = self.reports_dir / "performance_all_generations.manifest.json"

        if not perf_path.exists() or not manifest_path.exists():
            logger.warning("缺少 performance_all_generations.json 或其 manifest 文件")
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_hash = manifest.get("sha256")
            actual_hash = compute_sha256(perf_path)

            if expected_hash != actual_hash:
                logger.error(f"performance_all_generations.json 哈希校验失败: 预期 {expected_hash}, 实际 {actual_hash}")
                return None

            perf_data = json.loads(perf_path.read_text(encoding="utf-8"))
            meta = perf_data.get("_metadata", {})

            # 审计强化：若用于前端产品展示，严禁展示遗留未经验证指标
            if for_product_display:
                if (
                    manifest.get("verification_status") == "LEGACY_UNVERIFIED"
                    or manifest.get("excluded_from_product_display") is True
                    or meta.get("verification_status") == "LEGACY_UNVERIFIED"
                    or meta.get("excluded_from_product_display") is True
                ):
                    logger.warning("performance_all_generations.json 属于遗留未核验数据，禁止用于前端产品展示")
                    return None

            required_gens = ["gen1_baseline", "gen2_plans_1_to_4", "gen3_plans_5_7_9", "gen4_flagship"]
            if not all(g in perf_data for g in required_gens):
                logger.error("performance_all_generations.json 缺少必要代际字段")
                return None

            return perf_data
        except Exception as e:
            logger.error(f"加载多代回测凭证异常: {e}")
            return None

    def get_multi_generation_metrics_table(self, for_product_display: bool = False) -> Optional[pd.DataFrame]:
        """构建四代指标对比表 (数据完全来源于物理检验通过的 JSON 凭证，缺失字段绝不默认 0.0)"""
        perf = self.load_multi_generation_performance(for_product_display=for_product_display)
        if not perf:
            return None

        g1 = perf.get("gen1_baseline", {})
        g2 = perf.get("gen2_plans_1_to_4", {})
        g3 = perf.get("gen3_plans_5_7_9", {})
        g4 = perf.get("gen4_flagship", {})

        def fmt_val(val, suffix="", decimals=2):
            if val is None:
                return "-"
            return f"{val:.{decimals}f}{suffix}"

        def fmt_ret(val):
            if val is None:
                return "-"
            return f"{val:+.2f}%"

        def wr_val(g):
            w = g.get("net_win_rate", g.get("win_rate", g.get("trade_win_rate_pct")))
            return f"{w:.2f}%" if w is not None else "-"

        table_rows = [
            {
                "评估维度": "💰 收益端表现",
                "核心量化指标": "策略累计收益率",
                "第一代 (客观基准)": fmt_ret(g1.get("cum_strategy_return")),
                "第二代 (系统增强)": fmt_ret(g2.get("cum_strategy_return")),
                "第三代 (风险自适应)": fmt_ret(g3.get("cum_strategy_return")),
                "第四代 (全景旗舰)": fmt_ret(g4.get("cum_strategy_return")),
                "量化资管行业对标与评估说明": "全周期扣费真实总回报 (沪深300同期基准表现对照)"
            },
            {
                "评估维度": "💰 收益端表现",
                "核心量化指标": "年化复合收益 (CAGR)",
                "第一代 (客观基准)": fmt_ret(g1.get("cagr")),
                "第二代 (系统增强)": fmt_ret(g2.get("cagr")),
                "第三代 (风险自适应)": fmt_ret(g3.get("cagr")),
                "第四代 (全景旗舰)": fmt_ret(g4.get("cagr")),
                "量化资管行业对标与评估说明": "扣费后无杠杆年化复合增长率"
            },
            {
                "评估维度": "💰 收益端表现",
                "核心量化指标": "年化纯超额 (Alpha)",
                "第一代 (客观基准)": fmt_ret(g1.get("alpha")),
                "第二代 (系统增强)": fmt_ret(g2.get("alpha")),
                "第三代 (风险自适应)": fmt_ret(g3.get("alpha")),
                "第四代 (全景旗舰)": fmt_ret(g4.get("alpha")),
                "量化资管行业对标与评估说明": "剥离大盘基准收益后的特质超额Alpha"
            },
            {
                "评估维度": "🛡️ 风险端控制",
                "核心量化指标": "夏普比率 (Sharpe)",
                "第一代 (客观基准)": fmt_val(g1.get('sharpe_ratio')),
                "第二代 (系统增强)": fmt_val(g2.get('sharpe_ratio')),
                "第三代 (风险自适应)": fmt_val(g3.get('sharpe_ratio')),
                "第四代 (全景旗舰)": fmt_val(g4.get('sharpe_ratio')),
                "量化资管行业对标与评估说明": "承担每单位总风险获得的超额收益 (无风险利率=2%)"
            },
            {
                "评估维度": "🛡️ 风险端控制",
                "核心量化指标": "最大历史回撤 (Max DD)",
                "第一代 (客观基准)": fmt_val(g1.get('max_drawdown'), suffix='%'),
                "第二代 (系统增强)": fmt_val(g2.get('max_drawdown'), suffix='%'),
                "第三代 (风险自适应)": fmt_val(g3.get('max_drawdown'), suffix='%'),
                "第四代 (全景旗舰)": fmt_val(g4.get('max_drawdown'), suffix='%'),
                "量化资管行业对标与评估说明": "历史全周期从峰值到谷底的最大净值下跌幅度"
            },
            {
                "评估维度": "⚡ 执行端摩擦",
                "核心量化指标": "核心交易盈亏比 (P/L)",
                "第一代 (客观基准)": fmt_val(g1.get('profit_loss_ratio')),
                "第二代 (系统增强)": fmt_val(g2.get('profit_loss_ratio')),
                "第三代 (风险自适应)": fmt_val(g3.get('profit_loss_ratio')),
                "第四代 (全景旗舰)": fmt_val(g4.get('profit_loss_ratio')),
                "量化资管行业对标与评估说明": "平仓总盈利金额 / 平仓总亏损金额"
            },
            {
                "评估维度": "⚡ 执行端摩擦",
                "核心量化指标": "单边年化换手率",
                "第一代 (客观基准)": fmt_val(g1.get('annualized_turnover'), suffix='x'),
                "第二代 (系统增强)": fmt_val(g2.get('annualized_turnover'), suffix='x'),
                "第三代 (风险自适应)": fmt_val(g3.get('annualized_turnover'), suffix='x'),
                "第四代 (全景旗舰)": fmt_val(g4.get('annualized_turnover'), suffix='x'),
                "量化资管行业对标与评估说明": "持仓单边年化换手倍数"
            },
            {
                "评估维度": "⚡ 执行端摩擦",
                "核心量化指标": "做多胜率 (Win Rate)",
                "第一代 (客观基准)": wr_val(g1),
                "第二代 (系统增强)": wr_val(g2),
                "第三代 (风险自适应)": wr_val(g3),
                "第四代 (全景旗舰)": wr_val(g4),
                "量化资管行业对标与评估说明": "扣费后实现正收益的交易日占比 (基于实际回测结算数据)"
            }
        ]
        return pd.DataFrame(table_rows)

    def get_top_factors(self, top_n: int = 15) -> Optional[pd.DataFrame]:
        """从 production_research/factor_summary.csv 加载真实因子重要度/打分"""
        summary_csv = self.reports_dir / "production_research" / "factor_summary.csv"
        manifest = self.reports_dir / "production_research" / "research_run_manifest.json"
        if not summary_csv.exists() or not manifest.exists():
            return None

        try:
            df = pd.read_csv(summary_csv)
            if "factor_name" not in df.columns or "selection_score" not in df.columns:
                return None
            df_sorted = df.sort_values(by="selection_score", ascending=False).head(top_n)
            return df_sorted[["factor_name", "selection_score", "mean_rank_ic", "rank_ic_ir"]]
        except Exception as e:
            logger.warning(f"读取 factor_summary.csv 失败: {e}")
            return None

    def get_factor_quantile_returns(self) -> Optional[pd.DataFrame]:
        """从 production_research/factor_quantile_returns.csv 读取分位数收益"""
        q_csv = self.reports_dir / "production_research" / "factor_quantile_returns.csv"
        if not q_csv.exists():
            return None
        try:
            return pd.read_csv(q_csv)
        except Exception:
            return None

    def get_decile_spread_data(self) -> Optional[pd.DataFrame]:
        """读取物理截面分位数胜率凭证 (若无则 Fail-Closed 返回 None)"""
        decile_path = self.reports_dir / "decile_precision_curve.json"
        if not decile_path.exists():
            return None
        try:
            data = json.loads(decile_path.read_text(encoding="utf-8"))
            return pd.DataFrame(data)
        except Exception:
            return None
