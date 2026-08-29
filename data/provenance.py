"""
数据血缘与证据链认证引擎 (data/provenance.py)
实现严格的 Single-Source-of-Truth、Fail-Closed、Anti-Tampering、Anti-Impersonation 数据血缘与证据链检验。
核心原则：
1. NO EVIDENCE => NO VERIFIED
2. UNKNOWN != VERIFIED
3. SHA256 VERIFIED != SOURCE AUTHENTICATED (哈希仅证明未篡改，不证明来自官方)
4. FILE EXISTS != OFFICIAL SOURCE (任意 CSV 放入 raw/ 绝不自动成为官方)
5. TEST FIXTURE != PRODUCTION EVIDENCE
6. MANIFEST CLAIM != VERIFICATION
7. CALLER PROVIDED TRUE != VERIFICATION
8. COVERAGE CLAIM != ACTUAL BACKTEST COVERAGE
"""
import re
import json
import hashlib
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SourceClass(str, Enum):
    """数据源证据资质等级分类"""
    OFFICIAL_PRIMARY = "OFFICIAL_PRIMARY"       # 交易所/指数公司官方一手数据 (必须具有可核验的 Source Metadata)
    LICENSED_VENDOR = "LICENSED_VENDOR"         # 持牌专业金融数据终端 (如 Wind/Choice/彭博原始导出)
    THIRD_PARTY = "THIRD_PARTY"                 # 开源/第三方抓取源 (如 AKShare/TuShare 实时接口)
    TEST_FIXTURE = "TEST_FIXTURE"               # 单元测试专用固定 Fixture (仅用于代码能力验证，绝不可生产认证)
    SYNTHETIC = "SYNTHETIC"                     # 模拟/算法生成的测试数据 (严禁进入生产认证)
    UNKNOWN = "UNKNOWN"                         # 未知来源 (Fail-Closed 直接拒绝)

    @classmethod
    def is_production_eligible(cls, source_class: Union[str, "SourceClass"]) -> bool:
        """只有官方一手或持牌终端来源允许参与最高等级 VERIFIED 认证"""
        val = str(source_class).upper()
        if "." in val:
            val = val.split(".")[-1]
        return val in [cls.OFFICIAL_PRIMARY.value, cls.LICENSED_VENDOR.value]


class DataProvenanceError(Exception):
    """数据血缘或凭证校验异常 (Fail-Closed)"""
    pass


@dataclass
class SourceEvidenceMetadata:
    """Raw Evidence 独立来源元数据实体 (P0-1 Source Authentication)"""
    source_class: SourceClass = SourceClass.UNKNOWN
    source_name: str = ""
    source_url: Optional[str] = None
    source_reference: Optional[str] = None
    published_at: Optional[str] = None
    downloaded_at: Optional[str] = None
    original_filename: str = ""
    sha256: str = ""
    evidence_type: str = "INDEX_CONSTITUENT_ADJUSTMENT"  # or "BASELINE_SNAPSHOT"

    @classmethod
    def load_and_verify(cls, meta_path: Path, raw_file_path: Path) -> Tuple[Optional["SourceEvidenceMetadata"], List[str]]:
        """从 .source.json 文件加载并与实体 Raw 文件做双向校验"""
        errors = []
        if not meta_path.exists():
            return None, [f"missing_source_metadata_{meta_path.name}"]

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return None, [f"corrupted_source_metadata_{meta_path.name}_{str(e)}"]

        s_class_str = str(data.get("source_class", "UNKNOWN")).upper()
        try:
            s_class = SourceClass(s_class_str)
        except Exception:
            s_class = SourceClass.UNKNOWN

        s_name = str(data.get("source_name", "")).strip()
        s_url = data.get("source_url")
        s_ref = data.get("source_reference")
        orig_fn = str(data.get("original_filename", "")).strip()
        expected_sha = str(data.get("sha256", "")).strip().lower()
        ev_type = str(data.get("evidence_type", "INDEX_CONSTITUENT_ADJUSTMENT")).strip()

        if not s_name:
            errors.append("source_metadata_missing_source_name")
        if not s_url and not s_ref:
            errors.append("source_metadata_missing_source_url_or_reference")
        if orig_fn and orig_fn != raw_file_path.name:
            errors.append(f"source_metadata_filename_mismatch_{orig_fn}_vs_{raw_file_path.name}")

        if raw_file_path.exists():
            h = hashlib.sha256()
            with open(raw_file_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            actual_sha = h.hexdigest().lower()
            if expected_sha and actual_sha != expected_sha:
                errors.append(f"source_metadata_sha256_mismatch_{actual_sha}_vs_{expected_sha}")
        else:
            errors.append(f"raw_evidence_file_missing_{raw_file_path.name}")

        if s_class == SourceClass.UNKNOWN:
            errors.append("source_metadata_has_unknown_source_class")

        meta = cls(
            source_class=s_class,
            source_name=s_name,
            source_url=s_url,
            source_reference=s_ref,
            published_at=data.get("published_at"),
            downloaded_at=data.get("downloaded_at"),
            original_filename=orig_fn or raw_file_path.name,
            sha256=expected_sha or actual_sha,
            evidence_type=ev_type
        )
        return meta, errors


@dataclass
class UniverseVerificationResult:
    """股票池动态时点血缘认证结果 (严格由运行时计算得出，禁止直接注入 True)"""
    is_valid: bool = False
    source_class: SourceClass = SourceClass.UNKNOWN
    provenance_verified: bool = False
    raw_hash_verified: bool = False
    dataset_hash_verified: bool = False
    coverage_verified: bool = False
    source_verified: bool = False
    baseline_verified: bool = False
    event_integrity_verified: bool = False
    survivorship_bias_risk: bool = True
    mode: str = "STATIC_FALLBACK"
    failed_checks: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_production_verified(self) -> bool:
        """是否达成生产级 VERIFIED 最高认证"""
        return (
            self.is_valid
            and self.provenance_verified
            and self.raw_hash_verified
            and self.dataset_hash_verified
            and self.coverage_verified
            and self.source_verified
            and self.baseline_verified
            and self.event_integrity_verified
            and not self.survivorship_bias_risk
            and len(self.failed_checks) == 0
        )


class UniverseParserAdapter(ABC):
    """Raw Evidence 解析适配器抽象基类 (分离数据解析与来源认证)"""

    @abstractmethod
    def parse(self, raw_path: Path, source_meta: SourceEvidenceMetadata) -> List[Dict[str, Any]]:
        pass


class CSIRebalanceAnnouncementParser(UniverseParserAdapter):
    """中证指数调样公告文件解析适配器"""

    def parse(self, raw_path: Path, source_meta: SourceEvidenceMetadata) -> List[Dict[str, Any]]:
        if raw_path.suffix == ".csv":
            df = pd.read_csv(raw_path)
        elif raw_path.suffix == ".parquet":
            df = pd.read_parquet(raw_path)
        else:
            with open(raw_path, "r", encoding="utf-8") as f:
                df = pd.DataFrame(json.load(f))

        records = []
        for _, row in df.iterrows():
            records.append({
                "index_code": str(row.get("index_code", "000300")).strip().zfill(6),
                "effective_date": pd.to_datetime(row["effective_date"]).strftime("%Y-%m-%d"),
                "symbol": str(row["symbol"]).strip().upper(),
                "action": str(row["action"]).strip().upper(),
                "source_class": source_meta.source_class.value,
                "source_name": source_meta.source_name,
                "source_file": raw_path.name,
                "source_sha256": source_meta.sha256,
                "parser_version": "3.0"
            })
        return records


class CSIConstituentSnapshotParser(UniverseParserAdapter):
    """中证指数基线成分股完整快照解析适配器"""

    def parse(self, raw_path: Path, source_meta: SourceEvidenceMetadata) -> List[Dict[str, Any]]:
        if raw_path.suffix == ".csv":
            df = pd.read_csv(raw_path)
        elif raw_path.suffix == ".parquet":
            df = pd.read_parquet(raw_path)
        else:
            with open(raw_path, "r", encoding="utf-8") as f:
                df = pd.DataFrame(json.load(f))

        records = []
        snap_date = pd.to_datetime(df["date"].iloc[0] if "date" in df.columns else df["effective_date"].iloc[0]).strftime("%Y-%m-%d")
        for _, row in df.iterrows():
            sym = str(row["symbol"]).strip().upper()
            records.append({
                "index_code": str(row.get("index_code", "000300")).strip().zfill(6),
                "effective_date": snap_date,
                "symbol": sym,
                "action": "IN",
                "source_class": source_meta.source_class.value,
                "source_name": source_meta.source_name,
                "source_file": raw_path.name,
                "source_sha256": source_meta.sha256,
                "parser_version": "3.0"
            })
        return records


class ProvenanceVerifier:
    """
    点位股票池 (Point-In-Time) 证据链与数据血缘密码级校验器
    必须真正校验 Raw Evidence 磁盘文件、独立 Source Metadata、SHA256 哈希与时点因果性。
    """

    A_SHARE_SYMBOL_PATTERN = re.compile(
        r"^(?:(?:60\d{4}|688\d{3}|689\d{3})\.SH|(?:00\d{4}|300\d{3}|301\d{3})\.SZ|(?:8\d{5}|4\d{5}|920\d{3})\.BJ)$"
    )

    @classmethod
    def compute_file_sha256(cls, file_path: Path) -> str:
        """计算单个文件的 SHA256 哈希指纹"""
        if not file_path.exists():
            raise FileNotFoundError(f"Raw evidence file not found: {file_path}")
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def compute_dataframe_sha256(cls, df: pd.DataFrame) -> str:
        """确定性计算 DataFrame 的全局内容哈希 (流式逐行哈希组合)"""
        if df.empty:
            return hashlib.sha256(b"EMPTY_DATAFRAME").hexdigest()
        sorted_df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
        h = hashlib.sha256()
        schema_str = ",".join(f"{col}:{sorted_df[col].dtype}" for col in sorted_df.columns)
        h.update(schema_str.encode("utf-8"))
        for col in sorted_df.columns:
            s_bytes = sorted_df[col].astype(str).str.encode("utf-8").values
            for val_b in s_bytes:
                h.update(val_b)
        return h.hexdigest()

    @classmethod
    def verify_pit_universe(
        cls,
        normalized_df: Optional[pd.DataFrame],
        manifest_data: Optional[Dict[str, Any]],
        raw_evidence_dir: Optional[Path] = None,
        actual_backtest_start_date: Optional[Union[str, pd.Timestamp]] = None,
        actual_backtest_end_date: Optional[Union[str, pd.Timestamp]] = None,
        backtest_start_date: Optional[Union[str, pd.Timestamp]] = None,
        backtest_end_date: Optional[Union[str, pd.Timestamp]] = None,
        is_test_environment: bool = False
    ) -> UniverseVerificationResult:
        """
        执行全链路严格数据血缘审计检查：
        1. 检查 Raw Evidence 实体文件及对应的 .source.json 独立来源元数据
        2. 严格核验 source_class，严禁 Parser 自作主张提升资质
        3. 核验 Baseline Snapshot 独立证据，拒绝由第一条调样事件自动猜测
        4. 严格对比 actual_backtest_start / actual_backtest_end 覆盖范围
        5. Normalized 数据集 SHA256 精确对比
        6. Schema、标的代码格式、同日冲突 IN/OUT 校验
        """
        failed_checks: List[str] = []
        details: Dict[str, Any] = {}

        # 统一真实回测开始/结束日期
        act_start = actual_backtest_start_date or backtest_start_date
        act_end = actual_backtest_end_date or backtest_end_date

        if normalized_df is None or normalized_df.empty:
            failed_checks.append("normalized_pit_dataset_missing_or_empty")
            return UniverseVerificationResult(
                is_valid=False,
                survivorship_bias_risk=True,
                mode="STATIC_FALLBACK",
                failed_checks=failed_checks
            )

        if manifest_data is None or not isinstance(manifest_data, dict):
            failed_checks.append("manifest_missing_or_invalid")
            return UniverseVerificationResult(
                is_valid=False,
                survivorship_bias_risk=True,
                mode="STATIC_FALLBACK",
                failed_checks=failed_checks
            )

        raw_source_class_str = str(manifest_data.get("source_class", "UNKNOWN")).upper()
        try:
            manifest_source_class = SourceClass(raw_source_class_str)
        except Exception:
            manifest_source_class = SourceClass.UNKNOWN

        # 1. 检查 Raw Evidence 实体文件与独立 Source Metadata
        raw_evidence_hashes = manifest_data.get("raw_evidence_hashes", {})
        raw_files = manifest_data.get("source_files", [])
        raw_hash_verified = False
        source_verified = False
        resolved_source_class = manifest_source_class

        if not raw_files or not raw_evidence_hashes:
            failed_checks.append("no_raw_evidence_files_recorded_in_manifest")
        else:
            all_raw_ok = True
            all_source_meta_ok = True
            if raw_evidence_dir and raw_evidence_dir.exists():
                for r_file, expected_hash in raw_evidence_hashes.items():
                    r_path = raw_evidence_dir / r_file if not Path(r_file).is_absolute() else Path(r_file)
                    if not r_path.exists():
                        failed_checks.append(f"raw_evidence_file_missing_{r_file}")
                        all_raw_ok = False
                        continue
                    actual_hash = cls.compute_file_sha256(r_path)
                    if actual_hash.lower() != expected_hash.lower():
                        failed_checks.append(f"raw_evidence_hash_mismatch_{r_file}")
                        all_raw_ok = False

                    # P0-1: 检查对应独立的 .source.json 来源描述文件
                    meta_file = r_path.with_name(f"{r_path.name}.source.json")
                    source_meta, meta_errs = SourceEvidenceMetadata.load_and_verify(meta_file, r_path)
                    if meta_errs or source_meta is None:
                        failed_checks.extend(meta_errs)
                        all_source_meta_ok = False
                    else:
                        if not SourceClass.is_production_eligible(source_meta.source_class):
                            failed_checks.append(f"source_metadata_ineligible_for_production_{source_meta.source_class.value}")
                            all_source_meta_ok = False
                        resolved_source_class = source_meta.source_class

                raw_hash_verified = all_raw_ok
                source_verified = all_source_meta_ok and SourceClass.is_production_eligible(resolved_source_class)
            else:
                failed_checks.append("raw_evidence_directory_not_accessible")
                raw_hash_verified = False
                source_verified = False

        if not source_verified:
            if resolved_source_class == SourceClass.TEST_FIXTURE:
                failed_checks.append("source_class_is_test_fixture_not_production_eligible")
            elif resolved_source_class == SourceClass.SYNTHETIC:
                failed_checks.append("source_class_is_synthetic_strictly_forbidden")
            else:
                failed_checks.append(f"source_class_ineligible_for_production_{resolved_source_class.value}")

        details["source_class"] = resolved_source_class.value

        # 2. Normalized 数据集 SHA256 校验
        expected_dataset_sha256 = manifest_data.get("normalized_dataset_sha256")
        actual_dataset_sha256 = cls.compute_dataframe_sha256(normalized_df)
        dataset_hash_verified = bool(expected_dataset_sha256 and expected_dataset_sha256.lower() == actual_dataset_sha256.lower())
        if not dataset_hash_verified:
            failed_checks.append("normalized_dataset_sha256_mismatch")
        details["actual_dataset_sha256"] = actual_dataset_sha256
        details["expected_dataset_sha256"] = expected_dataset_sha256

        # 3. Schema 与事件合法性校验
        required_cols = {"effective_date", "symbol", "action"}
        if not required_cols.issubset(set(normalized_df.columns)):
            failed_checks.append("normalized_df_missing_required_columns")
            event_integrity_verified = False
        else:
            invalid_actions = set(normalized_df["action"].unique()) - {"IN", "OUT"}
            if invalid_actions:
                failed_checks.append(f"invalid_actions_found_{list(invalid_actions)}")

            symbols = normalized_df["symbol"].astype(str).str.strip().str.upper()
            invalid_syms = [s for s in symbols.unique() if not cls.A_SHARE_SYMBOL_PATTERN.match(s)]
            if invalid_syms:
                failed_checks.append(f"invalid_a_share_symbol_format_{invalid_syms[:5]}")

            conflict_grp = normalized_df.groupby(["effective_date", "symbol"])["action"].nunique()
            conflicts = conflict_grp[conflict_grp > 1]
            if not conflicts.empty:
                failed_checks.append(f"conflicting_in_out_events_on_same_day_count_{len(conflicts)}")

            event_integrity_verified = (
                not invalid_actions
                and not invalid_syms
                and conflicts.empty
            )

        # 4. Baseline 基线快照独立证据校验 (P1-1: 拒绝由调样事件推导)
        baseline_date = manifest_data.get("baseline_snapshot_date")
        baseline_symbols = manifest_data.get("baseline_symbols", [])
        baseline_file = manifest_data.get("baseline_snapshot_file")
        baseline_verified = False

        if not baseline_date or not baseline_symbols:
            failed_checks.append("baseline_snapshot_missing_in_manifest")
        elif not baseline_file:
            failed_checks.append("baseline_snapshot_independent_raw_file_missing_in_manifest")
        else:
            b_date_ts = pd.to_datetime(baseline_date)
            b_invalid = [s for s in baseline_symbols if not cls.A_SHARE_SYMBOL_PATTERN.match(str(s).strip().upper())]
            if b_invalid:
                failed_checks.append(f"baseline_contains_invalid_symbols_{b_invalid[:3]}")

            if len(baseline_symbols) < 250 and resolved_source_class != SourceClass.TEST_FIXTURE:
                failed_checks.append(f"baseline_symbol_count_insufficient_for_index_{len(baseline_symbols)}")

            if act_start:
                bt_start_ts = pd.to_datetime(act_start)
                if b_date_ts > bt_start_ts:
                    failed_checks.append(f"baseline_date_{baseline_date}_is_after_backtest_start_{act_start}")
                else:
                    baseline_verified = len(b_invalid) == 0 and (len(baseline_symbols) >= 250 or resolved_source_class == SourceClass.TEST_FIXTURE)
            else:
                baseline_verified = len(b_invalid) == 0 and (len(baseline_symbols) >= 250 or resolved_source_class == SourceClass.TEST_FIXTURE)

        # 5. 时间覆盖窗口校验 (P0-3: 必须精确覆盖真实回测起止点)
        cov_start = manifest_data.get("coverage_start")
        cov_end = manifest_data.get("coverage_end")
        coverage_verified = False

        if not cov_start or not cov_end:
            failed_checks.append("coverage_dates_missing_in_manifest")
        else:
            c_start_ts = pd.to_datetime(cov_start)
            c_end_ts = pd.to_datetime(cov_end)

            cov_ok = True
            if act_start:
                bt_start_ts = pd.to_datetime(act_start)
                if c_start_ts > bt_start_ts:
                    failed_checks.append(f"coverage_start_{cov_start}_after_actual_backtest_start_{act_start}")
                    cov_ok = False
            if act_end:
                bt_end_ts = pd.to_datetime(act_end)
                if c_end_ts < bt_end_ts:
                    failed_checks.append(f"coverage_end_{cov_end}_before_actual_backtest_end_{act_end}")
                    cov_ok = False

            today_ts = pd.Timestamp.now().normalize()
            if c_end_ts > today_ts + pd.Timedelta(days=180) and resolved_source_class != SourceClass.TEST_FIXTURE:
                failed_checks.append(f"unsubstantiated_future_coverage_claimed_{cov_end}")
                cov_ok = False

            coverage_verified = cov_ok

        provenance_verified = (
            source_verified
            and raw_hash_verified
            and dataset_hash_verified
        )

        survivorship_risk = not (
            provenance_verified
            and baseline_verified
            and coverage_verified
            and event_integrity_verified
        )

        mode = "POINT_IN_TIME_VERIFIED" if not survivorship_risk else "STATIC_FALLBACK"

        return UniverseVerificationResult(
            is_valid=(len(failed_checks) == 0),
            source_class=resolved_source_class,
            provenance_verified=provenance_verified,
            raw_hash_verified=raw_hash_verified,
            dataset_hash_verified=dataset_hash_verified,
            coverage_verified=coverage_verified,
            source_verified=source_verified,
            baseline_verified=baseline_verified,
            event_integrity_verified=event_integrity_verified,
            survivorship_bias_risk=survivorship_risk,
            mode=mode,
            failed_checks=failed_checks,
            details=details
        )
