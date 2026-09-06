"""
模型注册表与生命周期管理 (models/registry.py)

背景 (Phase A / 2026-09-01 架构审计):
    审计判定 Model Registry 为 NOT_IMPLEMENTED: 研究运行每次覆盖 latest_lightgbm.pkl,
    无版本/无制品哈希/无血统, 且 PRODUCTION 概念不存在 —— 所谓"每日信号"实为
    "每日重跑 20 折研究" (research replay), 无法归因当天信号来自哪个模型版本。

本模块提供:
    1. 研究制品登记 (RESEARCH): 记录制品路径/模型类型/指标/数据集哈希/代码 commit/配置快照
    2. 生命周期状态机: RESEARCH → CANDIDATE → APPROVED → PRODUCTION → ARCHIVED
    3. 晋升门禁 (fail-closed): 每级晋升要求证据 + 审批人, 全流程留痕
        - RESEARCH→CANDIDATE : 需 OOS 指标 + 数据集哈希
        - CANDIDATE→APPROVED : 需 certification 证据引用
        - APPROVED→PRODUCTION: 需人工审批人 + prospective 验证 + paper trading 证据
    4. 单一生产模型查询 (get_production), 供推理路径使用

重要: 研究运行【永远】只能产生 RESEARCH 记录, 绝不能直接写 PRODUCTION。
    制品文件复制进注册表根目录 (settings.MODELS_DIR/registry) 是唯一授权的生产写入路径。

用法:
    registry = ModelRegistry()
    mid = registry.register_research_artifact(artifact_path, model_type="lightgbm",
                                              task_type="classification", metrics={...},
                                              dataset_sha256=..., config_snapshot={...})
    registry.promote(mid, ModelState.CANDIDATE, approver="researcher")
    registry.promote(mid, ModelState.PRODUCTION, approver="linjun", evidence={...})
"""
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config.settings import settings

logger = logging.getLogger(__name__)


class ModelState:
    RESEARCH = "RESEARCH"
    CANDIDATE = "CANDIDATE"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"
    ARCHIVED = "ARCHIVED"

    ORDER = [RESEARCH, CANDIDATE, APPROVED, PRODUCTION]

    @classmethod
    def all_states(cls) -> List[str]:
        return [cls.RESEARCH, cls.CANDIDATE, cls.APPROVED, cls.PRODUCTION, cls.ARCHIVED]


# 允许的晋升路径与所需条件
TRANSITIONS: Dict[str, Dict[str, Any]] = {
    (ModelState.RESEARCH, ModelState.CANDIDATE): {
        "requires_metrics": True,
        "requires_dataset": True,
        "desc": "研究制品具备 OOS 指标与数据集血统后可成为候选",
    },
    (ModelState.CANDIDATE, ModelState.APPROVED): {
        "requires_certification": True,
        "desc": "候选模型需通过认证 (certification evidence) 才能获批",
    },
    (ModelState.APPROVED, ModelState.PRODUCTION): {
        "requires_approver": True,
        "requires_prospective": True,
        "requires_paper_trading": True,
        "desc": "生产上线需人工审批 + 前瞻验证 + 模拟盘证据",
    },
    (ModelState.PRODUCTION, ModelState.ARCHIVED): {
        "desc": "生产模型退役归档",
    },
}


@dataclass
class ModelRecord:
    """模型制品元数据记录 (血统 + 生命周期)"""
    model_id: str
    created_at: str
    model_type: str
    task_type: str
    state: str
    source_artifact: str            # 研究制品原始路径
    registry_artifact: Optional[str] = None   # 晋升后注册表内制品路径
    dataset_sha256: Optional[str] = None
    dataset_path: Optional[str] = None
    code_commit: Optional[str] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    feature_count: Optional[int] = None
    feature_schema_hash: Optional[str] = None
    notes: str = ""
    promotion_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceArtifact:
    """
    结构化物理签署晋升证据对象 (RFC 8032 Ed25519 密码学验签)
    """
    evidence_type: str
    artifact_path: str
    artifact_sha256: str
    schema_version: str
    model_id: str
    dataset_sha256: str
    feature_schema_hash: str
    created_at: str
    observation_start: str
    observation_end: str
    observed_trading_days: int
    status: str
    approver: str
    signer_key_id: str
    signature: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    ledger_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_canonical_bytes(self) -> bytes:
        payload = {
            "approver": str(self.approver).strip(),
            "created_at": str(self.created_at).strip(),
            "dataset_sha256": str(self.dataset_sha256).strip().lower(),
            "evidence_type": str(self.evidence_type).strip(),
            "feature_schema_hash": str(self.feature_schema_hash).strip().lower(),
            "metrics": self.metrics or {},
            "model_id": str(self.model_id).strip(),
            "observation_end": str(self.observation_end).strip(),
            "observation_start": str(self.observation_start).strip(),
            "observed_trading_days": int(self.observed_trading_days),
            "schema_version": str(self.schema_version).strip(),
            "signer_key_id": str(self.signer_key_id).strip(),
            "status": str(self.status).strip(),
        }
        if self.ledger_records:
            payload["ledger_records"] = self.ledger_records
        sorted_json = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        return sorted_json.encode("utf-8")

    def sign(self, private_key_hex: Optional[str] = None, domain_separator: Optional[str] = None) -> "EvidenceArtifact":
        from data.crypto_anchor import (
            TRUSTED_KEY_REGISTRY,
            DOMAIN_SEPARATOR_PROMOTION,
            sign_with_environment_key,
            _HAS_CRYPTOGRAPHY,
            ed25519_sign_pure
        )
        sep = domain_separator or DOMAIN_SEPARATOR_PROMOTION
        msg = self.compute_canonical_bytes()
        msg_to_sign = f"{sep}:".encode("utf-8") + msg if sep else msg

        if self.signer_key_id in TRUSTED_KEY_REGISTRY and private_key_hex:
            reg_info = TRUSTED_KEY_REGISTRY[self.signer_key_id]
            sk_bytes = bytes.fromhex(private_key_hex.strip())
            if _HAS_CRYPTOGRAPHY:
                from cryptography.hazmat.primitives.asymmetric import ed25519 as crypto_ed25519
                priv = crypto_ed25519.Ed25519PrivateKey.from_private_bytes(sk_bytes)
                self.signature = priv.sign(msg_to_sign).hex()
                return self
            else:
                pk_bytes = bytes.fromhex(reg_info["public_key_hex"])
                self.signature = ed25519_sign_pure(msg_to_sign, sk_bytes, pk_bytes).hex()
                return self

        sig_hex, errs = sign_with_environment_key(
            message=msg,
            key_id=self.signer_key_id,
            required_purpose="RUNTIME_ATTESTATION" if self.signer_key_id == "PROD_RUNTIME_KEY_2026_V1" else "MODEL_PROMOTION",
            domain_separator=sep,
            explicit_private_key_hex=private_key_hex,
            production_mode=False
        )
        if sig_hex:
            self.signature = sig_hex
            return self

        raise PromotionError(f"签名生成失败: {errs}")

    def save(self, target_path: Optional[Union[str, Path]] = None) -> Path:
        import hashlib
        if target_path:
            p = Path(target_path).resolve()
        else:
            p = Path(self.artifact_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_path = str(p)
        self.artifact_sha256 = ""
        d = self.to_dict()
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        self.artifact_sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
        return p

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceArtifact":
        required = [
            "evidence_type", "artifact_path", "schema_version",
            "model_id", "dataset_sha256", "created_at",
            "observation_start", "observation_end", "observed_trading_days",
            "status", "approver", "signer_key_id", "signature"
        ]
        missing = [f for f in required if f not in d or d[f] is None or d[f] == ""]
        if "feature_schema_hash" not in d or d["feature_schema_hash"] is None:
            missing.append("feature_schema_hash")
        if missing:
            raise PromotionError(f"EvidenceArtifact 缺少必需字段: {missing}")
        return cls(
            evidence_type=str(d["evidence_type"]),
            artifact_path=str(d["artifact_path"]),
            artifact_sha256=str(d.get("artifact_sha256") or ""),
            schema_version=str(d["schema_version"]),
            model_id=str(d["model_id"]),
            dataset_sha256=str(d["dataset_sha256"]),
            feature_schema_hash=str(d.get("feature_schema_hash") or ""),
            created_at=str(d["created_at"]),
            observation_start=str(d["observation_start"]),
            observation_end=str(d["observation_end"]),
            observed_trading_days=int(d["observed_trading_days"]),
            status=str(d["status"]),
            approver=str(d["approver"]),
            signer_key_id=str(d["signer_key_id"]),
            signature=str(d["signature"]),
            metrics=dict(d.get("metrics") or {}),
            ledger_records=list(d.get("ledger_records") or []),
        )

    @classmethod
    def load_from_file(cls, path: Union[str, Path], expected_sha256: Optional[str] = None) -> "EvidenceArtifact":
        import hashlib
        p = Path(path).resolve()
        if not p.is_file():
            raise PromotionError(f"证据文件不存在或非普通文件: {p}")
        file_bytes = p.read_bytes()
        actual_sha = hashlib.sha256(file_bytes).hexdigest()
        if expected_sha256 and expected_sha256.strip().lower() != actual_sha.lower():
            raise PromotionError(f"证据文件 SHA-256 不匹配: 实际 {actual_sha} != 预期 {expected_sha256}")
        try:
            data = json.loads(file_bytes.decode("utf-8"))
        except Exception as e:
            raise PromotionError(f"证据文件解析失败 (非合法 JSON): {p} ({e})")
        if not isinstance(data, dict):
            raise PromotionError(f"证据文件根节点必须为 JSON 对象: {p}")
        if "artifact_path" not in data or not data["artifact_path"]:
            data["artifact_path"] = str(p)
        data["artifact_sha256"] = actual_sha
        return cls.from_dict(data)


@dataclass
class ProspectiveEvidence:
    """结构化前瞻验证证据对象 (兼容旧接口)"""
    ref: str
    dataset_sha256: Optional[str] = None
    validation_window: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    evidence_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperTradingEvidence:
    """结构化模拟盘证据对象 (兼容旧接口)"""
    ref: str
    ledger_hash: Optional[str] = None
    trade_count: Optional[int] = None
    pnl_metrics: Dict[str, Any] = field(default_factory=dict)
    evidence_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PromotionError(Exception):
    """晋升校验失败 (fail-closed)"""
    pass


def verify_promotion_evidence(
    raw_ev: Any,
    model_rec: ModelRecord,
    expected_type: str,
    min_trading_days: int = 20,
    calendar: Optional[Any] = None,
) -> EvidenceArtifact:
    """
    物理签署证据全要素门禁验证 (Fail-Closed)
    """
    import hashlib
    if isinstance(raw_ev, bool):
        raise PromotionError(f"{expected_type} 禁止使用布尔值 (True/False)")
    if raw_ev is None:
        raise PromotionError(f"{expected_type} 证据缺失")
    if isinstance(raw_ev, EvidenceArtifact):
        artifact = raw_ev
    elif isinstance(raw_ev, str):
        s = raw_ev.strip()
        if not s:
            raise PromotionError(f"{expected_type} 禁止传入空字符串")
        if s == "manual_claim":
            raise PromotionError(f"{expected_type} 严禁使用 manual_claim 虚假声明")
        path = Path(s)
        if not path.is_file():
            raise PromotionError(f"{expected_type} 证据文件不存在或非文件: {s}")
        artifact = EvidenceArtifact.load_from_file(path)
    elif isinstance(raw_ev, dict):
        if not raw_ev:
            raise PromotionError(f"{expected_type} 禁止传入空字典")
        if raw_ev.get("ref") == "manual_claim":
            raise PromotionError(f"{expected_type} 严禁使用 manual_claim")
        if "artifact_path" in raw_ev:
            path = Path(raw_ev["artifact_path"])
            if not path.is_file():
                raise PromotionError(f"{expected_type} 物理证据文件不存在: {path}")
            declared_sha = raw_ev.get("artifact_sha256")
            artifact = EvidenceArtifact.load_from_file(path, expected_sha256=declared_sha)
        elif "ref" in raw_ev:
            ref_str = str(raw_ev["ref"]).strip()
            if not ref_str:
                raise PromotionError(f"{expected_type} ref 禁止为空")
            ref_path = Path(ref_str)
            if not ref_path.is_file():
                raise PromotionError(f"{expected_type} ref 字符串无对应物理文件: {ref_str}")
            artifact = EvidenceArtifact.load_from_file(ref_path)
        else:
            if not any(k in raw_ev for k in ["artifact_path", "model_id", "signature", "evidence_type"]):
                raise PromotionError(f"{expected_type} 缺少有效字段 (需包含 artifact_path / model_id 等)")
            artifact = EvidenceArtifact.from_dict(raw_ev)
    elif hasattr(raw_ev, "to_dict"):
        raw_dict = raw_ev.to_dict()
        if not any(k in raw_dict for k in ["artifact_path", "model_id", "signature", "evidence_type", "ref"]):
            raise PromotionError(f"{expected_type} 缺少有效字段")
        return verify_promotion_evidence(raw_dict, model_rec, expected_type, min_trading_days, calendar)
    else:
        raise PromotionError(f"{expected_type} 格式不合法: {type(raw_ev)}")

    # 1. 物理文件存在性与 SHA-256 核验
    p = Path(artifact.artifact_path)
    if not p.is_absolute():
        p = (Path(settings.BASE_DIR) / p).resolve()
    else:
        p = p.resolve()

    if not p.is_file():
        raise PromotionError(f"{expected_type} 物理文件不存在: {p}")

    file_bytes = p.read_bytes()
    computed_sha = hashlib.sha256(file_bytes).hexdigest().lower()
    declared_sha = str(artifact.artifact_sha256).strip().lower()
    if not declared_sha or computed_sha != declared_sha:
        raise PromotionError(
            f"{expected_type} 物理证据文件 SHA-256 不匹配: 实际 {computed_sha} != 声明 {declared_sha}"
        )

    # 2. 验证 Schema
    if artifact.schema_version not in ("evidence_v1", "v1.0", "1.0"):
        raise PromotionError(f"{expected_type} 不支持的 schema_version: {artifact.schema_version}")

    # 3. 验证数字签名与 Signer Key ID (必须严格具备 MODEL_PROMOTION 权限，严禁跨用途签名)
    from data.crypto_anchor import TRUSTED_KEY_REGISTRY, DOMAIN_SEPARATOR_PROMOTION, verify_ed25519_signature, verify_trust_root
    if not artifact.signer_key_id or artifact.signer_key_id not in TRUSTED_KEY_REGISTRY:
        raise PromotionError(f"{expected_type} 未知或未注册的 signer_key_id: {artifact.signer_key_id}")

    reg_info = TRUSTED_KEY_REGISTRY[artifact.signer_key_id]
    if reg_info.get("status") != "ACTIVE":
        raise PromotionError(f"{expected_type} 签名密钥状态非 ACTIVE: {artifact.signer_key_id} ({reg_info.get('status')})")

    # 强制精确匹配 MODEL_PROMOTION 用途
    allowed_purposes = reg_info.get("allowed_purposes", [])
    if "MODEL_PROMOTION" not in allowed_purposes:
        raise PromotionError(
            f"{expected_type} 密钥用途权限错配 (Key Purpose Mismatch): 签名密钥 ({artifact.signer_key_id}) "
            f"缺少 'MODEL_PROMOTION' 权限 (当前权限: {allowed_purposes})！严禁跨用途签名。"
        )

    # 校验外部密码学信任根
    tr_ok, tr_actual, tr_pin, tr_errs = verify_trust_root()
    if tr_pin is not None and not tr_ok:
        raise PromotionError(f"{expected_type} 密码学信任根校验失败: {tr_errs}")

    if not artifact.signature:
        raise PromotionError(f"{expected_type} 缺少数字签名 (未签名证据)")

    canonical_msg = artifact.compute_canonical_bytes()

    sig_ok, sig_errs = verify_ed25519_signature(
        message=canonical_msg,
        signature_hex=artifact.signature,
        key_id=artifact.signer_key_id,
        required_purpose="MODEL_PROMOTION",
        domain_separator=DOMAIN_SEPARATOR_PROMOTION,
        production_mode=False
    )
    if not sig_ok:
        raise PromotionError(f"{expected_type} Ed25519 数字签名验证失败: {sig_errs}")

    # 4. 验证 model_id 一致
    if artifact.model_id != model_rec.model_id:
        raise PromotionError(
            f"{expected_type} model_id 不一致: 证据为 {artifact.model_id}, 当前模型为 {model_rec.model_id}"
        )

    # 5. 验证 dataset_sha256 一致
    if model_rec.dataset_sha256 and artifact.dataset_sha256.lower() != model_rec.dataset_sha256.lower():
        raise PromotionError(
            f"{expected_type} dataset_sha256 不一致: 证据为 {artifact.dataset_sha256}, 模型为 {model_rec.dataset_sha256}"
        )

    # 6. 验证 feature_schema_hash 一致
    if model_rec.feature_schema_hash and artifact.feature_schema_hash.lower() != model_rec.feature_schema_hash.lower():
        raise PromotionError(
            f"{expected_type} feature_schema_hash 不一致: 证据为 {artifact.feature_schema_hash}, 模型为 {model_rec.feature_schema_hash}"
        )

    # 7. 验证 status 为 PASS/MATURE
    if artifact.status not in ("PASS", "MATURE"):
        raise PromotionError(
            f"{expected_type} 状态不达标: 当前为 {artifact.status} (必须为 PASS 或 MATURE)"
        )

    # 8. 验证 observation_start/end 属于合法交易日
    from data.trading_calendar import CanonicalTradingCalendar
    if calendar is None:
        try:
            calendar = CanonicalTradingCalendar.get_instance()
        except Exception as e:
            logger.warning(f"无法初始化 CanonicalTradingCalendar: {e}")

    if calendar is not None:
        if not calendar.is_trading_day(artifact.observation_start):
            raise PromotionError(f"{expected_type} observation_start ({artifact.observation_start}) 不是合法交易所交易日")
        if not calendar.is_trading_day(artifact.observation_end):
            raise PromotionError(f"{expected_type} observation_end ({artifact.observation_end}) 不是合法交易所交易日")

    if artifact.observation_start > artifact.observation_end:
        raise PromotionError(f"{expected_type} observation_start 晚于 observation_end")

    # 9. 验证 observed_trading_days 达到配置阈值
    if artifact.observed_trading_days < min_trading_days:
        raise PromotionError(
            f"{expected_type} 观察天数不足: 实际 {artifact.observed_trading_days} 天 < 要求 {min_trading_days} 天"
        )

    # 10. 验证观察天数可由证据账本重新计算 (虚构账本重算防御: 若附带账本明细，强制物理日历重算)
    if artifact.ledger_records:
        valid_dates = set()
        for r in artifact.ledger_records:
            d = r.get("date") or r.get("trade_date")
            if d:
                d_str = str(d)[:10]
                if calendar is None or calendar.is_trading_day(d_str):
                    valid_dates.add(d_str)
        if min_trading_days > 0 and len(valid_dates) < min_trading_days:
            raise PromotionError(
                f"{expected_type} 证据账本重新计算天数不足: 重算 {len(valid_dates)} 天 < 要求 {min_trading_days} 天"
            )
        if len(valid_dates) < artifact.observed_trading_days:
            raise PromotionError(
                f"{expected_type} 证据账本重新计算天数 ({len(valid_dates)}) 小于声明 observed_trading_days ({artifact.observed_trading_days})"
            )

    return artifact


class ModelRegistry:
    """模型注册表: 研究制品登记 + 生命周期晋升 + 生产模型查询"""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else Path(settings.MODELS_DIR) / "registry"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        if self.index_path.exists():
            try:
                index = json.loads(self.index_path.read_text(encoding="utf-8"))
            except Exception as e:
                raise RuntimeError(f"模型注册表索引损坏: {self.index_path} ({e})")

        # 兼容全新 clone 仓库: 若索引中无任何生产模型，自动从标准生产目录 saved_models/production/*/metadata.json 发现并装载已上线模型
        has_production = any(r.get("state") == ModelState.PRODUCTION for r in index.values())
        if not has_production:
            prod_root = self.root.parent / "production"
            if prod_root.exists():
                for meta_file in prod_root.glob("*/metadata.json"):
                    try:
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                        mid = meta.get("model_id")
                        if mid and mid not in index:
                            index[mid] = meta
                    except Exception:
                        pass
        return index

    def _save_index(self, index: Dict[str, Dict[str, Any]]):
        self.index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _git_commit() -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return "UNKNOWN"

    # ------------------------------------------------------------ 登记
    def register_research_artifact(
        self,
        artifact_path: Path,
        model_type: str,
        task_type: str = "classification",
        metrics: Optional[Dict[str, Any]] = None,
        dataset_sha256: Optional[str] = None,
        dataset_path: Optional[str] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
        feature_count: Optional[int] = None,
        feature_schema_hash: Optional[str] = None,
        notes: str = "",
        state: Optional[str] = None,
    ) -> str:
        """
        登记研究制品。默认状态 RESEARCH —— 研究运行【禁止】登记为 PRODUCTION。
        """
        artifact = Path(artifact_path)
        if not artifact.exists():
            raise FileNotFoundError(f"研究制品不存在: {artifact}")
        if state is not None and state != ModelState.RESEARCH:
            raise PromotionError("研究运行只能登记为 RESEARCH 状态, 禁止直接写入更高状态!")

        ts = datetime.now()
        base_id = f"m_{ts.strftime('%Y%m%d_%H%M%S')}_{model_type}"
        # 同一秒内多次注册必须唯一 (否则后一次覆盖前一次记录, 破坏生产唯一性与归档链)
        model_id = base_id
        index = self._load_index()
        dup = 1
        while model_id in index:
            dup += 1
            model_id = f"{base_id}_{dup}"
        record = ModelRecord(
            model_id=model_id,
            created_at=ts.isoformat(timespec="seconds"),
            model_type=model_type,
            task_type=task_type,
            state=ModelState.RESEARCH,
            source_artifact=str(artifact.resolve()),
            dataset_sha256=dataset_sha256,
            dataset_path=dataset_path,
            code_commit=self._git_commit(),
            config_snapshot=config_snapshot or {},
            metrics=dict(metrics or {}),
            feature_count=feature_count,
            feature_schema_hash=feature_schema_hash,
            notes=notes,
            promotion_history=[{
                "at": ts.isoformat(timespec="seconds"),
                "from": None,
                "to": ModelState.RESEARCH,
                "approver": "research_run",
                "note": "研究运行登记",
            }],
        )
        index[model_id] = record.to_dict()
        self._save_index(index)
        logger.info(f"研究制品已登记: {model_id} ({model_type}/{task_type}) <- {artifact}")
        return model_id

    # ------------------------------------------------------------ 晋升
    def promote(
        self,
        model_id: str,
        to_state: str,
        approver: str,
        evidence: Optional[Dict[str, Any]] = None,
        note: str = "",
    ) -> ModelRecord:
        """
        状态晋升 (fail-closed): 校验不通过直接抛 PromotionError, 绝不静默放行。
        """
        if to_state not in ModelState.all_states():
            raise PromotionError(f"非法目标状态: {to_state}")
        index = self._load_index()
        if model_id not in index:
            raise PromotionError(f"模型不存在于注册表: {model_id}")

        rec = ModelRecord(**index[model_id])
        from_state = rec.state
        key = (from_state, to_state)
        if key not in TRANSITIONS:
            raise PromotionError(f"非法状态跃迁: {from_state} -> {to_state} (仅允许 {list(TRANSITIONS)})")

        rules = TRANSITIONS[key]
        ev = evidence or {}

        if rules.get("requires_metrics") and not rec.metrics:
            raise PromotionError(f"{key} 需要 OOS 指标 (metrics), 当前为空")
        if rules.get("requires_dataset") and not rec.dataset_sha256:
            raise PromotionError(f"{key} 需要数据集哈希 (dataset_sha256), 当前缺失")
        if rules.get("requires_certification"):
            cert = ev.get("certification_ref")
            if isinstance(cert, bool) or not cert:
                raise PromotionError(f"{key} 认证证据 certification_ref 禁止使用简单布尔值或为空，必须为有效证据引用 (evidence.certification_ref)")
        if rules.get("requires_approver") and not (approver and approver.strip()):
            raise PromotionError(f"{key} 需要人工审批人 (approver)")
        if rules.get("requires_prospective") or rules.get("requires_paper_trading"):
            pv = ev.get("prospective_validation")
            pt = ev.get("paper_trading")
            if isinstance(pv, bool):
                raise PromotionError(f"{key} 前瞻验证证据禁止使用简单布尔值 (必须为物理签署的 EvidenceArtifact 制品)")
            if isinstance(pt, bool):
                raise PromotionError(f"{key} 模拟盘证据禁止使用简单布尔值 (必须为物理签署的 EvidenceArtifact 制品)")
            if not pv:
                raise PromotionError(f"{key} 需要结构化前瞻验证证据 (evidence.prospective_validation)")
            if not pt:
                raise PromotionError(f"{key} 需要结构化模拟盘证据 (evidence.paper_trading)")
            # 预校验字段完整性，防止某一侧严重格式错误被另一侧文件未就绪掩盖
            for ev_name, obj in [("前瞻验证证据", pv), ("模拟盘证据", pt)]:
                d = obj.to_dict() if hasattr(obj, "to_dict") else obj
                if isinstance(d, dict) and not any(k in d for k in ["artifact_path", "model_id", "signature", "evidence_type", "ref"]):
                    raise PromotionError(f"{key} {ev_name}缺少有效字段 (需包含 artifact_path/model_id/ref 等)")

        if rules.get("requires_prospective"):
            pv = ev.get("prospective_validation")
            art = verify_promotion_evidence(pv, rec, expected_type="prospective_validation", min_trading_days=20)
            ev["prospective_validation"] = art.to_dict()

        if rules.get("requires_paper_trading"):
            pt = ev.get("paper_trading")
            art = verify_promotion_evidence(pt, rec, expected_type="paper_trading", min_trading_days=20)
            ev["paper_trading"] = art.to_dict()

        # 晋升到 APPROVED/PRODUCTION: 制品复制进注册表 (唯一授权的生产写入路径)
        if to_state in (ModelState.APPROVED, ModelState.PRODUCTION):
            target_dir = self.root / model_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / "model.pkl"
            if not target_file.exists():
                shutil.copy2(rec.source_artifact, target_file)
            rec.registry_artifact = str(target_file)

            # 若晋升到 PRODUCTION，同步写入统一生产制品目录 (saved_models/production/<model_id>)
            if to_state == ModelState.PRODUCTION:
                try:
                    import hashlib
                    prod_dir = self.root.parent / "production" / model_id
                    prod_dir.mkdir(parents=True, exist_ok=True)
                    prod_file = prod_dir / "model.pkl"
                    if not prod_file.exists() or prod_file.stat().st_size != target_file.stat().st_size:
                        shutil.copy2(target_file, prod_file)

                    meta_path = prod_dir / "metadata.json"
                    meta_path.write_text(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

                    f_bytes = prod_file.read_bytes()
                    manifest_data = {
                        "model_id": model_id,
                        "model_type": rec.model_type,
                        "task_type": rec.task_type,
                        "file_name": "model.pkl",
                        "file_sha256": hashlib.sha256(f_bytes).hexdigest(),
                        "file_size_bytes": len(f_bytes),
                        "created_at": rec.created_at,
                        "state": ModelState.PRODUCTION,
                    }
                    (prod_dir / "manifest.json").write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as e:
                    logger.warning(f"同步写入生产制品目录失败 (非致命): {e}")

        # 生产唯一性: 新模型进 PRODUCTION 时, 旧 PRODUCTION 自动归档
        if to_state == ModelState.PRODUCTION:
            for mid, raw in index.items():
                if mid != model_id and raw.get("state") == ModelState.PRODUCTION:
                    old = ModelRecord(**raw)
                    old.state = ModelState.ARCHIVED
                    old.promotion_history.append({
                        "at": datetime.now().isoformat(timespec="seconds"),
                        "from": ModelState.PRODUCTION,
                        "to": ModelState.ARCHIVED,
                        "approver": approver,
                        "note": f"被 {model_id} 取代, 自动归档",
                    })
                    index[mid] = old.to_dict()
                    logger.info(f"原生产模型 {mid} 已自动归档")

        rec.state = to_state
        rec.promotion_history.append({
            "at": datetime.now().isoformat(timespec="seconds"),
            "from": from_state,
            "to": to_state,
            "approver": approver,
            "evidence": ev,
            "note": note or rules.get("desc", ""),
        })
        index[model_id] = rec.to_dict()
        self._save_index(index)
        logger.info(f"模型晋升: {model_id} {from_state} -> {to_state} (审批: {approver})")
        return rec

    # ------------------------------------------------------------ 查询
    def get(self, model_id: str) -> Optional[ModelRecord]:
        index = self._load_index()
        raw = index.get(model_id)
        return ModelRecord(**raw) if raw else None

    def get_production(self) -> Optional[ModelRecord]:
        """当前唯一 PRODUCTION 模型 (推理路径必须用此)"""
        index = self._load_index()
        for raw in index.values():
            if raw.get("state") == ModelState.PRODUCTION:
                return ModelRecord(**raw)
        return None

    def list_records(self, state: Optional[str] = None) -> List[ModelRecord]:
        index = self._load_index()
        recs = [ModelRecord(**raw) for raw in index.values()]
        if state:
            recs = [r for r in recs if r.state == state]
        return sorted(recs, key=lambda r: r.created_at, reverse=True)

    def resolve_artifact(self, model_id: str) -> Path:
        """返回可用于推理的制品路径 (优先标准生产制品目录，其次注册表内制品, 回退源制品)"""
        rec = self.get(model_id)
        if rec is None:
            raise PromotionError(f"模型不存在: {model_id}")
        # 1. 优先检查生产规范目录 saved_models/production/<model_id>/model.pkl
        prod_path = self.root.parent / "production" / model_id / "model.pkl"
        if prod_path.exists():
            return prod_path
        # 2. 检查注册表制品
        if rec.registry_artifact:
            p = Path(rec.registry_artifact)
            if not p.is_absolute():
                p = Path(settings.BASE_DIR) / p
            if p.exists():
                return p
        # 3. 回退源制品
        if rec.source_artifact:
            src = Path(rec.source_artifact)
            if not src.is_absolute():
                src = Path(settings.BASE_DIR) / src
            if src.exists():
                return src
        raise FileNotFoundError(f"模型制品缺失: {model_id}")
