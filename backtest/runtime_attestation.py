"""
运行时审计数字信封与非对称密码学防伪签名鉴证引擎 (backtest/runtime_attestation.py)
核心原则：
1. RUNTIME JSON != TRUSTED RUNTIME EVIDENCE (未通过 Ed25519 数字签名的本地 JSON 绝不能作为生产凭据)
2. ASYMMETRIC TRUST ANCHOR: 签名依靠外部注入私钥，验签仅依靠公开注册公钥，彻底杜绝源码推导假签名。
3. ZERO HARDCODED SECRETS: 仓库与默认配置不包含任何生产私钥，开发环境若无私钥则自动降级为 DEV_UNTRUSTED。
4. CODE COMMIT BINDING: 强绑定 Git Commit SHA、Tree Hash 与 Dirty 状态。
"""
import os
import sys
import json
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path

from backtest.audit import AuditMetadata, CertificationPolicy
from data.crypto_anchor import (
    TRUSTED_KEY_REGISTRY,
    verify_ed25519_signature,
    sign_with_environment_key
)


def get_git_environment_info() -> Dict[str, Any]:
    """获取当前工作区的真实 Git Commit、Tree Hash 与 Dirty 状态"""
    info = {
        "git_commit_sha": "unknown_commit",
        "git_tree_hash": "unknown_tree",
        "git_dirty": True
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        info["git_commit_sha"] = commit
    except Exception:
        pass

    try:
        tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], stderr=subprocess.DEVNULL, text=True).strip()
        info["git_tree_hash"] = tree
    except Exception:
        pass

    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip()
        info["git_dirty"] = len(status) > 0
    except Exception:
        info["git_dirty"] = True

    return info


def compute_canonical_audit_payload_hash(audit_meta: Union[AuditMetadata, Dict[str, Any]]) -> str:
    """确定性计算 AuditMetadata 的 Canonical JSON SHA256 哈希"""
    if isinstance(audit_meta, AuditMetadata):
        data = audit_meta.to_dict()
    else:
        data = dict(audit_meta)

    sorted_json = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(sorted_json.encode('utf-8')).hexdigest()


@dataclass
class RuntimeAttestationEnvelope:
    """运行时审计防伪数字信封 (Runtime Attestation Envelope - Ed25519 Asymmetric Protected)"""
    schema_version: str = "3.0"
    runtime_instance_id: str = ""
    git_commit_sha: str = ""
    git_tree_hash: str = ""
    git_dirty: bool = True
    python_version: str = platform.python_version()
    os_platform: str = platform.platform()
    runtime_config_hash: Optional[str] = None
    market_manifest_hash: Optional[str] = None
    factor_manifest_hash: Optional[str] = None
    universe_manifest_hash: Optional[str] = None
    pytest_xml_hash: Optional[str] = None
    audit_payload_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signing_key_id: Optional[str] = None
    envelope_signature: Optional[str] = None

    def compute_envelope_digest(self) -> str:
        """计算信封关键元数据的摘要"""
        raw_str = (
            f"{self.schema_version}|{self.runtime_instance_id}|"
            f"{self.git_commit_sha}|{self.git_tree_hash}|{self.git_dirty}|"
            f"{self.runtime_config_hash}|{self.market_manifest_hash}|"
            f"{self.factor_manifest_hash}|{self.universe_manifest_hash}|"
            f"{self.pytest_xml_hash}|{self.audit_payload_hash}|{self.created_at}"
        )
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    def sign(
        self,
        signing_key_id: str = "PROD_RUNTIME_KEY_2026_V1",
        explicit_private_key_hex: Optional[str] = None
    ) -> "RuntimeAttestationEnvelope":
        """
        使用指定的非对称私钥对信封进行 Ed25519 数字签名。
        若环境未注入私钥，则将 key_id 标记为 DEV_UNTRUSTED_KEY 且签名置空。
        """
        digest = self.compute_envelope_digest()
        sig_hex, errs = sign_with_environment_key(
            message=digest.encode("utf-8"),
            key_id=signing_key_id,
            explicit_private_key_hex=explicit_private_key_hex
        )
        if sig_hex:
            self.signing_key_id = signing_key_id
            self.envelope_signature = sig_hex
        else:
            self.signing_key_id = "DEV_UNTRUSTED_KEY"
            self.envelope_signature = None
        return self

    def verify(
        self,
        audit_payload_data: Dict[str, Any],
        require_clean_git: bool = False
    ) -> Tuple[bool, List[str]]:
        """全方位校验信封 Ed25519 签名、Payload 哈希一致性与代码版本绑定"""
        errors = []

        # 1. 校验 Payload 哈希一致性
        actual_payload_hash = compute_canonical_audit_payload_hash(audit_payload_data)
        if actual_payload_hash.lower() != str(self.audit_payload_hash).lower():
            errors.append(f"audit_payload_hash_mismatch_{actual_payload_hash}_vs_{self.audit_payload_hash}")

        # 2. 校验 Ed25519 非对称密码学数字签名 (严禁任何自算 SHA256 伪签名)
        if not self.signing_key_id or self.signing_key_id not in TRUSTED_KEY_REGISTRY:
            errors.append(f"unregistered_runtime_signing_key_{self.signing_key_id}")
        elif not self.envelope_signature:
            errors.append("missing_runtime_envelope_signature")
        else:
            digest = self.compute_envelope_digest()
            sig_ok, sig_errs = verify_ed25519_signature(
                message=digest.encode("utf-8"),
                signature_hex=self.envelope_signature,
                key_id=self.signing_key_id
            )
            if not sig_ok:
                errors.extend(sig_errs)

        # 3. 校验 Git 状态 (生产环境 VERIFIED 严格要求干净工作区)
        if require_clean_git and self.git_dirty:
            errors.append("git_worktree_is_dirty_untrusted_for_production_verification")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_signed_runtime_attestation(
    audit_meta: AuditMetadata,
    pytest_xml_path: Optional[Path] = None,
    signing_key_id: str = "PROD_RUNTIME_KEY_2026_V1",
    explicit_private_key_hex: Optional[str] = None
) -> Dict[str, Any]:
    """根据运行时审计元数据打包生成防伪数字信封与完整 Attestation 文档"""
    git_info = get_git_environment_info()
    payload_hash = compute_canonical_audit_payload_hash(audit_meta)

    xml_hash = None
    if pytest_xml_path and Path(pytest_xml_path).exists():
        h = hashlib.sha256()
        with open(pytest_xml_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        xml_hash = h.hexdigest()

    envelope = RuntimeAttestationEnvelope(
        runtime_instance_id=audit_meta.runtime_instance_id,
        git_commit_sha=git_info["git_commit_sha"],
        git_tree_hash=git_info["git_tree_hash"],
        git_dirty=git_info["git_dirty"],
        runtime_config_hash=audit_meta.runtime_config_hash,
        market_manifest_hash=audit_meta.market_manifest_hash,
        factor_manifest_hash=audit_meta.factor_manifest_hash,
        universe_manifest_hash=audit_meta.universe_manifest_hash,
        pytest_xml_hash=xml_hash,
        audit_payload_hash=payload_hash
    )
    envelope.sign(
        signing_key_id=signing_key_id,
        explicit_private_key_hex=explicit_private_key_hex
    )

    return {
        "attestation_envelope": envelope.to_dict(),
        "audit_metadata": audit_meta.to_dict()
    }
