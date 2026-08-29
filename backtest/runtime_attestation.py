"""
运行时审计数字信封与非对称密码学防伪签名鉴证引擎 (backtest/runtime_attestation.py)
核心原则：
1. RUNTIME JSON != TRUSTED RUNTIME EVIDENCE (未通过 Ed25519 数字签名的本地 JSON 绝不能作为生产凭据)
2. ASYMMETRIC TRUST ANCHOR: 签名依靠外部注入私钥，验签仅依靠公开注册公钥与严格 Purpose 检查。
3. GIT COMMIT & TREE BINDING: 生产认证强制要求 Envelope Commit/Tree 与当前 HEAD 保持 100% 一致。
4. DIRTY WORKTREE GATE: Envelope 启动时或当前验证时若存在脏工作区，严格禁止获得最高 VERIFIED。
5. HISTORICAL VS CURRENT ARTIFACT: 显式支持并区分历史已归档报告与当前运行时回测报告。
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
    DOMAIN_SEPARATOR_RUNTIME,
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
    corporate_action_manifest_hash: Optional[str] = None
    pytest_xml_hash: Optional[str] = None
    audit_payload_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signing_key_id: Optional[str] = None
    envelope_signature: Optional[str] = None

    def compute_canonical_bytes(self) -> bytes:
        """确定性计算信封待签名字节流"""
        raw_dict = {
            "audit_payload_hash": self.audit_payload_hash,
            "corporate_action_manifest_hash": self.corporate_action_manifest_hash,
            "created_at": self.created_at,
            "factor_manifest_hash": self.factor_manifest_hash,
            "git_commit_sha": self.git_commit_sha,
            "git_dirty": self.git_dirty,
            "git_tree_hash": self.git_tree_hash,
            "market_manifest_hash": self.market_manifest_hash,
            "os_platform": self.os_platform,
            "pytest_xml_hash": self.pytest_xml_hash,
            "python_version": self.python_version,
            "runtime_config_hash": self.runtime_config_hash,
            "runtime_instance_id": self.runtime_instance_id,
            "schema_version": self.schema_version,
            "universe_manifest_hash": self.universe_manifest_hash
        }
        sorted_json = json.dumps(raw_dict, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        return sorted_json.encode("utf-8")

    def sign(
        self,
        signing_key_id: str = "PROD_RUNTIME_KEY_2026_V1",
        explicit_private_key_hex: Optional[str] = None
    ) -> "RuntimeAttestationEnvelope":
        """
        使用指定的非对称私钥对信封进行 Ed25519 数字签名。
        强制校验 Purpose=RUNTIME_ATTESTATION 与 Domain Separator 隔离。
        """
        msg_bytes = self.compute_canonical_bytes()
        sig_hex, errs = sign_with_environment_key(
            message=msg_bytes,
            key_id=signing_key_id,
            required_purpose="RUNTIME_ATTESTATION",
            domain_separator=DOMAIN_SEPARATOR_RUNTIME,
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
        require_clean_git: bool = True,
        verify_current_git_binding: bool = True,
        is_historical: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        全方位校验信封 Ed25519 签名、Payload 哈希一致性、代码版本强绑定与工作区纯净性
        """
        errors = []

        # 1. 校验 Payload 哈希一致性
        actual_payload_hash = compute_canonical_audit_payload_hash(audit_payload_data)
        if actual_payload_hash.lower() != str(self.audit_payload_hash).lower():
            errors.append(f"audit_payload_hash_mismatch_{actual_payload_hash}_vs_{self.audit_payload_hash}")

        # 2. 校验 Ed25519 非对称密码学数字签名 (严格用途与注册公钥检查)
        if not self.signing_key_id or self.signing_key_id not in TRUSTED_KEY_REGISTRY:
            errors.append(f"unregistered_runtime_signing_key_{self.signing_key_id}")
        elif not self.envelope_signature:
            errors.append("missing_runtime_envelope_signature")
        else:
            msg_bytes = self.compute_canonical_bytes()
            sig_ok, sig_errs = verify_ed25519_signature(
                message=msg_bytes,
                signature_hex=self.envelope_signature,
                key_id=self.signing_key_id,
                required_purpose="RUNTIME_ATTESTATION",
                domain_separator=DOMAIN_SEPARATOR_RUNTIME,
                created_at_iso=self.created_at
            )
            if not sig_ok:
                errors.extend(sig_errs)

        # 3. 校验 Git Commit 与 Tree 强绑定 (P0 Git Binding Guard)
        if verify_current_git_binding and not is_historical:
            current_git = get_git_environment_info()
            if self.git_commit_sha != current_git["git_commit_sha"]:
                errors.append(f"runtime_commit_mismatch_{self.git_commit_sha}_vs_{current_git['git_commit_sha']}")
            if self.git_tree_hash != current_git["git_tree_hash"]:
                errors.append(f"runtime_tree_hash_mismatch_{self.git_tree_hash}_vs_{current_git['git_tree_hash']}")

        # 4. 校验 Git 纯净性状态 (生产环境 VERIFIED 严格要求干净工作区)
        if require_clean_git and not is_historical:
            if self.git_dirty:
                errors.append("runtime_started_from_dirty_worktree")
            current_git = get_git_environment_info()
            if current_git.get("git_dirty", True):
                errors.append("current_worktree_dirty")

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
        corporate_action_manifest_hash=audit_meta.corporate_action_manifest_hash,
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
