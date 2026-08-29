"""
运行时审计防伪数字信封与密码学凭据签发引擎 (backtest/runtime_attestation.py)
核心原则：
1. 真实非对称签名：使用 Ed25519 非对称公私钥对 AuditMetadata 的规范化哈希与关键构建指纹进行数字签名。
2. 外部 Trust Root 强绑定：信封待签文字节流强制绑定 trusted_keyring_hash，防止 Agent 任意替换公钥表。
3. 源码 Git Binding 与产物解耦：绑定 Pipeline 启动时的 code_commit_sha 与 code_tree_hash，
   精准分离 source_code_dirty (源码脏) 与 generated_artifact_dirty (构建产物写入)，杜绝自引用循环。
4. Fail-Closed 机制：在无生产私钥或外部 Trust Root Pin 缺失/不匹配时，自动标记为 DEV_UNTRUSTED_KEY 并判定为 HIGH_RISK。
"""
import os
import re
import json
import hashlib
import platform
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union

from backtest.audit import AuditMetadata, CertificationPolicy, compute_canonical_runtime_config_hash
from data.crypto_anchor import (
    TRUSTED_KEY_REGISTRY,
    DOMAIN_SEPARATOR_RUNTIME,
    verify_ed25519_signature,
    sign_with_environment_key,
    compute_canonical_keyring_hash,
    verify_trust_root
)


def get_git_environment_info(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    获取工作区的真实 Git Commit、Tree Hash 与精细化 Dirty 状态。
    严格区分源代码脏状态 (source_code_dirty) 与生成构建产物脏状态 (generated_artifact_dirty)。
    """
    info = {
        "git_commit_sha": "unknown_commit",
        "code_commit_sha": "unknown_commit",
        "git_tree_hash": "unknown_tree",
        "code_tree_hash": "unknown_tree",
        "source_code_dirty": True,
        "generated_artifact_dirty": False,
        "git_dirty": True
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, cwd=repo_root).strip()
        info["git_commit_sha"] = commit
        info["code_commit_sha"] = commit
    except Exception:
        pass

    try:
        tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], stderr=subprocess.DEVNULL, text=True, cwd=repo_root).strip()
        info["git_tree_hash"] = tree
        info["code_tree_hash"] = tree
    except Exception:
        pass

    try:
        status_raw = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True, cwd=repo_root).strip()
        if not status_raw:
            info["git_dirty"] = False
            info["source_code_dirty"] = False
            info["generated_artifact_dirty"] = False
        else:
            info["git_dirty"] = True
            generated_prefixes = (
                "artifacts/", "reports/", "CAPABILITY_REPORT.md", "RUNTIME_ATTESTATION.md",
                "MASTER_REPORT.md", "data_storage/", ".gemini/"
            )
            source_dirty = False
            artifact_dirty = False
            for line in status_raw.splitlines():
                clean_line = line.strip()
                if not clean_line:
                    continue
                parts = clean_line.split(maxsplit=1)
                fn = parts[1] if len(parts) > 1 else clean_line
                fn_norm = fn.replace("\\", "/").strip('"')
                if any(fn_norm.startswith(p) or fn_norm == p for p in generated_prefixes):
                    artifact_dirty = True
                else:
                    source_dirty = True
            info["source_code_dirty"] = source_dirty
            info["generated_artifact_dirty"] = artifact_dirty
    except Exception:
        info["git_dirty"] = True
        info["source_code_dirty"] = True

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
    schema_version: str = "3.1"
    runtime_instance_id: str = ""
    code_commit_sha: str = ""
    code_tree_hash: str = ""
    source_code_dirty: bool = True
    git_commit_sha: str = ""  # 保持旧字段兼容
    git_tree_hash: str = ""    # 保持旧字段兼容
    git_dirty: bool = True     # 保持旧字段兼容
    python_version: str = platform.python_version()
    os_platform: str = platform.platform()
    runtime_config_hash: Optional[str] = None
    market_manifest_hash: Optional[str] = None
    factor_manifest_hash: Optional[str] = None
    universe_manifest_hash: Optional[str] = None
    corporate_action_manifest_hash: Optional[str] = None
    pytest_xml_hash: Optional[str] = None
    trusted_keyring_hash: Optional[str] = None
    audit_payload_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signing_key_id: Optional[str] = None
    envelope_signature: Optional[str] = None

    def compute_canonical_bytes(self) -> bytes:
        """确定性计算信封待签名字节流 (绑定代码 Commit、Tree、配置、Manifest 与 Keyring 指纹)"""
        commit_sha = self.code_commit_sha or self.git_commit_sha
        tree_hash = self.code_tree_hash or self.git_tree_hash
        src_dirty = self.source_code_dirty if self.source_code_dirty is not None else self.git_dirty

        raw_dict = {
            "audit_payload_hash": self.audit_payload_hash,
            "code_commit_sha": commit_sha,
            "code_tree_hash": tree_hash,
            "corporate_action_manifest_hash": self.corporate_action_manifest_hash,
            "created_at": self.created_at,
            "factor_manifest_hash": self.factor_manifest_hash,
            "git_commit_sha": commit_sha,
            "git_dirty": self.git_dirty,
            "git_tree_hash": tree_hash,
            "market_manifest_hash": self.market_manifest_hash,
            "os_platform": self.os_platform,
            "pytest_xml_hash": self.pytest_xml_hash,
            "python_version": self.python_version,
            "runtime_config_hash": self.runtime_config_hash,
            "runtime_instance_id": self.runtime_instance_id,
            "schema_version": self.schema_version,
            "source_code_dirty": src_dirty,
            "trusted_keyring_hash": self.trusted_keyring_hash,
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
        强制自动绑定当前 Trust Root Keyring 哈希，并校验 Purpose=RUNTIME_ATTESTATION 与 Domain 隔离。
        """
        self.trusted_keyring_hash = compute_canonical_keyring_hash()
        msg_bytes = self.compute_canonical_bytes()
        sig_hex, errs = sign_with_environment_key(
            message=msg_bytes,
            key_id=signing_key_id,
            required_purpose="RUNTIME_ATTESTATION",
            domain_separator=DOMAIN_SEPARATOR_RUNTIME,
            explicit_private_key_hex=explicit_private_key_hex,
            production_mode=True
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
        is_historical: bool = False,
        require_trust_root: bool = True,
        external_keyring_pin: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        全方位校验信封 Ed25519 签名、Payload 哈希一致性、代码版本强绑定、源码纯净性与外部 Trust Root 锚定
        """
        errors = []

        # 1. 校验 Payload 哈希一致性
        actual_payload_hash = compute_canonical_audit_payload_hash(audit_payload_data)
        if actual_payload_hash.lower() != str(self.audit_payload_hash).lower():
            errors.append(f"audit_payload_hash_mismatch_{actual_payload_hash}_vs_{self.audit_payload_hash}")

        # 2. 校验外部 Trust Root 锚定状态 (P0 External Pinning Guard)
        current_keyring_hash = compute_canonical_keyring_hash()
        if self.trusted_keyring_hash and self.trusted_keyring_hash.lower() != current_keyring_hash.lower():
            errors.append(f"envelope_keyring_hash_mismatch_{self.trusted_keyring_hash}_vs_repo_{current_keyring_hash}")

        if require_trust_root:
            tr_ok, _, _, tr_errs = verify_trust_root(explicit_external_pin=external_keyring_pin)
            if not tr_ok:
                errors.extend(tr_errs)

        # 3. 校验 Ed25519 非对称密码学数字签名 (严格用途与注册公钥检查)
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
                created_at_iso=self.created_at,
                production_mode=True
            )
            if not sig_ok:
                errors.extend(sig_errs)

        # 4. 校验 Git Commit 与 Tree 强绑定 (P0 Git Binding Guard)
        if verify_current_git_binding and not is_historical:
            current_git = get_git_environment_info()
            envelope_commit = self.code_commit_sha or self.git_commit_sha
            envelope_tree = self.code_tree_hash or self.git_tree_hash
            if envelope_commit != current_git["code_commit_sha"]:
                errors.append(f"runtime_commit_mismatch_{envelope_commit}_vs_{current_git['code_commit_sha']}")
            if envelope_tree != current_git["code_tree_hash"]:
                errors.append(f"runtime_tree_hash_mismatch_{envelope_tree}_vs_{current_git['code_tree_hash']}")

        # 5. 校验源码纯净性状态 (生产环境 VERIFIED 严格要求源代码干净)
        if require_clean_git and not is_historical:
            src_dirty_at_start = self.source_code_dirty if self.source_code_dirty is not None else self.git_dirty
            if src_dirty_at_start:
                errors.append("runtime_started_from_dirty_source_code")
            current_git = get_git_environment_info()
            if current_git.get("source_code_dirty", True):
                errors.append("current_source_code_dirty")

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
        code_commit_sha=git_info["code_commit_sha"],
        code_tree_hash=git_info["code_tree_hash"],
        source_code_dirty=git_info["source_code_dirty"],
        git_commit_sha=git_info["git_commit_sha"],
        git_tree_hash=git_info["git_tree_hash"],
        git_dirty=git_info["git_dirty"],
        runtime_config_hash=audit_meta.runtime_config_hash,
        market_manifest_hash=audit_meta.market_manifest_hash,
        factor_manifest_hash=audit_meta.factor_manifest_hash,
        universe_manifest_hash=audit_meta.universe_manifest_hash,
        corporate_action_manifest_hash=audit_meta.corporate_action_manifest_hash,
        pytest_xml_hash=xml_hash,
        trusted_keyring_hash=audit_meta.trusted_keyring_hash or compute_canonical_keyring_hash(),
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
