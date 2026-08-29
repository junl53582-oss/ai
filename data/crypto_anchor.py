"""
企业级独立密码学信任锚点与非对称数字签名引擎 (data/crypto_anchor.py)
核心原则：
1. PUBLIC KEYS ONLY IN REPO: 仓库注册表仅保存公钥、Key ID 与用途权限，绝无任何私钥或生产 Secret。
2. EXTERNAL TRUST ROOT PINNING: 生产认证强制校验 REPO-EXTERNAL ROOT OF TRUST (QUANT_TRUSTED_KEYRING_SHA256)。
   防止任何具备仓库写权限的 Agent 通过修改公钥注册表实现自签名伪造 (Agent Trust Root Replacement Defense)。
3. FAIL-CLOSED TRUST ROOT: 外部 Trust Root Pin 缺失或不匹配一律判定为 FAIL-CLOSED (HIGH_RISK)。
4. STRICT BACKEND POLICY: 生产模式强制要求成熟 cryptography 库 (RFC 8032 Ed25519)，自研纯 Python 仅作为测试与向量比对参考。
5. KEY PURPOSE SEPARATION & DOMAIN SEPARATORS: 强制用途隔离与域分隔符，彻底防御跨协议/跨组件签名混淆与重放攻击。
"""
import os
import re
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path

# 优先导入成熟生产级 cryptography 库
try:
    from cryptography.hazmat.primitives.asymmetric import ed25519 as crypto_ed25519
    from cryptography.exceptions import InvalidSignature
    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False


# =========================================================================
# 1. 域分隔符与路径安全约束 (Domain Separators & Safe Path Containment)
# =========================================================================

DOMAIN_SEPARATOR_RUNTIME = "QUANT_RUNTIME_ATTESTATION_V1"
DOMAIN_SEPARATOR_ACQUISITION = "QUANT_ACQUISITION_RECEIPT_V1"
DOMAIN_SEPARATOR_OPERATOR = "QUANT_VENDOR_OPERATOR_ATTESTATION_V1"
DOMAIN_SEPARATOR_CORPORATE_ACTION = "QUANT_CORPORATE_ACTION_ATTESTATION_V1"


def safe_resolve_path(base_dir: Union[str, Path], subpath: Union[str, Path]) -> Optional[Path]:
    """
    严格防御路径穿越攻击 (Path Traversal Guard):
    解析 subpath 并强制确保其位于 base_dir 物理目录树内，拒绝 ../、绝对路径跨越及恶意软链接逃逸。
    """
    try:
        base = Path(base_dir).resolve()
        target = (base / subpath).resolve()
        if target == base or base in target.parents:
            return target
        return None
    except Exception:
        return None


# =========================================================================
# 2. RFC 8032 Ed25519 纯 Python 算法 (Reference / Test Vector Validation Only)
# =========================================================================

_p = 2**255 - 19
_d = -121665 * pow(121666, -1, _p) % _p
_q = 2**252 + 27742317777372353535851937790883648493


def _H(m: bytes) -> bytes:
    return hashlib.sha512(m).digest()


def _expmod(b: int, e: int, m: int) -> int:
    return pow(b, e, m)


def _inv(x: int) -> int:
    return _expmod(x, _p - 2, _p)


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * _inv(_d * y * y + 1)
    x = _expmod(xx, (_p + 3) // 8, _p)
    if (x * x - xx) % _p != 0:
        x = (x * _expmod(2, (_p - 1) // 4, _p)) % _p
    if x % 2 != 0:
        x = _p - x
    return x


_By = 4 * _inv(5) % _p
_Bx = _xrecover(_By)
_B = [_Bx % _p, _By % _p]


def _edwards_add(P: List[int], Q: List[int]) -> List[int]:
    x1, y1 = P[0], P[1]
    x2, y2 = Q[0], Q[1]
    x3 = (x1 * y2 + x2 * y1) * _inv(1 + _d * x1 * x2 * y1 * y2) % _p
    y3 = (y1 * y2 + x1 * x2) * _inv(1 - _d * x1 * x2 * y1 * y2) % _p
    return [x3, y3]


def _scalarmult(P: List[int], e: int) -> List[int]:
    if e == 0:
        return [0, 1]
    Q = _scalarmult(P, e // 2)
    Q = _edwards_add(Q, Q)
    if e & 1:
        Q = _edwards_add(Q, P)
    return Q


def _encodeint(y: int) -> bytes:
    return (y).to_bytes(32, "little")


def _decodeint(b: bytes) -> int:
    return int.from_bytes(b, "little")


def _encodepoint(P: List[int]) -> bytes:
    x, y = P[0], P[1]
    bits = (y & ((1 << 255) - 1)) | ((x & 1) << 255)
    return _encodeint(bits)


def _decodepoint(b: bytes) -> Optional[List[int]]:
    try:
        y = _decodeint(b) & ((1 << 255) - 1)
        x = _xrecover(y)
        if (x & 1) != ((_decodeint(b) >> 255) & 1):
            x = _p - x
        return [x, y]
    except Exception:
        return None


def ed25519_publickey_pure(sk: bytes) -> bytes:
    """纯 Python 计算 Ed25519 公钥 (仅供测试对照)"""
    h = _H(sk)
    a = 2**254 + sum(2**i * ((h[i // 8] >> (i % 8)) & 1) for i in range(3, 254))
    A = _scalarmult(_B, a)
    return _encodepoint(A)


def ed25519_sign_pure(m: bytes, sk: bytes, pk: bytes) -> bytes:
    """纯 Python 生成 64 字节 Ed25519 签名 (仅供测试对照)"""
    h = _H(sk)
    a = 2**254 + sum(2**i * ((h[i // 8] >> (i % 8)) & 1) for i in range(3, 254))
    r = int.from_bytes(_H(h[32:] + m), "little") % _q
    R = _scalarmult(_B, r)
    R_enc = _encodepoint(R)
    k = int.from_bytes(_H(R_enc + pk + m), "little") % _q
    S = (r + k * a) % _q
    return R_enc + _encodeint(S)


def ed25519_verify_pure(sig: bytes, m: bytes, pk: bytes) -> bool:
    """纯 Python 验证 Ed25519 签名 (仅供测试对照)"""
    if len(sig) != 64 or len(pk) != 32:
        return False
    try:
        R = _decodepoint(sig[:32])
        A = _decodepoint(pk)
        if R is None or A is None:
            return False
        S = _decodeint(sig[32:])
        if S >= _q:
            return False
        k = int.from_bytes(_H(sig[:32] + pk + m), "little") % _q
        v1 = _scalarmult(_B, S)
        v2 = _edwards_add(R, _scalarmult(A, k))
        return v1 == v2
    except Exception:
        return False


def generate_keypair() -> Tuple[bytes, bytes]:
    """生成一对全新的 Ed25519 (私钥 32 字节, 公钥 32 字节)"""
    if _HAS_CRYPTOGRAPHY:
        priv = crypto_ed25519.Ed25519PrivateKey.generate()
        sk = priv.private_bytes_raw()
        pk = priv.public_key().public_bytes_raw()
        return sk, pk
    else:
        sk = os.urandom(32)
        pk = ed25519_publickey_pure(sk)
        return sk, pk


# =========================================================================
# 3. 受信任公钥注册表 (PUBLIC KEYS ONLY - 包含用途隔离与生命周期管理)
# =========================================================================

TRUSTED_KEY_REGISTRY: Dict[str, Dict[str, Any]] = {
    # 1. 项目数据采集器签名公钥 (仅用于 Acquisition Receipt，严禁用于签发运行时报告)
    "PROD_DOWNLOADER_KEY_2026_V1": {
        "algorithm": "ED25519",
        "key_id": "PROD_DOWNLOADER_KEY_2026_V1",
        "public_key_hex": "4d16d000632d43105ff7f6e3c048bcfa28ebc91a08e1bd074f7be33f2e105d15",
        "allowed_purposes": ["ACQUISITION_RECEIPT"],
        "issuer_type": "PROJECT",
        "institution": "PROJECT_TRUSTED_DOWNLOADER_AUTHORITY",
        "env_private_key_var": "QUANT_PROD_ACQUISITION_PRIVATE_KEY",
        "status": "ACTIVE",
        "not_before": "2025-01-01T00:00:00Z",
        "not_after": "2030-01-01T00:00:00Z",
        "is_production": True
    },
    # 2. 项目运行时审计防伪信封签名公钥 (仅用于 Runtime Attestation Envelope)
    "PROD_RUNTIME_KEY_2026_V1": {
        "algorithm": "ED25519",
        "key_id": "PROD_RUNTIME_KEY_2026_V1",
        "public_key_hex": "9a38549646b95ee36b22b64d1f2fb280c7d5c7c258d46d0a7a372691361c77f0",
        "allowed_purposes": ["RUNTIME_ATTESTATION"],
        "issuer_type": "PROJECT",
        "institution": "QUANT_INFRA_ATTESTATION_AUTHORITY",
        "env_private_key_var": "QUANT_PROD_RUNTIME_PRIVATE_KEY",
        "status": "ACTIVE",
        "not_before": "2025-01-01T00:00:00Z",
        "not_after": "2030-01-01T00:00:00Z",
        "is_production": True
    },
    # 3. Wind 终端导出操作员凭据签名公钥 (项目登记的操作员凭据，非 Wind 公司官方数字证书)
    "WIND_OPERATOR_KEY_001": {
        "algorithm": "ED25519",
        "key_id": "WIND_OPERATOR_KEY_001",
        "public_key_hex": "2b467ef73c68a417578326a27e7fa69911e3b62cfb9b877cb63a48e7e1f4094a",
        "allowed_purposes": ["LICENSED_VENDOR_OPERATOR_ATTESTATION"],
        "issuer_type": "PROJECT_REGISTERED_VENDOR_OPERATOR",
        "institution": "WIND_INFORMATION_TERMINAL_OPERATOR",
        "env_private_key_var": "WIND_OPERATOR_PRIVATE_KEY",
        "status": "ACTIVE",
        "not_before": "2025-01-01T00:00:00Z",
        "not_after": "2030-01-01T00:00:00Z",
        "is_production": True
    },
    # 4. Choice 终端导出操作员凭据签名公钥 (项目登记的操作员凭据，非 Choice 官方数字证书)
    "CHOICE_OPERATOR_KEY_001": {
        "algorithm": "ED25519",
        "key_id": "CHOICE_OPERATOR_KEY_001",
        "public_key_hex": "d2a04874c7fd239fb8b1b36be9f28d8b9d3b7e41b21235b2e5352eb9c9b4e723",
        "allowed_purposes": ["LICENSED_VENDOR_OPERATOR_ATTESTATION"],
        "issuer_type": "PROJECT_REGISTERED_VENDOR_OPERATOR",
        "institution": "EASTMONEY_CHOICE_TERMINAL_OPERATOR",
        "env_private_key_var": "CHOICE_OPERATOR_PRIVATE_KEY",
        "status": "ACTIVE",
        "not_before": "2025-01-01T00:00:00Z",
        "not_after": "2030-01-01T00:00:00Z",
        "is_production": True
    },
    # 5. 已吊销的测试公钥 (用于轮换/吊销测试)
    "REVOKED_TEST_KEY_2025": {
        "algorithm": "ED25519",
        "key_id": "REVOKED_TEST_KEY_2025",
        "public_key_hex": "11" * 32,
        "allowed_purposes": ["RUNTIME_ATTESTATION", "ACQUISITION_RECEIPT"],
        "issuer_type": "PROJECT",
        "institution": "REVOKED_TEST_AUTHORITY",
        "status": "REVOKED",
        "not_before": "2025-01-01T00:00:00Z",
        "not_after": "2030-01-01T00:00:00Z",
        "is_production": False
    },
    # 6. 已过期的测试公钥 (用于过期时间窗口测试)
    "EXPIRED_TEST_KEY_2020": {
        "algorithm": "ED25519",
        "key_id": "EXPIRED_TEST_KEY_2020",
        "public_key_hex": "22" * 32,
        "allowed_purposes": ["RUNTIME_ATTESTATION"],
        "issuer_type": "PROJECT",
        "institution": "EXPIRED_TEST_AUTHORITY",
        "status": "EXPIRED",
        "not_before": "2019-01-01T00:00:00Z",
        "not_after": "2020-01-01T00:00:00Z",
        "is_production": False
    },
    # 7. 尚未生效的测试公钥 (用于未来时间窗口测试)
    "FUTURE_TEST_KEY_2099": {
        "algorithm": "ED25519",
        "key_id": "FUTURE_TEST_KEY_2099",
        "public_key_hex": "33" * 32,
        "allowed_purposes": ["RUNTIME_ATTESTATION"],
        "issuer_type": "PROJECT",
        "institution": "FUTURE_TEST_AUTHORITY",
        "status": "ACTIVE",
        "not_before": "2099-01-01T00:00:00Z",
        "not_after": "2100-01-01T00:00:00Z",
        "is_production": False
    }
}


# =========================================================================
# 4. 外部受信任密钥环指纹与 Trust Root 锚定 (External Trust Root Pinning)
# =========================================================================

def compute_canonical_keyring_hash(keyring: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """
    确定性计算受信任公钥环的 Canonical SHA256 哈希 (Trusted Keyring Canonical Hash)。
    严格根据 Key ID 排序并提取关键身份字段，防止 Agent 任意修改公钥表自签名。
    """
    reg = keyring if keyring is not None else TRUSTED_KEY_REGISTRY
    canonical_data: Dict[str, Any] = {}
    for key_id in sorted(reg.keys()):
        k_info = reg[key_id]
        canonical_data[key_id] = {
            "allowed_purposes": sorted(list(k_info.get("allowed_purposes", []))),
            "issuer_type": str(k_info.get("issuer_type", "")),
            "not_after": str(k_info.get("not_after", "")),
            "not_before": str(k_info.get("not_before", "")),
            "public_key_hex": str(k_info.get("public_key_hex", "")).lower().strip(),
            "status": str(k_info.get("status", ""))
        }
    sorted_json = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(sorted_json.encode('utf-8')).hexdigest()


def verify_trust_root(
    keyring: Optional[Dict[str, Dict[str, Any]]] = None,
    explicit_external_pin: Optional[str] = None
) -> Tuple[bool, str, Optional[str], List[str]]:
    """
    核验当前公钥注册表是否与 REPO-EXTERNAL ROOT OF TRUST 保持绝对一致 (P0)。
    若未设置环境变量 QUANT_TRUSTED_KEYRING_SHA256 或哈希不匹配，严格 Fail-Closed。
    """
    actual_hash = compute_canonical_keyring_hash(keyring)
    env_pin = explicit_external_pin or os.environ.get("QUANT_TRUSTED_KEYRING_SHA256")

    errors: List[str] = []
    if not env_pin:
        errors.append("missing_external_trust_root_pin_QUANT_TRUSTED_KEYRING_SHA256")
        return False, actual_hash, None, errors

    clean_pin = str(env_pin).strip().lower()
    if clean_pin != actual_hash.lower():
        errors.append(f"trust_root_tampered_keyring_hash_mismatch_{actual_hash}_vs_pin_{clean_pin}")
        return False, actual_hash, clean_pin, errors

    return True, actual_hash, clean_pin, []


# =========================================================================
# 5. 受信任操作员注册表 (Trusted Operator Registry)
# =========================================================================

TRUSTED_OPERATOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "WIND_OPERATOR_001": {
        "vendor_source_id": "WIND",
        "signing_key_id": "WIND_OPERATOR_KEY_001",
        "operator_name": "Wind Production Export Operator 001",
        "status": "ACTIVE"
    },
    "CHOICE_OPERATOR_001": {
        "vendor_source_id": "CHOICE",
        "signing_key_id": "CHOICE_OPERATOR_KEY_001",
        "operator_name": "Choice Production Export Operator 001",
        "status": "ACTIVE"
    }
}


def get_trusted_public_key(key_id: str) -> Optional[bytes]:
    """从注册表获取指定 Key ID 的公开验证公钥"""
    if key_id not in TRUSTED_KEY_REGISTRY:
        return None
    info = TRUSTED_KEY_REGISTRY[key_id]
    if info.get("status") != "ACTIVE":
        return None
    try:
        return bytes.fromhex(info["public_key_hex"])
    except Exception:
        return None


def _check_key_validity_window(info: Dict[str, Any], at_time_iso: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """核验公钥有效期窗口"""
    status = info.get("status", "ACTIVE")
    if status != "ACTIVE":
        return False, f"key_status_is_{status}"

    check_dt = datetime.fromisoformat(at_time_iso.replace("Z", "+00:00")) if at_time_iso else datetime.now(timezone.utc)

    not_before = info.get("not_before")
    if not_before:
        nb_dt = datetime.fromisoformat(not_before.replace("Z", "+00:00"))
        if check_dt < nb_dt:
            return False, f"key_not_yet_valid_before_{not_before}"

    not_after = info.get("not_after")
    if not_after:
        na_dt = datetime.fromisoformat(not_after.replace("Z", "+00:00"))
        if check_dt > na_dt:
            return False, f"key_expired_after_{not_after}"

    return True, None


def sign_with_environment_key(
    message: bytes,
    key_id: str,
    required_purpose: str,
    domain_separator: Optional[str] = None,
    explicit_private_key_hex: Optional[str] = None,
    production_mode: bool = False
) -> Tuple[Optional[str], List[str]]:
    """
    使用环境变量或显式传入的私钥对数据进行 Ed25519 签名。
    严格实施 Key Purpose 检查、Domain Separator 前缀隔离与 Production Backend 门禁。
    """
    if production_mode and not _HAS_CRYPTOGRAPHY:
        return None, ["cryptography_library_required_for_production_signing"]

    if key_id not in TRUSTED_KEY_REGISTRY:
        return None, [f"unregistered_signing_key_id_{key_id}"]

    reg_info = TRUSTED_KEY_REGISTRY[key_id]

    # 1. 严格用途校验 (Purpose Isolation)
    allowed_purposes = reg_info.get("allowed_purposes", [])
    if required_purpose not in allowed_purposes:
        return None, [f"key_purpose_mismatch_{key_id}_not_authorized_for_{required_purpose}"]

    # 2. 状态与有效期检查
    valid_window, err_msg = _check_key_validity_window(reg_info)
    if not valid_window:
        return None, [err_msg]

    # 3. 获取私钥
    env_var = reg_info.get("env_private_key_var", "")
    sk_hex = explicit_private_key_hex or (os.environ.get(env_var) if env_var else None)
    if not sk_hex:
        return None, [f"missing_private_key_in_environment_for_{key_id}"]

    try:
        sk = bytes.fromhex(sk_hex.strip())
        if len(sk) != 32:
            return None, ["invalid_private_key_length_must_be_32_bytes"]

        msg_to_sign = f"{domain_separator}:".encode("utf-8") + message if domain_separator else message

        if _HAS_CRYPTOGRAPHY:
            priv = crypto_ed25519.Ed25519PrivateKey.from_private_bytes(sk)
            pk = priv.public_key().public_bytes_raw()
            expected_pk_hex = reg_info.get("public_key_hex", "").lower()
            if pk.hex().lower() != expected_pk_hex:
                return None, ["private_key_does_not_match_registered_public_key"]
            sig = priv.sign(msg_to_sign)
            return sig.hex(), []
        else:
            if production_mode:
                return None, ["cryptography_library_required_for_production_signing"]
            pk = ed25519_publickey_pure(sk)
            expected_pk_hex = reg_info.get("public_key_hex", "").lower()
            if pk.hex().lower() != expected_pk_hex:
                return None, ["private_key_does_not_match_registered_public_key"]
            sig = ed25519_sign_pure(msg_to_sign, sk, pk)
            return sig.hex(), []

    except Exception as e:
        return None, [f"signing_failed_{str(e)}"]


def verify_ed25519_signature(
    message: bytes,
    signature_hex: str,
    key_id: str,
    required_purpose: str,
    domain_separator: Optional[str] = None,
    created_at_iso: Optional[str] = None,
    production_mode: bool = False
) -> Tuple[bool, List[str]]:
    """
    使用仓库中注册的公开公钥对 Ed25519 数字签名进行密码学严格验签。
    支持 Purpose 检查、Domain Separator 隔离、时间窗口/吊销检查与 Production Backend 门禁。
    """
    errors: List[str] = []
    if production_mode and not _HAS_CRYPTOGRAPHY:
        return False, ["cryptography_library_required_for_production_verification"]

    if not key_id or key_id not in TRUSTED_KEY_REGISTRY:
        return False, [f"unregistered_signing_key_id_{key_id}"]

    reg_info = TRUSTED_KEY_REGISTRY[key_id]

    # 1. 严格用途校验 (Purpose Isolation)
    allowed_purposes = reg_info.get("allowed_purposes", [])
    if required_purpose not in allowed_purposes:
        return False, [f"key_purpose_mismatch_{key_id}_not_authorized_for_{required_purpose}"]

    # 2. 状态与有效期检查
    valid_window, err_msg = _check_key_validity_window(reg_info, created_at_iso)
    if not valid_window:
        return False, [err_msg]

    # 3. 提取公钥
    pk_bytes = get_trusted_public_key(key_id)
    if not pk_bytes:
        return False, [f"inactive_or_invalid_public_key_for_{key_id}"]

    if not signature_hex or not re.match(r"^[0-9a-fA-F]{128}$", str(signature_hex).strip()):
        return False, ["invalid_or_missing_ed25519_signature_format_128_hex"]

    try:
        sig_bytes = bytes.fromhex(signature_hex.strip())
        msg_to_verify = f"{domain_separator}:".encode("utf-8") + message if domain_separator else message

        if _HAS_CRYPTOGRAPHY:
            pub = crypto_ed25519.Ed25519PublicKey.from_public_bytes(pk_bytes)
            try:
                pub.verify(sig_bytes, msg_to_verify)
                return True, []
            except InvalidSignature:
                errors.append("ed25519_cryptographic_signature_verification_failed")
                return False, errors
        else:
            if production_mode:
                return False, ["cryptography_library_required_for_production_verification"]
            is_valid = ed25519_verify_pure(sig_bytes, msg_to_verify, pk_bytes)
            if not is_valid:
                errors.append("ed25519_cryptographic_signature_verification_failed")
                return False, errors
            return True, []

    except Exception as e:
        return False, [f"signature_verification_error_{str(e)}"]
