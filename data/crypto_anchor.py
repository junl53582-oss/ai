"""
企业级独立密码学信任锚点与非对称数字签名引擎 (data/crypto_anchor.py)
实现符合 RFC 8032 标准的纯 Python Ed25519 密码学签名与验签引擎：
最高原则：
1. PUBLIC KEYS ONLY IN REPO: 仓库注册表仅保存公钥与 Key ID，绝无任何私钥或生产 Secret。
2. ASYMMETRIC VERIFICATION: 验签仅依赖公开公钥，任何攻击者无法仅凭源码推导或伪造合法签名。
3. ZERO DEFAULT SECRETS: 若运行环境未注入安全私钥 (OS Secret / HSM / 环境变量)，系统拒绝假冒签名。
4. FAIL-CLOSED: 伪造签名、公钥不匹配或自算哈希一律判定为 False。
"""
import os
import re
import json
import hashlib
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path


# =========================================================================
# 1. RFC 8032 Ed25519 纯 Python 密码学实现 (Edwards-curve Digital Signatures)
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


def ed25519_publickey(sk: bytes) -> bytes:
    """由 32 字节私钥生成 32 字节公钥"""
    h = _H(sk)
    a = 2**254 + sum(2**i * ((h[i // 8] >> (i % 8)) & 1) for i in range(3, 254))
    A = _scalarmult(_B, a)
    return _encodepoint(A)


def ed25519_sign(m: bytes, sk: bytes, pk: bytes) -> bytes:
    """使用 32 字节私钥与公钥对消息生成 64 字节 Ed25519 数字签名"""
    h = _H(sk)
    a = 2**254 + sum(2**i * ((h[i // 8] >> (i % 8)) & 1) for i in range(3, 254))
    r = int.from_bytes(_H(h[32:] + m), "little") % _q
    R = _scalarmult(_B, r)
    R_enc = _encodepoint(R)
    k = int.from_bytes(_H(R_enc + pk + m), "little") % _q
    S = (r + k * a) % _q
    return R_enc + _encodeint(S)


def ed25519_verify(sig: bytes, m: bytes, pk: bytes) -> bool:
    """使用 32 字节公钥验证 64 字节 Ed25519 数字签名"""
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
    sk = os.urandom(32)
    pk = ed25519_publickey(sk)
    return sk, pk


# =========================================================================
# 2. 受信任公钥注册表 (PUBLIC KEYS ONLY - 严禁在代码中硬编码任何私钥)
# =========================================================================

TRUSTED_KEY_REGISTRY: Dict[str, Dict[str, Any]] = {
    # 生产官方数据采集信任公钥 (中证/交易所原始数据同步签名)
    "PROD_DOWNLOADER_KEY_2026_V1": {
        "algorithm": "ED25519",
        "key_id": "PROD_DOWNLOADER_KEY_2026_V1",
        "public_key_hex": "4d16d000632d43105ff7f6e3c048bcfa28ebc91a08e1bd074f7be33f2e105d15",
        "env_private_key_var": "QUANT_PROD_ACQUISITION_PRIVATE_KEY",
        "institution": "CHINA_SECURITIES_INDEX_AUTHENTICATED_GATEWAY",
        "status": "ACTIVE",
        "is_production": True
    },
    # 生产运行时回测审计数字信封签名公钥
    "PROD_RUNTIME_KEY_2026_V1": {
        "algorithm": "ED25519",
        "key_id": "PROD_RUNTIME_KEY_2026_V1",
        "public_key_hex": "9a38549646b95ee36b22b64d1f2fb280c7d5c7c258d46d0a7a372691361c77f0",
        "env_private_key_var": "QUANT_PROD_RUNTIME_PRIVATE_KEY",
        "institution": "QUANT_INFRA_ATTESTATION_AUTHORITY",
        "status": "ACTIVE",
        "is_production": True
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


def sign_with_environment_key(
    message: bytes,
    key_id: str = "PROD_RUNTIME_KEY_2026_V1",
    explicit_private_key_hex: Optional[str] = None
) -> Tuple[Optional[str], List[str]]:
    """
    使用环境变量或显式传入的私钥对数据进行 Ed25519 签名。
    若生产私钥未在环境变量提供，则严格返回 None 并报错 (Fail-Closed)。
    """
    if key_id not in TRUSTED_KEY_REGISTRY:
        return None, [f"unregistered_signing_key_id_{key_id}"]

    reg_info = TRUSTED_KEY_REGISTRY[key_id]
    env_var = reg_info.get("env_private_key_var", "")

    sk_hex = explicit_private_key_hex or (os.environ.get(env_var) if env_var else None)
    if not sk_hex:
        return None, [f"missing_private_key_in_environment_for_{key_id}"]

    try:
        sk = bytes.fromhex(sk_hex.strip())
        if len(sk) != 32:
            return None, ["invalid_private_key_length_must_be_32_bytes"]
        pk = ed25519_publickey(sk)
        expected_pk_hex = reg_info.get("public_key_hex", "").lower()
        if pk.hex().lower() != expected_pk_hex:
            return None, ["private_key_does_not_match_registered_public_key"]

        sig = ed25519_sign(message, sk, pk)
        return sig.hex(), []
    except Exception as e:
        return None, [f"signing_failed_{str(e)}"]


def verify_ed25519_signature(
    message: bytes,
    signature_hex: str,
    key_id: str
) -> Tuple[bool, List[str]]:
    """
    使用仓库中注册的公开公钥对 Ed25519 数字签名进行密码学严格验签。
    攻击者即便拥有整个源码与公开公钥，也绝无法在没有私钥的情况下伪造此签名。
    """
    errors = []
    if not key_id or key_id not in TRUSTED_KEY_REGISTRY:
        return False, [f"unregistered_signing_key_id_{key_id}"]

    pk_bytes = get_trusted_public_key(key_id)
    if not pk_bytes:
        return False, [f"inactive_or_invalid_public_key_for_{key_id}"]

    if not signature_hex or not re.match(r"^[0-9a-fA-F]{128}$", str(signature_hex).strip()):
        return False, ["invalid_or_missing_ed25519_signature_format_128_hex"]

    try:
        sig_bytes = bytes.fromhex(signature_hex.strip())
        is_valid = ed25519_verify(sig_bytes, message, pk_bytes)
        if not is_valid:
            errors.append("ed25519_cryptographic_signature_verification_failed")
            return False, errors
        return True, []
    except Exception as e:
        return False, [f"signature_verification_error_{str(e)}"]
