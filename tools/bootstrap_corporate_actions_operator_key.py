"""
公司行为操作员密钥引导工具 (tools/bootstrap_corporate_actions_operator_key.py)

用途: 生成/轮换 CNINFO 公司行为采集回执的 Ed25519 操作员密钥, 并打印:
    1. 需要登记进 data/crypto_anchor.py TRUSTED_KEY_REGISTRY 的条目
    2. 需要设置的环境变量 (私钥 + 外部信任根钉住)

注意:
    - 私钥是操作员秘密, 只应保存在环境变量/密钥管理器中, 严禁提交进仓库
    - 每次更换密钥都会改变 keyring 哈希, 必须同步更新 QUANT_TRUSTED_KEYRING_SHA256
    - 生成后请手工把公钥条目加入 crypto_anchor.py 并提交 (公钥公开无风险)

用法:
    python tools/bootstrap_corporate_actions_operator_key.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.crypto_anchor import compute_canonical_keyring_hash, generate_keypair  # noqa: E402


def main():
    sk, pk = generate_keypair()
    sk_hex = sk.hex()
    pk_hex = pk.hex()

    print("=" * 70)
    print("CNINFO 公司行为操作员 Ed25519 密钥对 (已生成, 未持久化)")
    print("=" * 70)
    print()
    print("【1】把以下公钥条目登记进 data/crypto_anchor.py 的 TRUSTED_KEY_REGISTRY:")
    print(f"""
    "CNINFO_OPERATOR_KEY_001": {{
        "algorithm": "ED25519",
        "key_id": "CNINFO_OPERATOR_KEY_001",
        "public_key_hex": "{pk_hex}",
        "allowed_purposes": ["ACQUISITION_RECEIPT"],
        "issuer_type": "PROJECT",
        "institution": "CNINFO_DISCLOSURE_OPERATOR",
        "env_private_key_var": "CNINFO_OPERATOR_PRIVATE_KEY",
        "status": "ACTIVE",
        "not_before": "2025-01-01T00:00:00Z",
        "not_after": "2030-01-01T00:00:00Z",
        "is_production": True
    }},""")
    print()
    print("【2】设置环境变量 (私钥仅存环境, 严禁入库):")
    print(f"  export CNINFO_OPERATOR_PRIVATE_KEY={sk_hex}")
    print(f"  export QUANT_TRUSTED_KEYRING_SHA256={compute_canonical_keyring_hash()}")
    print()
    print("【3】重新生成公司行为证据与签名回执:")
    print("  python tools/build_corporate_actions_certified.py")
    print()


if __name__ == "__main__":
    main()
