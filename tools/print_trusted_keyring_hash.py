"""
受信任公钥环哈希计算与人工外部锚定辅助工具 (tools/print_trusted_keyring_hash.py)
用于计算当前 TRUSTED_KEY_REGISTRY 的 Canonical SHA256 哈希值。
注意：本脚本仅用于人工审计与环境初始化，严禁任何 Agent 或自动化流水线自动回填环境变量！
"""
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from data.crypto_anchor import compute_canonical_keyring_hash, verify_trust_root


def main():
    h = compute_canonical_keyring_hash()
    is_verified, actual, pin, errors = verify_trust_root()
    print("=================================================================")
    print("🛡️ A股量化系统 · 外部 Trust Root 公钥环指纹计算器")
    print("=================================================================")
    print(f"当前仓库公钥环 Canonical SHA256:\n  {h}\n")
    if pin:
        print(f"当前环境变量 QUANT_TRUSTED_KEYRING_SHA256:\n  {pin}\n")
        print(f"锚定状态: {'✅ VERIFIED (一致)' if is_verified else '❌ MISMATCH (已偏离)'}")
    else:
        print("当前环境变量 QUANT_TRUSTED_KEYRING_SHA256: [未设置]")
        print("生产环境配置方法 (Windows PowerShell):")
        print(f'  $env:QUANT_TRUSTED_KEYRING_SHA256 = "{h}"')
        print("生产环境配置方法 (Linux/macOS Bash):")
        print(f'  export QUANT_TRUSTED_KEYRING_SHA256="{h}"')
    print("=================================================================")


if __name__ == "__main__":
    main()
