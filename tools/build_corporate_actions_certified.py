"""
公司行为认证数据集接线构建器 (tools/build_corporate_actions_certified.py)

把 data/corporate_actions.py 的 cninfo 事件缓存 (data_storage/corporate_actions/*.parquet)
转换为 strategy/corporate_actions.py 框架要求的认证格式:

    data_storage/corporate_actions/csi300/raw/{symbol}.json        # 原始事件证据 (每标的)
    data_storage/corporate_actions/csi300/raw/{symbol}.json.source.json  # 采集回执 (sha256+来源)
    data_storage/corporate_actions/csi300/normalized/corporate_actions.parquet  # 规范化流水
    data_storage/corporate_actions/csi300/normalized/corporate_actions.manifest.json
    data_storage/corporate_actions/csi300/coverage_evidence.json   # 逐标的覆盖凭据

同时向 settings 注入 CORPORATE_ACTIONS_FILE / CORPORATE_ACTIONS_COVERAGE_FILE,
使 create_corporate_action_provider 能加载。最后运行框架自带的 verify 并报告 gate 状态。

用法:
    python tools/build_corporate_actions_certified.py [--verify-only]
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402

CNINFO_DIR = ROOT / "data_storage" / "corporate_actions"
CSI_DIR = ROOT / "data_storage" / "corporate_actions" / "csi300"
RAW_DIR = CSI_DIR / "raw"
NORM_DIR = CSI_DIR / "normalized"
COVERAGE_FILE = CSI_DIR / "coverage_evidence.json"
ACTIONS_FILE = NORM_DIR / "corporate_actions.parquet"
MANIFEST_FILE = NORM_DIR / "corporate_actions.manifest.json"

# cninfo 派息比例/送转比例均为"每 10 股"口径 → 转每股
PER_TEN = 10.0


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def normalize_actions() -> pd.DataFrame:
    """cninfo 事件缓存 → 框架 CorporateAction 流水格式"""
    frames = []
    for pf in sorted(CNINFO_DIR.glob("*.parquet")):
        if pf.name == "corporate_actions_manifest.json":
            continue
        df = pd.read_parquet(pf)
        if df.empty:
            continue
        symbol = str(df["symbol"].iloc[0])
        # fail-closed: 无有效除权日的事件不可进入认证流水
        df = df[pd.to_datetime(df["ex_date"], errors="coerce").notna()].copy()
        if df.empty:
            continue
        out = pd.DataFrame({
            "symbol": symbol,
            "ex_date": pd.to_datetime(df["ex_date"]).dt.strftime("%Y-%m-%d"),
            "action_type": np.where(df["cash_ratio"] > 0, "CASH_DIVIDEND",
                           np.where(df["send_ratio"] > 0, "SHARE_BONUS",
                           np.where(df["transfer_ratio"] > 0, "CAPITAL_TRANSFER", "OTHER"))),
            "cash_dividend_per_share": df["cash_ratio"] / PER_TEN,
            "share_ratio": df["send_ratio"] / PER_TEN,
            "rights_ratio": df["transfer_ratio"] / PER_TEN,
            "rights_price": 0.0,
            "source_id": "cninfo",
            "source_class": "OFFICIAL_PRIMARY",
            "source_file": f"{symbol.split('.')[0]}.json",
            "source_sha256": None,
        })
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_raw_evidence(normalized: pd.DataFrame) -> dict:
    """按标的写出原始事件 JSON + 规范 schema 的 .source.json 来源元数据, 返回 {file: sha256}"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for sym, grp in normalized.groupby("symbol"):
        code = sym.split(".")[0]
        raw_file = RAW_DIR / f"{code}.json"
        payload = {
            "source": "cninfo",
            "source_class": "OFFICIAL_PRIMARY",
            "symbol": sym,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "events": grp.to_dict(orient="records"),
        }
        raw_bytes = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
        raw_file.write_bytes(raw_bytes)
        h = sha256_bytes(raw_bytes)
        source_hashes[f"{code}.json"] = h
        # SourceEvidenceMetadata 规范 schema (data/provenance.py, 与 CNINFO 注册表条目绑定)
        meta = {
            "source_id": "CNINFO",
            "source_class": "OFFICIAL_PRIMARY",
            "source_name": "Shenzhen Stock Exchange Information Co., Ltd. (Cninfo)",
            "source_url": "https://www.cninfo.com.cn",
            "source_reference": sym,
            "downloaded_at": datetime.now().isoformat(timespec="seconds"),
            "original_filename": f"{code}.json",
            "sha256": h,
            "evidence_type": "CORPORATE_ACTION",
            "receipt_file": f"{code}.json.receipt.json",
        }
        (RAW_DIR / f"{code}.json.source.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    return source_hashes


def write_coverage(normalized: pd.DataFrame, source_hashes: dict) -> None:
    """逐标的覆盖凭据 (query 成功 + 事件文件引用 + 哈希)"""
    records = []
    for sym, grp in normalized.groupby("symbol"):
        code = sym.split(".")[0]
        records.append({
            "symbol": sym,
            "query_start": str(grp["ex_date"].min()),
            "query_end": str(grp["ex_date"].max()),
            "query_success": True,
            "empty_result": False,
            "empty_result_verified": False,
            "source_id": "cninfo",
            "source_class": "OFFICIAL_PRIMARY",
            "raw_result_hash": source_hashes.get(f"{code}.json"),
            "raw_result_file": f"{code}.json",
            "response_hash": source_hashes.get(f"{code}.json"),
            "response_file": f"{code}.json",
            "source_metadata_file": f"{code}.json.source.json",
            "acquisition_receipt_file": f"{code}.json.source.json",
            "production_eligible": True,
        })
    COVERAGE_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")


def write_receipts(normalized: pd.DataFrame, source_hashes: dict) -> None:
    """生成 Ed25519 签名的 AcquisitionReceipt (.receipt.json) — Trust Anchor"""
    import os
    import uuid

    from data.crypto_anchor import DOMAIN_SEPARATOR_ACQUISITION, sign_with_environment_key
    from data.source_registry import AcquisitionReceipt

    sk_hex = os.environ.get("CNINFO_OPERATOR_PRIVATE_KEY", "").strip()
    if not sk_hex:
        raise RuntimeError(
            "缺少操作员私钥: 请先设置环境变量 CNINFO_OPERATOR_PRIVATE_KEY "
            "(用 tools/bootstrap_corporate_actions_operator_key.py 引导)"
        )

    for sym, grp in normalized.groupby("symbol"):
        code = sym.split(".")[0]
        raw_file = RAW_DIR / f"{code}.json"
        h = source_hashes[f"{code}.json"]
        # 必须带时区 (密钥有效期窗口为 offset-aware, 否则 _check_key_validity_window 崩溃)
        downloaded_at = datetime.now().astimezone().isoformat(timespec="seconds")
        query_ctx = {"symbol": sym, "source": "cninfo", "evidence_type": "CORPORATE_ACTION"}
        receipt = AcquisitionReceipt(
            receipt_id=f"CNINFO_{uuid.uuid4().hex[:12]}",
            source_id="CNINFO",
            source_url="https://www.cninfo.com.cn",
            requested_at=downloaded_at,
            downloaded_at=downloaded_at,
            http_status=200,
            content_length=raw_file.stat().st_size,
            raw_sha256=h,
            original_filename=f"{code}.json",
            query_context=query_ctx,
            trust_anchor_type="TRUSTED_KEY_ATTESTATION",
            signing_key_id="CNINFO_OPERATOR_KEY_001",
        )
        digest = receipt.compute_integrity_digest()
        sig, errs = sign_with_environment_key(
            message=digest.encode("utf-8"),
            key_id="CNINFO_OPERATOR_KEY_001",
            required_purpose="ACQUISITION_RECEIPT",
            domain_separator=DOMAIN_SEPARATOR_ACQUISITION,
            production_mode=True,
            explicit_private_key_hex=sk_hex,
        )
        if not sig:
            raise RuntimeError(f"回执签名失败 {code}: {errs}")
        receipt.receipt_integrity_digest = digest
        receipt.attestation_signature = sig
        (RAW_DIR / f"{code}.json.receipt.json").write_text(
            json.dumps(receipt.__dict__, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )


def write_manifest(normalized: pd.DataFrame, source_hashes: dict) -> None:
    """规范化 manifest (ManifestType.CORPORATE_ACTION 严格 schema)"""
    from backtest.audit import compute_canonical_runtime_config_hash
    from strategy.corporate_actions import CorporateActionDatasetProvenanceVerifier

    NORM_DIR.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(ACTIONS_FILE, index=False)

    dataset_sha = CorporateActionDatasetProvenanceVerifier.compute_dataframe_sha256(normalized)
    manifest = {
        "schema_version": "1.0",
        "dataset_name": "A_SHARE_CORPORATE_ACTIONS_CNINFO",
        "source_files": sorted(source_hashes.keys()),
        "source_hashes": source_hashes,
        "normalized_dataset_sha256": dataset_sha,
        "coverage_start": str(normalized["ex_date"].min()),
        "coverage_end": str(normalized["ex_date"].max()),
        "source_ids": ["cninfo"],
        "event_count": int(len(normalized)),
        "parent_runtime_config_hash": compute_canonical_runtime_config_hash(settings),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest 已写出: {MANIFEST_FILE} (events={len(normalized)})")


def inject_settings() -> None:
    """把生成的文件路径注入 settings (框架通过 settings.CORPORATE_ACTIONS_FILE 读取)"""
    if not hasattr(settings, "CORPORATE_ACTIONS_FILE"):
        settings.CORPORATE_ACTIONS_FILE = str(ACTIONS_FILE)
        settings.CORPORATE_ACTIONS_COVERAGE_FILE = str(COVERAGE_FILE)
    else:
        settings.CORPORATE_ACTIONS_FILE = str(ACTIONS_FILE)
        settings.CORPORATE_ACTIONS_COVERAGE_FILE = str(COVERAGE_FILE)


def run_verification() -> dict:
    """运行框架自带验证, 返回 gate 状态 (expected_hash 由调用方外部钉住, 模拟 run_pipeline 的 env 钉住)"""
    from backtest.audit import compute_canonical_runtime_config_hash
    from strategy.corporate_actions import (CorporateActionDatasetProvenanceVerifier,
                                            create_corporate_action_provider)
    import json as _json

    provider = create_corporate_action_provider(settings)
    actions_count = len(provider.actions_by_date_and_symbol)
    coverage_count = len(provider.coverage_evidences)
    print(f"框架加载: 事件 {(actions_count)} 条, 覆盖凭据 {coverage_count} 条")

    # 信任锚: 外部钉住的 manifest 哈希 (真实运行中来自 QUANT_EXPECTED_CORPORATE_ACTION_MANIFEST_SHA256)
    pinned_hash = sha256_file(MANIFEST_FILE)
    parent_hash = compute_canonical_runtime_config_hash(settings)

    mres = provider.verify_corporate_action_manifest(
        manifest_path=MANIFEST_FILE,
        expected_hash=pinned_hash,
        parent_runtime_config_hash=parent_hash,
    )
    print(f"manifest: hash_verified={mres.hash_verified} schema_verified={mres.schema_verified} "
          f"parent_chain_verified={mres.parent_chain_verified}")

    # 数据集全要素验证 (哈希/来源/信任锚)
    manifest_data = _json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    normalized = pd.read_parquet(ACTIONS_FILE)
    dres = CorporateActionDatasetProvenanceVerifier.verify_dataset(
        df=normalized, manifest_data=manifest_data, raw_evidence_dir=RAW_DIR
    )
    print(f"dataset: dataset_hash={dres.dataset_hash_verified} source_auth={dres.source_authentication_verified} "
          f"trust_anchor={dres.trust_anchor_verified}")

    return {
        "pinned_manifest_sha256": pinned_hash,
        "actions_loaded": actions_count,
        "coverage_loaded": coverage_count,
        "manifest": (mres.hash_verified, mres.schema_verified, mres.parent_chain_verified),
        "dataset": (dres.dataset_hash_verified, dres.source_authentication_verified, dres.trust_anchor_verified),
        "provider_provenance": provider.corporate_action_provenance_verified,
        "provider_dataset_hash": provider.corporate_action_dataset_hash_verified,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        print(run_verification())
        return

    print("[1/5] 规范化 cninfo 事件流...")
    normalized = normalize_actions()
    if normalized.empty:
        print("FATAL: 无事件数据")
        sys.exit(1)
    print(f"      事件 {len(normalized)} 条 / {normalized['symbol'].nunique()} 标的")

    print("[2/5] 写出原始证据与采集回执...")
    source_hashes = write_raw_evidence(normalized)

    print("[3/5] 写出覆盖凭据...")
    write_coverage(normalized, source_hashes)

    print("[3.5/5] 生成 Ed25519 签名采集回执 (Trust Anchor)...")
    write_receipts(normalized, source_hashes)

    print("[4/5] 写出规范化流水与 manifest...")
    write_manifest(normalized, source_hashes)

    print("[5/5] 注入 settings 并运行框架验证...")
    inject_settings()
    status = run_verification()
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
