"""
沪深300历史成分股 Point-in-Time 数据解析与规范化工具 (tools/build_pit_universe.py)
严格遵循 Single-Source-of-Truth、Fail-Closed 与 Anti-Impersonation 原则：
- 仅从 data_storage/universe/csi300/raw/ 目录解析官方/持牌数据源的原始文件 (Raw Evidence)
- 每个 Raw Evidence 必须存在独立的 .source.json 来源认证描述文件，Parser 绝不自行指定或猜测 OFFICIAL_PRIMARY
- Baseline Snapshot 必须来自独立的基线快照文件 (evidence_type: BASELINE_SNAPSHOT)，严禁由第一条调样事件推导
- 若无原始数据或缺少合规的 Source Metadata，直接抛出 DataProvenanceError 并拒绝生成伪造 Manifest
"""
import sys
import io
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.settings import settings
from data.provenance import (
    SourceClass,
    DataProvenanceError,
    ProvenanceVerifier,
    SourceEvidenceMetadata,
    CSIRebalanceAnnouncementParser,
    CSIConstituentSnapshotParser
)


def build_csi300_pit_universe_from_raw(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fail_closed: bool = True
) -> Dict[str, Any]:
    """
    从真实 Raw Evidence 文件与独立 Source Metadata 解析并生成规范化 PIT 历史成分股事件
    生产环境严格禁止生成任何模拟历史成分股，亦严禁 Parser 伪造 OFFICIAL 标签。
    """
    base_raw = raw_dir or (settings.DATA_DIR / "universe" / "csi300" / "raw")
    base_out = output_dir or (settings.DATA_DIR / "universe" / "csi300" / "normalized")
    base_out.mkdir(parents=True, exist_ok=True)

    if not base_raw.exists():
        if fail_closed:
            raise DataProvenanceError(
                f"❌ 原始数据血缘目录不存在: {base_raw}。\n"
                "生产环境严禁脱离真实 Raw Evidence 伪造历史成分变动数据！"
            )
        return {"status": "FAILED", "reason": "RAW_DIR_NOT_FOUND"}

    # 扫描 Raw Evidence 原始文件 (过滤掉 .source.json)
    all_files = list(base_raw.glob("*.csv")) + list(base_raw.glob("*.json")) + list(base_raw.glob("*.parquet"))
    raw_files = [f for f in all_files if not f.name.endswith(".source.json") and not f.name.endswith(".manifest.json")]

    if not raw_files:
        if fail_closed:
            raise DataProvenanceError(
                f"❌ 在 {base_raw} 下未发现任何可信原始证据文件 (Raw Evidence)。\n"
                "原则: NO EVIDENCE => NO VERIFIED (严禁自造历史事件并自发凭证)。"
            )
        return {"status": "FAILED", "reason": "NO_RAW_FILES"}

    print(f">> 发现 {len(raw_files)} 个原始数据文件，开始校验独立 Source Metadata...")
    parsed_events: List[Dict[str, Any]] = []
    raw_hashes: Dict[str, str] = {}
    verified_source_classes: Set[str] = set()

    baseline_snapshot_file: Optional[str] = None
    baseline_snapshot_date: Optional[str] = None
    baseline_symbols: List[str] = []
    baseline_snapshot_sha256: Optional[str] = None

    rebal_parser = CSIRebalanceAnnouncementParser()
    snap_parser = CSIConstituentSnapshotParser()

    for r_file in raw_files:
        f_hash = ProvenanceVerifier.compute_file_sha256(r_file)
        raw_hashes[r_file.name] = f_hash

        # P0-1: 校验独立来源元数据文件 .source.json
        meta_file = r_file.with_name(f"{r_file.name}.source.json")
        source_meta, meta_errs = SourceEvidenceMetadata.load_and_verify(meta_file, r_file)

        if meta_errs or source_meta is None:
            if fail_closed:
                raise DataProvenanceError(
                    f"❌ 原始数据文件 {r_file.name} 缺少合规的独立来源认证描述 (.source.json):\n"
                    f"错误详情: {meta_errs}\n"
                    "原则: SHA256 仅证明无篡改，不证明来自官方。必须提供经过鉴证的 Source Metadata！"
                )
            source_class_val = SourceClass.UNKNOWN.value
            source_meta = SourceEvidenceMetadata(
                source_class=SourceClass.UNKNOWN,
                source_name="UNKNOWN_UNVERIFIED",
                original_filename=r_file.name,
                sha256=f_hash
            )
        else:
            source_class_val = source_meta.source_class.value

        verified_source_classes.add(source_class_val)

        # P1-1: 区分基线快照与定期调样
        if source_meta.evidence_type == "BASELINE_SNAPSHOT":
            events = snap_parser.parse(r_file, source_meta)
            if events:
                baseline_snapshot_file = r_file.name
                baseline_snapshot_date = events[0]["effective_date"]
                baseline_symbols = sorted(list(set(e["symbol"] for e in events)))
                baseline_snapshot_sha256 = f_hash
                parsed_events.extend(events)
        else:
            events = rebal_parser.parse(r_file, source_meta)
            parsed_events.extend(events)

    if not parsed_events:
        if fail_closed:
            raise DataProvenanceError("❌ 未能从原始数据文件中解析出任何有效事件。")
        return {"status": "FAILED", "reason": "EMPTY_PARSED_EVENTS"}

    if not baseline_snapshot_file or not baseline_symbols:
        if fail_closed:
            raise DataProvenanceError(
                "❌ 缺少独立的基线成分股快照文件 (evidence_type: BASELINE_SNAPSHOT)。\n"
                "原则: P1-1 严禁由第一条调样事件自动猜测 Baseline Snapshot！"
            )

    events_df = pd.DataFrame(parsed_events)
    events_df.sort_values(by=["effective_date", "symbol"], inplace=True)
    events_df.drop_duplicates(subset=["effective_date", "symbol", "action"], inplace=True)
    events_df.reset_index(drop=True, inplace=True)

    # 导出规范化 Parquet
    out_parquet = base_out / "universe_pit_events.parquet"
    events_df.to_parquet(out_parquet, index=False)
    dataset_sha256 = ProvenanceVerifier.compute_dataframe_sha256(events_df)

    overall_source_class = (
        SourceClass.OFFICIAL_PRIMARY.value
        if verified_source_classes == {SourceClass.OFFICIAL_PRIMARY.value}
        else (
            SourceClass.LICENSED_VENDOR.value
            if all(sc in [SourceClass.OFFICIAL_PRIMARY.value, SourceClass.LICENSED_VENDOR.value] for sc in verified_source_classes)
            else SourceClass.UNKNOWN.value
        )
    )

    # 生成事实 Manifest (绝不硬编码自我认证布尔值)
    manifest = {
        "dataset_name": "CSI300_POINT_IN_TIME_UNIVERSE",
        "dataset_version": "3.0",
        "index_code": "000300",
        "source_class": overall_source_class,
        "source_files": [f.name for f in raw_files],
        "raw_evidence_hashes": raw_hashes,
        "normalized_dataset_sha256": dataset_sha256,
        "baseline_snapshot_file": baseline_snapshot_file,
        "baseline_snapshot_date": baseline_snapshot_date,
        "baseline_snapshot_sha256": baseline_snapshot_sha256,
        "baseline_symbols": baseline_symbols,
        "baseline_symbol_count": len(baseline_symbols),
        "coverage_start": events_df["effective_date"].min(),
        "coverage_end": events_df["effective_date"].max(),
        "event_count": len(events_df),
        "parser_version": "3.0"
    }

    out_manifest = base_out / "universe_pit_events.manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"✅ 真实 PIT 事件流解析完成: {out_parquet} ({len(events_df)} 条经双向鉴证的真实记录)")
    return manifest


if __name__ == "__main__":
    try:
        build_csi300_pit_universe_from_raw()
    except DataProvenanceError as e:
        print(f"\n[Fail-Closed 保护触发]\n{e}")
        sys.exit(0)
