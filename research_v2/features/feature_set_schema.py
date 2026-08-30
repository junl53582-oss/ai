"""
Versioned Feature Set Schema (research_v2/features/feature_set_schema.py)
定义版本化特征集元数据架构，杜绝隐式全局特征定义。
"""
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List


@dataclass
class VersionedFeatureSetSchema:
    feature_set_id: str
    feature_groups: Dict[str, List[str]]
    feature_names: List[str]
    feature_count: int
    created_from_commit: str
    feature_hash: str = ""

    def validate(self) -> None:
        if not self.feature_set_id:
            raise ValueError("feature_set_id is required")
        all_group_feats = []
        for g_feats in self.feature_groups.values():
            all_group_feats.extend(g_feats)
        if set(all_group_feats) != set(self.feature_names):
            raise ValueError("Feature names in groups do not match flat feature_names list")
        if len(self.feature_names) != self.feature_count:
            raise ValueError(f"Feature count mismatch: {len(self.feature_names)} vs {self.feature_count}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_hash(self) -> str:
        self.validate()
        sorted_feats = sorted(self.feature_names)
        raw_str = ",".join(sorted_feats)
        h = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
        self.feature_hash = h
        return h
