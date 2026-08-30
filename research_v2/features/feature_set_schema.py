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
        if not self.created_from_commit:
            raise ValueError("created_from_commit is required")
        if not isinstance(self.feature_groups, dict) or not self.feature_groups:
            raise ValueError("feature_groups must be a non-empty mapping")
        if not isinstance(self.feature_names, list) or not self.feature_names:
            raise ValueError("feature_names must be a non-empty list")
        if any(not isinstance(name, str) or not name for name in self.feature_names):
            raise ValueError("feature_names must contain non-empty strings")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Duplicate feature names are not allowed")

        all_group_feats: List[str] = []
        for group_name, group_feats in self.feature_groups.items():
            if not isinstance(group_name, str) or not group_name:
                raise ValueError("Feature group names must be non-empty strings")
            if not isinstance(group_feats, list):
                raise ValueError(
                    f"Feature group '{group_name}' must contain a list"
                )
            if len(set(group_feats)) != len(group_feats):
                raise ValueError(
                    f"Duplicate features found inside group '{group_name}'"
                )
            all_group_feats.extend(group_feats)

        # 一个特征只能属于一个 group；否则 schema 的分组语义不唯一。
        if len(set(all_group_feats)) != len(all_group_feats):
            raise ValueError("A feature may not appear in multiple feature groups")

        if all_group_feats != self.feature_names:
            raise ValueError(
                "Feature names/order in groups do not match flat feature_names list"
            )
        if len(self.feature_names) != self.feature_count:
            raise ValueError(
                f"Feature count mismatch: {len(self.feature_names)} vs "
                f"{self.feature_count}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_hash(self) -> str:
        """
        对完整、顺序敏感的 feature schema 做 canonical SHA256。

        旧实现仅对排序后的 feature_names 做 Hash，会让 [f1,f2] 与 [f2,f1]
        得到同一 Hash，也忽略 group/commit 元数据，无法严格保证训练输入一致。
        """
        self.validate()
        canonical_payload = {
            "feature_set_id": self.feature_set_id,
            # dict key 排序交给 json.dumps(sort_keys=True)，每组内部顺序保留。
            "feature_groups": self.feature_groups,
            "feature_names": self.feature_names,
            "feature_count": self.feature_count,
            "created_from_commit": self.created_from_commit,
        }
        raw_json = json.dumps(
            canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        h = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
        self.feature_hash = h
        return h
