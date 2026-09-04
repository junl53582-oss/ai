"""
模型生命周期管理 CLI (tools/promote_model.py)

提供注册表查询与晋升/归档操作, 每次晋升强制记录审批人与证据 (fail-closed)。

用法:
    python tools/promote_model.py list [--state PRODUCTION]
    python tools/promote_model.py register --artifact <pkl路径> --model-type lightgbm \
        --dataset-sha <sha256> --metrics-json <metrics.json>
    python tools/promote_model.py promote --model-id m_20260901_120000_lightgbm --to CANDIDATE --approver linjun
    python tools/promote_model.py promote --model-id <id> --to APPROVED --approver linjun \
        --certification-ref reports/model_research/certification.json
    python tools/promote_model.py promote --model-id <id> --to PRODUCTION --approver linjun \
        --certification-ref <path> --prospective --paper-trading
    python tools/promote_model.py archive --model-id <id> --approver linjun
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.registry import ModelRegistry, ModelState, PromotionError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("PromoteModel")


def _parse_evidence(args) -> dict:
    ev = {}
    if getattr(args, "approval_artifact", None):
        art_path = Path(args.approval_artifact)
        if not art_path.exists():
            raise FileNotFoundError(f"审批凭证制品不存在: {art_path}")
        data = json.loads(art_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"审批凭证制品格式错误 (必须为 JSON 对象): {art_path}")
        # 从外部审批制品中解析认证与实盘证据
        if "certification_ref" in data:
            ev["certification_ref"] = data["certification_ref"]
        if "prospective_validation" in data:
            ev["prospective_validation"] = data["prospective_validation"]
        if "paper_trading" in data:
            ev["paper_trading"] = data["paper_trading"]
        if "evidence" in data and isinstance(data["evidence"], dict):
            ev.update(data["evidence"])

    if getattr(args, "certification_ref", None):
        ev["certification_ref"] = args.certification_ref
    if getattr(args, "prospective", False):
        ev["prospective_validation"] = {"ref": getattr(args, "prospective_ref", None) or "manual_claim"}
    if getattr(args, "paper_trading", False):
        ev["paper_trading"] = {"ref": getattr(args, "paper_ref", None) or "manual_claim"}
    return ev


def main():
    parser = argparse.ArgumentParser(description="模型注册表与生命周期管理")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出模型记录")
    p_list.add_argument("--state", type=str, default=None)

    p_reg = sub.add_parser("register", help="登记研究制品 (状态固定为 RESEARCH)")
    p_reg.add_argument("--artifact", type=str, required=True)
    p_reg.add_argument("--model-type", type=str, default="lightgbm")
    p_reg.add_argument("--task-type", type=str, default="classification")
    p_reg.add_argument("--dataset-sha", type=str, default=None)
    p_reg.add_argument("--dataset-path", type=str, default=None)
    p_reg.add_argument("--metrics-json", type=str, default=None, help="OOS 指标 JSON 文件路径")
    p_reg.add_argument("--notes", type=str, default="")

    p_promote = sub.add_parser("promote", help="晋升模型状态")
    p_promote.add_argument("--model-id", type=str, required=True)
    p_promote.add_argument("--to", type=str, required=True, choices=ModelState.all_states())
    p_promote.add_argument("--approver", type=str, default=None, help="审批人 (人工签名)")
    p_promote.add_argument("--approval-artifact", type=str, default=None, help="外部审批凭证 JSON 制品路径 (由投研委员会或负责人签署)")
    p_promote.add_argument("--certification-ref", type=str, default=None)
    p_promote.add_argument("--prospective", action="store_true", help="提供前瞻验证证据")
    p_promote.add_argument("--prospective-ref", type=str, default=None)
    p_promote.add_argument("--paper-trading", action="store_true", help="提供模拟盘证据")
    p_promote.add_argument("--paper-ref", type=str, default=None)
    p_promote.add_argument("--note", type=str, default="")

    p_arch = sub.add_parser("archive", help="归档模型 (PRODUCTION -> ARCHIVED)")
    p_arch.add_argument("--model-id", type=str, required=True)
    p_arch.add_argument("--approver", type=str, required=True)

    args = parser.parse_args()
    registry = ModelRegistry()

    if args.cmd == "list":
        recs = registry.list_records(state=args.state)
        if not recs:
            print("(无记录)")
            return
        print(f"{'MODEL_ID':<34}{'STATE':<12}{'TYPE':<20}{'DATASET':<10}{'CREATED'}")
        for r in recs:
            ds = (r.dataset_sha256 or "")[:8]
            print(f"{r.model_id:<34}{r.state:<12}{r.model_type:<20}{ds:<10}{r.created_at}")
        return

    if args.cmd == "register":
        metrics = {}
        if args.metrics_json:
            metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
        mid = registry.register_research_artifact(
            artifact_path=Path(args.artifact),
            model_type=args.model_type,
            task_type=args.task_type,
            metrics=metrics,
            dataset_sha256=args.dataset_sha,
            dataset_path=args.dataset_path,
            notes=args.notes,
        )
        print(f"已登记: {mid}")
        return

    if args.cmd in ("promote", "archive"):
        to_state = args.to if args.cmd == "promote" else ModelState.ARCHIVED
        approver = getattr(args, "approver", None)
        if not approver and getattr(args, "approval_artifact", None):
            try:
                art_data = json.loads(Path(args.approval_artifact).read_text(encoding="utf-8"))
                approver = art_data.get("approver")
            except Exception:
                pass
        if not approver:
            logger.error("晋升失败: 缺少审批人签名 (必须通过 --approver 或 --approval-artifact 指定)")
            sys.exit(1)

        try:
            rec = registry.promote(
                model_id=args.model_id,
                to_state=to_state,
                approver=approver,
                evidence=_parse_evidence(args),
                note=args.note,
            )
            print(f"OK: {rec.model_id} -> {rec.state}")
        except PromotionError as e:
            logger.error(f"晋升被拒绝 (fail-closed): {e}")
            sys.exit(1)
        return


if __name__ == "__main__":
    main()
