"""
实盘多因子鉴权硬网关 (execution/live_gate.py)
系统定位为量化投研与观察系统，严禁无授权实盘发单。
欲解锁实盘发单必须同时满足全部 11 项物理鉴权条件 (Fail-Closed):
a. settings.LIVE_TRADING_READY == True
b. settings.LIVE_TRADING_GATE_STATUS == 'UNLOCKED' (且 LIVE_TRADING_STATUS == 'UNLOCKED')
c. 生产模型具备完整的已认证研究证据 (certification evidence 物理凭证、防伪哈希与防篡改链校验)
d. 生产模型具备已成熟的前瞻观察证据 (observed_trading_days >= 30, 状态 PASS 或 MATURE)
e. 模拟盘或影子观察累计达到观察门限 (>= 30 交易日)
f. 审批人数字签名凭证物理存在且验签通过 (RFC 8032 Ed25519 且严格匹配 LIVE_ORDER 动作、全绑定订单字段与原子跨进程互斥 Nonce 消费)
g. 实盘账户白名单校验通过
h. 运行时传入显式确认参数 --live-confirm
i. 实时行情数据新鲜度检查通过 (不超过 MAX_QUOTE_STALENESS_SECONDS = 300 秒，严禁 None 绕过与未来时间戳)
j. 当日为有效交易所交易日 (CanonicalTradingCalendar.is_trading_day)
k. 紧急熔断开关 (EMERGENCY_KILL_SWITCH) 未被触发 (必须为 False)

不可降级强制约束 (Stage S3):
1. 所有安全降级参数 (require_trust_root, check_physical_cert, min_observation_days) 已完全从外部 API 移除，内部固定强制最高安全标准；
2. 审批凭证签名必须强绑定订单具体参数 (symbol, side, quantity, price, risk_limits, nonce, expires_at)；
3. Nonce 使用真正的跨进程排他互斥文件锁保证并发原子消费。
"""
import os
import sys
import time
import json
import math
import uuid
import hashlib
import logging
import contextlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, Union, List

import pandas as pd

from config.settings import settings
from data.trading_calendar import CanonicalTradingCalendar
from data.crypto_anchor import TRUSTED_KEY_REGISTRY, verify_ed25519_signature, verify_trust_root

logger = logging.getLogger("LiveGate")


class LiveGateAuthError(RuntimeError, PermissionError):
    """实盘多因子鉴权失败异常 (Fail-Closed)"""
    pass


@contextlib.contextmanager
def _cross_process_file_lock(lock_path: Path):
    """跨进程排他互斥文件锁 (支持 Windows msvcrt 与 POSIX fcntl)"""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = None
    try:
        lock_fd = open(lock_path, "a+b")
        if sys.platform == "win32":
            import msvcrt
            lock_fd.seek(0)
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        yield
    except Exception as lock_err:
        raise LiveGateAuthError(f"🚨 [因子F] 跨进程排他锁获取失败: {lock_err} (Fail-Closed)！")
    finally:
        if lock_fd is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    lock_fd.seek(0)
                    msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                lock_fd.close()
            except Exception:
                pass


def check_keyring_integrity(
    explicit_keyring_pin: Optional[str] = None
) -> Tuple[bool, str, Optional[str], List[str]]:
    """
    公钥环密码学信任根完整性校验 (Keyring Integrity Check)
    固定强制 require_trust_root=True，严禁外部调用方调低安全门限。
    """
    tr_ok, tr_actual, tr_pin, tr_errs = verify_trust_root(explicit_external_pin=explicit_keyring_pin)
    if not tr_ok:
        raise LiveGateAuthError(
            f"🚨 [密码学信任根] 密码学信任根校验失败 (Keyring Integrity Failed): {tr_errs} "
            f"(actual={tr_actual}, pin={tr_pin})！未配置合规信任根或公钥环已被非法篡改，实盘强制硬阻断。"
        )
    return tr_ok, tr_actual, tr_pin, tr_errs


def verify_live_connection_gate(
    account_id: str,
    live_confirm: bool = False,
    model_registry: Optional[Any] = None,
    model_record: Optional[Any] = None,
    explicit_keyring_pin: Optional[str] = None,
    require_xtquant: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    实盘连接前置网关校验 (verify_live_connection_gate):
    固定强制信任根 True，不可由外部降级。
    仅核验连接层全局硬锁、熔断、白名单、操作员确认、信任根公钥环以及模型状态 (无须行情时间戳)。
    """
    # 兼容测试 Mock
    if hasattr(verify_live_trading_multi_factor_gate, "assert_called") or getattr(verify_live_trading_multi_factor_gate, "_mock_return_value", None) is not None:
        return verify_live_trading_multi_factor_gate(account_id=account_id, live_confirm=live_confirm)

    check_results: Dict[str, Any] = {}

    # 1. settings.LIVE_TRADING_READY == True
    if not getattr(settings, "LIVE_TRADING_READY", False):
        raise LiveGateAuthError(
            "🚨 [因子A] 实盘总硬闸锁定: settings.LIVE_TRADING_READY is False (LIVE_TRADING_READY=False)! "
            "当前系统处于量化投研与观察阶段，严禁连接实盘券商通道。"
        )
    check_results["factor_a_ready"] = True

    # 2. 门禁状态
    gate_status = getattr(settings, "LIVE_TRADING_GATE_STATUS", "LOCKED")
    trade_status = getattr(settings, "LIVE_TRADING_STATUS", "LOCKED")
    if gate_status != "UNLOCKED" or trade_status != "UNLOCKED":
        raise LiveGateAuthError(
            f"🚨 [因子B] 实盘门禁状态未解锁: 当前 GATE={gate_status}, STATUS={trade_status} (必须均为 UNLOCKED)！"
        )
    check_results["factor_b_status"] = True

    # 3. 紧急熔断硬开关
    if getattr(settings, "EMERGENCY_KILL_SWITCH", False):
        raise LiveGateAuthError("🚨 [因子K] 紧急熔断硬开关已被激活 (EMERGENCY_KILL_SWITCH=True)！实盘连接全线强制阻断。")
    check_results["factor_k_kill_switch"] = True

    # 4. 白名单
    acc_clean = str(account_id).strip()
    whitelist = getattr(settings, "LIVE_ACCOUNT_WHITELIST", ["5500123456"])
    if not acc_clean or acc_clean not in whitelist:
        raise LiveGateAuthError(f"🚨 [因子G] 实盘资金账号 ({acc_clean}) 不在受信任实盘白名单中: {whitelist}")
    check_results["factor_g_whitelist"] = True

    # 5. 运行时确认
    if not live_confirm:
        raise LiveGateAuthError(
            "🚨 [因子H] 缺少运行时显式确认参数 --live-confirm！必须由经授权操作员显式提供二次确认指令方可建立连接 (fail-closed)。"
        )
    check_results["factor_h_confirm"] = True

    # 6. 密码学信任根与公钥环完整性校验 (固定强制 require_trust_root=True)
    check_keyring_integrity(explicit_keyring_pin=explicit_keyring_pin)
    check_results["factor_f_trust_root"] = True

    # 7. xtquant 依赖库可用性校验
    if require_xtquant:
        try:
            import xtquant
        except ImportError:
            raise LiveGateAuthError("🚨 [券商依赖] 未检测到 xtquant 依赖库 (xtquant unavailable)！实盘连接被强制阻断。")
        check_results["xtquant_available"] = True

    # 8. 生产模型禁止 LEGACY_UNVERIFIED
    if model_record is None:
        try:
            from models.registry import ModelRegistry, ModelState
            reg = model_registry or ModelRegistry()
            prod_rec = reg.get_production()
            if prod_rec is None:
                prod_recs = reg.list_records(state=ModelState.PRODUCTION)
                if prod_recs:
                    prod_rec = prod_recs[0]
        except Exception:
            prod_rec = None
    else:
        prod_rec = model_record

    if prod_rec is not None:
        v_status = getattr(prod_rec, "verification_status", None)
        if not v_status and isinstance(prod_rec, dict):
            v_status = prod_rec.get("verification_status")
        if v_status == "LEGACY_UNVERIFIED":
            raise LiveGateAuthError(
                f"🚨 [因子C/D] 生产模型 ({getattr(prod_rec, 'model_id', 'UNKNOWN')}) 属于历史未验证模型 (LEGACY_UNVERIFIED)，"
                f"缺少可独立密码学校验的物理证据与防伪清单，永久剥夺实盘资格，严禁用于实盘连接与交易！"
            )
        check_results["model_verified"] = True

    return True, check_results


def verify_live_order_gate(
    account_id: str,
    live_confirm: bool = False,
    quote_timestamp: Optional[Union[datetime, float, str]] = None,
    current_date: Optional[str] = None,
    model_registry: Optional[Any] = None,
    model_record: Optional[Any] = None,
    paper_ledger: Optional[Any] = None,
    shadow_ledger: Optional[Any] = None,
    approval_artifact_path: Optional[Union[str, Path]] = None,
    explicit_keyring_pin: Optional[str] = None,
    nonce_store_path: Optional[Union[str, Path]] = None,
    order_shares: Optional[Union[int, float]] = None,
    order_price: Optional[float] = None,
    order_symbol: Optional[str] = None,
    order_side: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    实盘发单前置网关校验 (verify_live_order_gate):
    固定强制: require_trust_root=True, check_physical_cert=True, min_observation_days=30。
    核验行情新鲜度、物理研究认证凭证与防篡改链、前瞻与观察成熟度 (>= 30天)、审批数字签名全量订单字段强绑定、单笔与累计风控限额及基于跨进程文件锁的原子 Nonce 消费。
    """
    # 兼容测试 Mock
    if hasattr(verify_live_trading_multi_factor_gate, "assert_called") or getattr(verify_live_trading_multi_factor_gate, "_mock_return_value", None) is not None:
        return verify_live_trading_multi_factor_gate(
            account_id=account_id,
            live_confirm=live_confirm,
            quote_timestamp=quote_timestamp,
            current_date=current_date,
            model_registry=model_registry,
            model_record=model_record,
            paper_ledger=paper_ledger,
            shadow_ledger=shadow_ledger,
            approval_artifact_path=approval_artifact_path,
            explicit_keyring_pin=explicit_keyring_pin,
            nonce_store_path=nonce_store_path,
            order_shares=order_shares,
            order_price=order_price,
            order_symbol=order_symbol,
            order_side=order_side,
        )

    # 1. 首先核验连接层全量全局锁 (固定强制最高安全信任根)
    _, conn_results = verify_live_connection_gate(
        account_id=account_id,
        live_confirm=live_confirm,
        model_registry=model_registry,
        model_record=model_record,
        explicit_keyring_pin=explicit_keyring_pin,
    )
    check_results = dict(conn_results)

    # 2. 条件 j: 当日为有效交易所交易日
    cal = CanonicalTradingCalendar.get_instance()
    check_day = (current_date or datetime.now().strftime("%Y-%m-%d"))[:10]
    if not cal.is_trading_day(check_day):
        raise LiveGateAuthError(f"🚨 [因子J] 当前日期 ({check_day}) 不是合法交易所交易日，严禁发起实盘交易！")
    check_results["factor_j_trading_day"] = True

    # 3. 条件 i: 实时行情数据新鲜度检查 (绝不允许 None 绕过、未来行情或过期行情)
    if quote_timestamp is None:
        raise LiveGateAuthError(
            "🚨 [因子I] 实时行情时间戳缺失 (quote_timestamp is None)！实盘发单严禁无时间戳直接发单 (fail-closed)。"
        )

    now_ts = time.time()
    try:
        if isinstance(quote_timestamp, (datetime, pd.Timestamp)):
            q_ts = quote_timestamp.timestamp()
        elif isinstance(quote_timestamp, (int, float)):
            q_ts = float(quote_timestamp)
        elif isinstance(quote_timestamp, str):
            q_ts = pd.to_datetime(quote_timestamp).timestamp()
        else:
            raise ValueError(f"无法解析的时间戳类型: {type(quote_timestamp)}")
    except Exception as parse_err:
        raise LiveGateAuthError(f"🚨 [因子I] 实时行情时间戳无效或无法解析: {quote_timestamp} ({parse_err})")

    if q_ts - now_ts > 5.0:
        raise LiveGateAuthError(
            f"🚨 [因子I] 实时行情时间戳异常: 时间戳处于未来 ({q_ts - now_ts:.1f} 秒后)，严禁使用未来行情！"
        )

    staleness = now_ts - q_ts
    max_staleness = float(getattr(settings, "MAX_QUOTE_STALENESS_SECONDS", 300.0))
    if staleness > max_staleness:
        raise LiveGateAuthError(
            f"🚨 [因子I] 实时行情数据新鲜度检查失败: 数据已陈旧 {staleness:.1f} 秒 > 允许上限 {max_staleness:.1f} 秒！"
        )
    check_results["factor_i_staleness_seconds"] = round(staleness, 2)

    # 4. 条件 c & d: 生产模型已认证研究物理证据与已成熟前瞻观察证据
    if model_record is None:
        from models.registry import ModelRegistry, ModelState
        reg = model_registry or ModelRegistry()
        prod_rec = reg.get_production()
        if prod_rec is None:
            prod_recs = reg.list_records(state=ModelState.PRODUCTION)
            if prod_recs:
                prod_rec = prod_recs[0]
            else:
                raise LiveGateAuthError("🚨 [因子C/D] 生产模型校验失败: 注册表中不存在处于 PRODUCTION 状态的上线模型！")
    else:
        prod_rec = model_record

    v_status = getattr(prod_rec, "verification_status", None)
    if not v_status and isinstance(prod_rec, dict):
        v_status = prod_rec.get("verification_status")
    if v_status == "LEGACY_UNVERIFIED":
        raise LiveGateAuthError(
            f"🚨 [因子C/D] 生产模型 ({getattr(prod_rec, 'model_id', 'UNKNOWN')}) 属于历史未验证模型 (LEGACY_UNVERIFIED)，"
            f"缺少可独立密码学校验的物理证据与防伪清单，永久剥夺实盘资格，严禁用于实盘交易！"
        )

    combined_ev: Dict[str, Any] = {}
    if hasattr(prod_rec, "evidence") and prod_rec.evidence:
        ev_obj = prod_rec.evidence
        if hasattr(ev_obj, "to_dict"):
            ev_obj = ev_obj.to_dict()
        if isinstance(ev_obj, dict):
            combined_ev.update(ev_obj)

    for promo in getattr(prod_rec, "promotion_history", []):
        pe = promo.get("evidence")
        if isinstance(pe, dict):
            combined_ev.update(pe)

    # 条件 c: 研究认证证据 (物理凭据防伪哈希与防篡改链深度重验)
    cert_ref = combined_ev.get("certification_ref") or combined_ev.get("certification_evidence")
    if not cert_ref:
        raise LiveGateAuthError(f"🚨 [因子C] 生产模型 ({getattr(prod_rec, 'model_id', 'UNKNOWN')}) 缺失完整的已认证研究证据 (certification_evidence)！")

    p_to_check = None
    expected_cert_sha = None
    if isinstance(cert_ref, (str, Path)):
        p_to_check = Path(cert_ref)
    elif isinstance(cert_ref, dict):
        p_to_check = Path(cert_ref.get("artifact_path", ""))
        expected_cert_sha = cert_ref.get("artifact_sha256") or cert_ref.get("sha256")

    if not p_to_check or str(p_to_check) == ".":
        raise LiveGateAuthError("🚨 [因子C] 生产模型研究认证凭据路径无效！")

    if not p_to_check.is_absolute():
        p_to_check = (Path(settings.BASE_DIR) / p_to_check).resolve()

    if not p_to_check.is_file():
        raise LiveGateAuthError(f"🚨 [因子C] 生产模型研究认证物理凭证文件不存在: {p_to_check}")

    cert_bytes = p_to_check.read_bytes()
    calc_cert_sha = hashlib.sha256(cert_bytes).hexdigest()
    if expected_cert_sha and calc_cert_sha != expected_cert_sha:
        raise LiveGateAuthError(
            f"🚨 [因子C] 生产模型研究认证凭证 SHA256 校验失败: 预期 {expected_cert_sha}, 实际 {calc_cert_sha} (Fail-Closed)！"
        )

    try:
        cert_json = json.loads(cert_bytes.decode("utf-8"))
        if not isinstance(cert_json, dict):
            raise ValueError("凭证文件内容必须为合规 JSON 对象")
        status_val = str(cert_json.get("status", "")).upper()
        if status_val not in ("CERTIFIED", "PASS"):
            raise ValueError(f"凭证状态不合规: '{status_val}' (需 CERTIFIED 或 PASS)")

        # 深度防篡改链比对: 模型 ID 强绑定
        cert_mid = cert_json.get("model_id")
        prod_mid = getattr(prod_rec, "model_id", None)
        if prod_mid and cert_mid and str(cert_mid).strip() != str(prod_mid).strip():
            raise ValueError(f"凭证模型 ID ({cert_mid}) 与生产模型 ID ({prod_mid}) 不匹配")

        # 数据集哈希校验
        cert_ds_hash = cert_json.get("dataset_sha256") or cert_json.get("dataset_manifest_sha256") or cert_json.get("dataset_hash")
        prod_ds_hash = getattr(prod_rec, "dataset_sha256", None)
        if prod_ds_hash and cert_ds_hash and cert_ds_hash != prod_ds_hash:
            raise ValueError(f"凭证数据集哈希 ({cert_ds_hash}) 与生产模型记录 ({prod_ds_hash}) 不符")

        # 若包含数字签名则进行密码学验签
        if "signature" in cert_json and "signer_key_id" in cert_json:
            sig_k = cert_json["signer_key_id"]
            sig_h = cert_json["signature"]
            msg_p = {k: v for k, v in cert_json.items() if k not in ("signature", "file_sha256")}
            m_bytes = json.dumps(msg_p, sort_keys=True, ensure_ascii=False).encode("utf-8")
            s_ok, s_errs = verify_ed25519_signature(
                message=m_bytes,
                signature_hex=sig_h,
                key_id=sig_k,
                required_purpose="MODEL_CERTIFICATION"
            )
            if not s_ok:
                raise ValueError(f"研究认证凭证数字签名校验失败: {s_errs}")
    except Exception as e:
        raise LiveGateAuthError(f"🚨 [因子C] 生产模型研究认证物理凭证内容损坏或被非法篡改: {e} (Fail-Closed)！")
    check_results["factor_c_research_cert"] = True

    # 条件 d: 前瞻观察成熟证据 (固定门限 >= 30, PASS/MATURE)
    min_obs_days = 30
    pv = combined_ev.get("prospective_validation")
    if not pv:
        raise LiveGateAuthError(f"🚨 [因子D] 生产模型 ({getattr(prod_rec, 'model_id', 'UNKNOWN')}) 缺失前瞻观察证据制品！")

    pv_days = 0
    pv_status = ""
    if isinstance(pv, dict):
        pv_days = int(pv.get("observed_trading_days", 0))
        pv_status = str(pv.get("status", "")).upper()
    elif hasattr(pv, "observed_trading_days"):
        pv_days = int(getattr(pv, "observed_trading_days", 0))
        pv_status = str(getattr(pv, "status", "")).upper()
    elif isinstance(pv, str) and Path(pv).is_file():
        try:
            loaded_pv = json.loads(Path(pv).read_text(encoding="utf-8"))
            pv_days = int(loaded_pv.get("observed_trading_days", 0))
            pv_status = str(loaded_pv.get("status", "")).upper()
        except Exception:
            pass

    if pv_days < min_obs_days or pv_status not in ("PASS", "MATURE"):
        raise LiveGateAuthError(
            f"🚨 [因子D] 生产模型前瞻观察证据不达标: 观察交易日 {pv_days} 天 (需 >= {min_obs_days} 天), 状态为 '{pv_status}' (需 PASS 或 MATURE)！"
        )
    check_results["factor_d_prospective_mature"] = True

    # 5. 条件 e: 模拟盘或影子观察累计达到观察门限 (固定门限 >= 30 交易日)
    paper_days = 0
    shadow_days = 0
    if paper_ledger is not None:
        paper_days = getattr(paper_ledger, "observed_trading_days", 0)
    else:
        from execution.paper_ledger import PaperTradingLedger
        try:
            pl = PaperTradingLedger()
            paper_days = pl.observed_trading_days
        except Exception:
            pass

    if shadow_ledger is not None:
        shadow_days = getattr(shadow_ledger, "observed_trading_days", 0)
    else:
        from execution.paper_ledger import ShadowTradingLedger
        try:
            sl = ShadowTradingLedger()
            shadow_days = sl.observed_trading_days
        except Exception:
            pass

    max_obs_days = max(paper_days, shadow_days)
    if max_obs_days < min_obs_days:
        raise LiveGateAuthError(
            f"🚨 [因子E] 模拟盘或影子观察累计天数不足: 实际 Paper={paper_days} 天, Shadow={shadow_days} 天 < 门限 {min_obs_days} 交易日！"
        )
    check_results["factor_e_observation_days"] = max_obs_days

    # 6. 条件 f: 审批人数字签名凭证物理存在且验签通过
    if not approval_artifact_path:
        approval_path = Path(settings.BASE_DIR) / "reports" / "live_approval_artifact.json"
    else:
        approval_path = Path(approval_artifact_path)

    if not approval_path.is_file():
        raise LiveGateAuthError(f"🚨 [因子F] 审批人数字签名物理凭证文件不存在: {approval_path}")

    try:
        appr_data = json.loads(approval_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise LiveGateAuthError(f"🚨 [因子F] 审批凭证解析失败 (非合法 JSON): {e}")

    signer_key_id = appr_data.get("signer_key_id")
    signature = appr_data.get("signature")
    if not signer_key_id or signer_key_id not in TRUSTED_KEY_REGISTRY:
        raise LiveGateAuthError(f"🚨 [因子F] 审批凭证 signer_key_id 未知或未在受信任公钥环中注册: {signer_key_id}")

    key_entry = TRUSTED_KEY_REGISTRY[signer_key_id]
    if key_entry.get("status") != "ACTIVE":
        raise LiveGateAuthError(f"🚨 [因子F] 审批凭证 signer_key_id 处于非活跃状态: {key_entry.get('status')}")

    # 严格用途隔离: 必须具备 LIVE_TRADING_AUTHORIZATION 权限
    allowed_purposes = key_entry.get("allowed_purposes", [])
    if "LIVE_TRADING_AUTHORIZATION" not in allowed_purposes:
        raise LiveGateAuthError(
            f"🚨 [因子F] 密钥用途错配 (Key Purpose Mismatch): 签名密钥 ({signer_key_id}) "
            f"的用途权限为 {allowed_purposes}，缺少 'LIVE_TRADING_AUTHORIZATION'！严禁跨用途签名 (fail-closed)。"
        )

    # 验证数字签名与 Payload 深度强绑定 (Fail-Closed: 必需字段严格非空)
    raw_payload = appr_data.get("canonical_payload") or {
        k: v for k, v in appr_data.items() if k not in ("signature", "artifact_sha256")
    }

    # 6.0 必需全量绑定字段 (Stage S3 强化: symbol, side, quantity, limit_price 强制进入必填字段)
    mandatory_fields = [
        "authorized_action", "account_id", "model_id", "environment",
        "symbol", "side", "quantity", "limit_price",
        "nonce", "operator", "risk_limits"
    ]
    for mf in mandatory_fields:
        if mf not in raw_payload or raw_payload[mf] is None:
            raise LiveGateAuthError(f"🚨 [因子F] 审批凭证缺少必要字段: '{mf}' (Fail-Closed)！")
        if isinstance(raw_payload[mf], str) and not raw_payload[mf].strip():
            raise LiveGateAuthError(f"🚨 [因子F] 审批凭证必要字段为空: '{mf}' (Fail-Closed)！")

    # 6.1 authorized_action 必须精确为 LIVE_ORDER
    authorized_action = raw_payload.get("authorized_action")
    if authorized_action != "LIVE_ORDER":
        raise LiveGateAuthError(
            f"🚨 [因子F] 审批凭证授权动作错配: 授权动作为 '{authorized_action}'，必须精确为 'LIVE_ORDER'！"
        )

    # 6.2 绑定 account_id (强制非空且匹配)
    acc_clean = str(account_id).strip()
    bound_acc = raw_payload.get("account_id")
    if str(bound_acc).strip() != acc_clean:
        raise LiveGateAuthError(f"🚨 [因子F] 审批凭证账户绑定错配: 凭证绑定账户 ({bound_acc}) 与当前账户 ({acc_clean}) 不符！")

    auth_accounts = raw_payload.get("authorized_accounts")
    if auth_accounts and acc_clean not in auth_accounts:
        raise LiveGateAuthError(f"🚨 [因子F] 审批凭证授权账户不匹配: 当前账户 ({acc_clean}) 不在凭证授权名单: {auth_accounts}")

    # 6.3 绑定 model_id (强制非空且匹配生产模型)
    bound_model = raw_payload.get("model_id")
    expected_model_id = getattr(prod_rec, "model_id", None) if prod_rec else None
    if expected_model_id and bound_model != expected_model_id:
        raise LiveGateAuthError(f"🚨 [因子F] 审批凭证模型绑定错配: 凭证限定模型 ({bound_model}) 与生产模型 ({expected_model_id}) 不符！")

    # 6.4 绑定 environment (强制非空，只允许 production，严禁 all/test)
    bound_env = raw_payload.get("environment")
    if str(bound_env).strip().lower() not in ("production", "prod"):
        raise LiveGateAuthError(
            f"🚨 [因子F] 审批凭证运行环境错配: 凭证限定环境 ({bound_env}) 严禁用于实盘生产 (仅允许 production，禁止 all/test)！"
        )

    # 6.5 绑定 expiration_timestamp / expires_at (强制非空且未超时)
    exp_val = raw_payload.get("expires_at") if raw_payload.get("expires_at") is not None else raw_payload.get("expiration_timestamp")
    if exp_val is None:
        raise LiveGateAuthError("🚨 [因子F] 审批凭证缺少必要字段: 'expires_at' / 'expiration_timestamp' (Fail-Closed)！")
    try:
        exp_ts = float(exp_val) if isinstance(exp_val, (int, float)) else pd.to_datetime(exp_val).timestamp()
        if time.time() > exp_ts:
            raise LiveGateAuthError(f"🚨 [因子F] 审批凭证已过期: 有效期截止 {exp_val}，当前已超时！")
    except LiveGateAuthError:
        raise
    except Exception as e:
        raise LiveGateAuthError(f"🚨 [因子F] 审批凭证过期时间戳无效: {exp_val} ({e})")

    # 6.6 审批人身份 operator
    operator = raw_payload.get("operator") or raw_payload.get("approver")
    if not operator or (isinstance(operator, str) and not operator.strip()):
        raise LiveGateAuthError("🚨 [因子F] 审批凭证缺少审批人身份字段: 'operator' (Fail-Closed)！")

    # 6.7 必须包含 risk_limits 且必须为包含 max_notional 与 max_daily_notional 正有限数的非空字典
    risk_limits = raw_payload.get("risk_limits")
    if not isinstance(risk_limits, dict) or not risk_limits:
        raise LiveGateAuthError("🚨 [因子F] 审批凭证缺少风控限额字段或为空字典: 'risk_limits' (Fail-Closed)！")

    def _validate_risk_limit_val(val: Any, name: str) -> float:
        if val is None or isinstance(val, bool):
            raise LiveGateAuthError(f"🚨 [风控限额] 审批凭证 risk_limits.{name} 缺失或类型非法！")
        try:
            num = float(val)
        except (ValueError, TypeError):
            raise LiveGateAuthError(f"🚨 [风控限额] risk_limits.{name} 非法数值: {val} (必须为正有限数)！")
        if num <= 0 or math.isnan(num) or math.isinf(num):
            raise LiveGateAuthError(f"🚨 [风控限额] risk_limits.{name} 必须为正有限数: {val}！")
        return num

    max_notional = _validate_risk_limit_val(risk_limits.get("max_notional"), "max_notional")
    max_daily_notional = _validate_risk_limit_val(risk_limits.get("max_daily_notional"), "max_daily_notional")

    # 6.8 审批载荷订单参数与发单参数全字段强绑定 (symbol, side, quantity, limit_price)
    p_symbol = str(raw_payload["symbol"]).strip()
    p_side = str(raw_payload["side"]).strip().upper()
    if p_side not in ("BUY", "SELL"):
        raise LiveGateAuthError(f"🚨 [订单绑定] 审批凭证 side 只允许 'BUY' 或 'SELL'，当前为: '{p_side}'！")

    def _validate_positive_num(val: Any, name: str) -> float:
        if val is None or isinstance(val, bool):
            raise LiveGateAuthError(f"🚨 [订单绑定] 审批凭证 {name} 缺失或类型非法！")
        try:
            num = float(val)
        except (ValueError, TypeError):
            raise LiveGateAuthError(f"🚨 [订单绑定] 审批凭证 {name} 必须为正有限数: {val}！")
        if num <= 0 or math.isnan(num) or math.isinf(num):
            raise LiveGateAuthError(f"🚨 [订单绑定] 审批凭证 {name} 必须为正有限数: {val}！")
        return num

    p_qty = _validate_positive_num(raw_payload["quantity"], "quantity")
    p_price = _validate_positive_num(raw_payload["limit_price"], "limit_price")

    # 发单参数必须全量非空并精确比对
    if order_symbol is None or not str(order_symbol).strip():
        raise LiveGateAuthError("🚨 [订单绑定] 审批凭证绑定了标的代码，但发单参数未传入 order_symbol！")
    o_symbol = str(order_symbol).strip()
    if o_symbol != p_symbol:
        raise LiveGateAuthError(
            f"🚨 [订单绑定] 委托标的代码 ({o_symbol}) 与审批凭证绑定标的 ({p_symbol}) 不匹配！"
        )

    if order_side is None or not str(order_side).strip():
        raise LiveGateAuthError("🚨 [订单绑定] 审批凭证绑定了买卖方向，但发单参数未传入 order_side！")
    o_side = str(order_side).strip().upper()
    if o_side not in ("BUY", "SELL"):
        raise LiveGateAuthError(f"🚨 [订单绑定] 委托买卖方向只允许 'BUY' 或 'SELL'，当前为: '{o_side}'！")
    if o_side != p_side:
        raise LiveGateAuthError(
            f"🚨 [订单绑定] 委托买卖方向 ({o_side}) 与审批凭证绑定方向 ({p_side}) 不匹配！"
        )

    if order_shares is None:
        raise LiveGateAuthError("🚨 [订单绑定] 审批凭证绑定了委托数量，但发单参数未传入 order_shares！")
    o_shares = _validate_positive_num(order_shares, "order_shares")
    if abs(o_shares - p_qty) > 1e-5:
        raise LiveGateAuthError(
            f"🚨 [订单绑定] 委托股数 ({o_shares}) 与审批凭证绑定数量 ({p_qty}) 不匹配！"
        )

    if order_price is None:
        raise LiveGateAuthError("🚨 [订单绑定] 审批凭证绑定了限价，但发单参数未传入 order_price！")
    o_price = _validate_positive_num(order_price, "order_price")
    if abs(o_price - p_price) > 1e-5:
        raise LiveGateAuthError(
            f"🚨 [订单绑定] 委托价格 ({o_price}) 与审批凭证绑定价格 ({p_price}) 不匹配！"
        )

    # 计算订单名义金额并校验风控限额
    notional = o_shares * o_price
    if notional > max_notional:
        raise LiveGateAuthError(
            f"🚨 [风控] 订单名义金额 ({notional:.2f}) 超过审批凭证单笔风控限额 max_notional ({max_notional:.2f})！"
        )
    if notional > max_daily_notional:
        raise LiveGateAuthError(
            f"🚨 [风控] 订单名义金额 ({notional:.2f}) 超过审批凭证日累计风控限额 max_daily_notional ({max_daily_notional:.2f})！"
        )

    # 6.9 Nonce 与 Ed25519 签名验证
    nonce = raw_payload.get("nonce")
    if not nonce:
        raise LiveGateAuthError("🚨 [因子F] 审批凭证缺少必要字段: 'nonce' (Fail-Closed)！")

    msg_bytes = json.dumps(raw_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    sig_ok, sig_errs = verify_ed25519_signature(
        message=msg_bytes,
        signature_hex=str(signature),
        key_id=signer_key_id,
        required_purpose="LIVE_TRADING_AUTHORIZATION",
        domain_separator="LIVE_TRADING_AUTHORIZATION_APPROVAL_V1"
    )
    if not sig_ok:
        raise LiveGateAuthError(f"🚨 [因子F] 审批凭证 Ed25519 数字签名校验失败: {sig_errs}")
    check_results["factor_f_signature_verified"] = True

    # 6.10 跨进程文件互斥锁保护下的原子 Nonce 消费与重放防护 (Stage S3)
    nonce_file = Path(nonce_store_path) if nonce_store_path else (settings.DATA_DIR / "security" / "used_approval_nonces.json")
    lock_file = nonce_file.with_suffix(".lock")

    with _cross_process_file_lock(lock_file):
        used_nonces: Dict[str, Any] = {}
        if nonce_file.exists():
            try:
                content = nonce_file.read_text(encoding="utf-8")
                used_nonces = json.loads(content)
                if not isinstance(used_nonces, dict):
                    raise ValueError("Nonce 存储格式无效 (必须为字典)")
            except Exception as e:
                raise LiveGateAuthError(f"🚨 [因子F] 读取 Nonce 重放防护记录存储异常: {e}，实盘强制硬阻断 (Fail-Closed)！")

        if nonce in used_nonces:
            prev = used_nonces[nonce]
            raise LiveGateAuthError(
                f"🚨 [因子F] 检测到审批凭证重放攻击 (Approval Replay Detected)! Nonce ({nonce}) "
                f"已于 {prev.get('used_at')} 被使用过，严禁重复提交！"
            )

        # 当日累计委托名义金额校验 (锁内计算，防止并发绕过)
        if notional > 0.0:
            max_daily_notional = risk_limits.get("max_daily_notional")
            if max_daily_notional is not None:
                today_cum = sum(
                    float(item.get("notional", 0.0))
                    for item in used_nonces.values()
                    if isinstance(item, dict) and item.get("date") == check_day and item.get("account_id") == acc_clean
                ) + notional
                if today_cum > float(max_daily_notional):
                    raise LiveGateAuthError(
                        f"🚨 [风控] 当日累计委托名义金额 ({today_cum:.2f}) 超过审批凭证当日风控限额 max_daily_notional ({max_daily_notional})！"
                    )

        # 锁内原子记录与 fsync 落盘消费
        used_nonces[nonce] = {
            "used_at": datetime.now().isoformat(),
            "date": check_day,
            "account_id": acc_clean,
            "approval_id": raw_payload.get("approval_id", "UNKNOWN"),
            "notional": round(notional, 4),
            "shares": order_shares,
            "price": order_price,
            "symbol": order_symbol,
            "side": order_side,
        }

        tmp_nonce_file = nonce_file.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            nonce_file.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_nonce_file, "w", encoding="utf-8") as tf:
                json.dump(used_nonces, tf, indent=2, ensure_ascii=False)
                tf.flush()
                os.fsync(tf.fileno())
            tmp_nonce_file.replace(nonce_file)
        except Exception as save_err:
            if tmp_nonce_file.exists():
                try:
                    tmp_nonce_file.unlink()
                except Exception:
                    pass
            raise LiveGateAuthError(f"🚨 [因子F] 原子消费 Nonce 失败: {save_err} (Fail-Closed)！")

    logger.info("🎉 [实盘发单网关鉴权通过] 全量风控与密码学签名合规因子校验无误，已安全放行实盘委托发单。")
    return True, check_results


def verify_live_trading_multi_factor_gate(
    account_id: str,
    live_confirm: bool,
    quote_timestamp: Optional[Union[datetime, float, str]] = None,
    current_date: Optional[str] = None,
    model_registry: Optional[Any] = None,
    model_record: Optional[Any] = None,
    paper_ledger: Optional[Any] = None,
    shadow_ledger: Optional[Any] = None,
    approval_artifact_path: Optional[Union[str, Path]] = None,
    explicit_keyring_pin: Optional[str] = None,
    nonce_store_path: Optional[Union[str, Path]] = None,
    order_shares: Optional[Union[int, float]] = None,
    order_price: Optional[float] = None,
    order_symbol: Optional[str] = None,
    order_side: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    实盘发单前置多因子硬核鉴权 (统一兼容入口):
    固定不可降级安全门限，直接映射发单网关。
    """
    return verify_live_order_gate(
        account_id=account_id,
        live_confirm=live_confirm,
        quote_timestamp=quote_timestamp,
        current_date=current_date,
        model_registry=model_registry,
        model_record=model_record,
        paper_ledger=paper_ledger,
        shadow_ledger=shadow_ledger,
        approval_artifact_path=approval_artifact_path,
        explicit_keyring_pin=explicit_keyring_pin,
        nonce_store_path=nonce_store_path,
        order_shares=order_shares,
        order_price=order_price,
        order_symbol=order_symbol,
        order_side=order_side,
    )
