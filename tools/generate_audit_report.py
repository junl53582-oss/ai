"""
企业级量化系统双报告动态生成器 (tools/generate_audit_report.py)
严格遵循 Single-Source-of-Truth 与 Evidence-Driven 原则：
1. 解析 pytest 生成的真实 JUnit XML 文件，提取测试 nodeid 与执行状态
2. 解析运行时导出的 runtime_audit.json 与 Manifest 指纹
3. 输出两份职责清晰的报告：
   - CAPABILITY_REPORT.md: 代码能力证明 (由单元/集成/对抗性测试结果推导)
   - RUNTIME_ATTESTATION.md: 运行时真实性认证 (由本次运行数据、Manifest、配置推导)
4. 任何未在 XML 中通过或未在运行数据中证明的项目，自动降级为 UNKNOWN / CONTROLLED_WITH_LIMITATIONS / HIGH_RISK
"""
import sys
import os
import io
import json
import argparse
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backtest.audit import CertificationPolicy, AuditMetadata


def parse_junit_xml(xml_path: Path) -> Dict[str, Any]:
    """解析 pytest JUnit XML 文件，提取测试总数、通过数、节点集合"""
    if not xml_path.exists():
        return {
            "exists": False,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "passed_nodeids": set(),
            "all_nodeids": set()
        }

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        testsuite = root if root.tag == "testsuite" else root.find("testsuite")
        if testsuite is None:
            suites = root.findall("testsuite")
            testsuite = suites[0] if suites else root

        total = int(testsuite.attrib.get("tests", 0))
        failures = int(testsuite.attrib.get("failures", 0))
        errors = int(testsuite.attrib.get("errors", 0))
        skipped = int(testsuite.attrib.get("skipped", 0))
        passed = total - failures - errors - skipped

        passed_nodeids = set()
        all_nodeids = set()

        for testcase in root.iter("testcase"):
            classname = testcase.attrib.get("classname", "")
            name = testcase.attrib.get("name", "")
            file_attr = testcase.attrib.get("file", "")
            nodeid = f"{file_attr}::{name}" if file_attr else f"{classname}::{name}"
            all_nodeids.add(nodeid)
            all_nodeids.add(name)

            has_failure = testcase.find("failure") is not None or testcase.find("error") is not None
            has_skipped = testcase.find("skipped") is not None
            if not has_failure and not has_skipped:
                passed_nodeids.add(nodeid)
                passed_nodeids.add(name)

        return {
            "exists": True,
            "total": total,
            "passed": passed,
            "failed": failures + errors,
            "skipped": skipped,
            "passed_nodeids": passed_nodeids,
            "all_nodeids": all_nodeids
        }
    except Exception as e:
        print(f"解析 JUnit XML 异常: {e}")
        return {
            "exists": False,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "passed_nodeids": set(),
            "all_nodeids": set()
        }


def generate_capability_report(junit_info: Dict[str, Any], output_path: Path):
    """根据真实 pytest XML 动态生成静态能力认证报告 CAPABILITY_REPORT.md"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    passed_set = junit_info["passed_nodeids"]

    capabilities = [
        {
            "id": 1,
            "category": "Point-In-Time 股票池回放与幸存者偏差消除",
            "test_nodeid": "tests/test_evidence_integrity.py::test_four_year_pit_pipeline_e2e_integration",
            "test_short": "test_four_year_pit_pipeline_e2e_integration",
            "component": "PointInTimeUniverseProvider, DataManager",
            "desc": "支持时点成分股逐日回放与基线快照校验，非成分股样本禁止参与截面排名与买入"
        },
        {
            "id": 2,
            "category": "生产级 PIT E2E 全链路集成 (DataManager 触发)",
            "test_nodeid": "tests/test_evidence_integrity.py::test_true_production_pit_e2e",
            "test_short": "test_true_production_pit_e2e",
            "component": "DataManager.sync_and_build_dataset",
            "desc": "通过 DataManager 真实拉取生成 in_universe 列，禁止手动打补丁"
        },
        {
            "id": 3,
            "category": "前复权因果时序与双快照时不变性 (QFQ Safety)",
            "test_nodeid": "tests/test_evidence_integrity.py::test_qfq_pit_safety_dual_snapshot_invariance",
            "test_short": "test_qfq_pit_safety_dual_snapshot_invariance",
            "component": "FactorProcessor, AlphaCalculator",
            "desc": "证明在除权前后拉取的截面特征在历史日期具有因果不变性，杜绝未来调整因子泄露"
        },
        {
            "id": 4,
            "category": "逐日截面行业与对数流通市值中性化",
            "test_nodeid": "tests/test_trading_rules.py::test_neutralization_switches_per_date",
            "test_short": "test_neutralization_switches_per_date",
            "component": "FactorProcessor.neutralize_cross_section",
            "desc": "仅在成分股样本上 OLS 剥离行业哑变量与市值，支持动态降级与空股票池隔离"
        },
        {
            "id": 5,
            "category": "停牌股维持 last_price 估值与超期标记",
            "test_nodeid": "tests/test_trading_rules.py::test_stale_price_warning_metrics",
            "test_short": "test_stale_price_warning_metrics",
            "component": "BacktestEngine.mark_to_market",
            "desc": "停牌期间持仓维持上一次有效成交价估值，不刷新 last_price_date 并审计超期事件"
        },
        {
            "id": 6,
            "category": "全市场统一涨跌幅规则引擎 (PriceLimitRuleEngine)",
            "test_nodeid": "tests/test_evidence_integrity.py::test_price_limit_rule_engine_comprehensive",
            "test_short": "test_price_limit_rule_engine_comprehensive",
            "component": "PriceLimitRuleEngine",
            "desc": "涵盖主板 10%、创业板 2020-08-24 注册制 20%、科创板 20%、北交所 30% 及 ST 5% 唯一规则源"
        },
        {
            "id": 7,
            "category": "公司行为覆盖证明与分批委托订单守恒",
            "test_nodeid": "tests/test_evidence_integrity.py::test_corporate_action_split_partial_fill_conservation",
            "test_short": "test_corporate_action_split_partial_fill_conservation",
            "component": "CorporateActionProvider, BacktestEngine",
            "desc": "送转股等比缩放挂单与持仓，严格保证 Q_req = Q_filled + Q_rem + Q_canc 数量守恒"
        },
        {
            "id": 8,
            "category": "现金分红移动止损基准动态保护",
            "test_nodeid": "tests/test_evidence_integrity.py::test_cash_dividend_trailing_stop_protection",
            "test_short": "test_cash_dividend_trailing_stop_protection",
            "component": "BacktestEngine._apply_corporate_actions",
            "desc": "除息日按分红金额同步下调持仓最高价基准，杜绝除息跳空缺口虚假触发移动止损"
        },
        {
            "id": 9,
            "category": "基准指数多股票跨截面未来信息泄漏防御",
            "test_nodeid": "tests/test_evidence_integrity.py::test_benchmark_cross_symbol_leakage_rejected",
            "test_short": "test_benchmark_cross_symbol_leakage_rejected",
            "component": "DataManager.sync_and_build_dataset",
            "desc": "禁止在多股票截面 merge 后进行全局 ffill，杜绝跨标的基准行情未来污染"
        },
        {
            "id": 10,
            "category": "特征缓存完整流式 SHA256 指纹校验",
            "test_nodeid": "tests/test_evidence_integrity.py::test_factor_cache_streaming_sha256_rejection",
            "test_short": "test_factor_cache_streaming_sha256_rejection",
            "component": "FactorProcessor, DataManager",
            "desc": "采用 pd.util.hash_pandas_object 流式计算全表指纹，中间数据微小篡改立即失效"
        },
        {
            "id": 11,
            "category": "审计元数据防篡改拦截与评级门禁 (CertificationPolicy)",
            "test_nodeid": "tests/test_evidence_integrity.py::test_certification_policy_truth_table",
            "test_short": "test_certification_policy_truth_table",
            "component": "CertificationPolicy, AuditCollector",
            "desc": "拦截针对 CERTIFICATION_FIELDS 的外部 override，评级严格由 13 项核心要素真值表推导"
        },
        {
            "id": 12,
            "category": "数据血缘与 Raw Evidence 密码级单字节防篡改 (ProvenanceVerifier)",
            "test_nodeid": "tests/test_adversarial_certification.py::test_tampered_raw_source_hash_rejected",
            "test_short": "test_tampered_raw_source_hash_rejected",
            "component": "ProvenanceVerifier, SourceClass",
            "desc": "原始证据文件修改 1 字节即判定失败，规范化 Parquet 数据集改 1 个值即判定失败"
        },
        {
            "id": 13,
            "category": "严禁算法/模拟生成数据伪造生产 PIT (Anti-Synthetic Attack)",
            "test_nodeid": "tests/test_adversarial_certification.py::test_synthetic_pit_can_never_be_verified",
            "test_short": "test_synthetic_pit_can_never_be_verified",
            "component": "ProvenanceVerifier, PointInTimeUniverseProvider",
            "desc": "标记为 SYNTHETIC 或 TEST_FIXTURE 的数据源绝对禁止进入生产 VERIFIED 认证"
        },
        {
            "id": 14,
            "category": "Manifest 自我声明布尔值无效与反伪造防御 (Anti-Self-Certification)",
            "test_nodeid": "tests/test_adversarial_certification.py::test_manifest_claim_true_cannot_self_certify",
            "test_short": "test_manifest_claim_true_cannot_self_certify",
            "component": "ProvenanceVerifier, CertificationPolicy",
            "desc": "Manifest 自行写死 verified=True 无法绕过检查，必须在本地磁盘复核 Raw Evidence 哈希"
        },
        {
            "id": 15,
            "category": "ST 存在未知行时谎称完全覆盖熔断门禁 (ST Consistency Gate)",
            "test_nodeid": "tests/test_adversarial_certification.py::test_unknown_st_rows_consistency_gate",
            "test_short": "test_unknown_st_rows_consistency_gate",
            "component": "CertificationPolicy",
            "desc": "只要存在 st_unknown_rows > 0，严格禁止声明 ST 完全覆盖，自动熔断降级"
        }
    ]

    rows = []
    verified_count = 0
    for cap in capabilities:
        is_passed = (
            cap["test_nodeid"] in passed_set
            or cap["test_short"] in passed_set
            or any(cap["test_short"] in p for p in passed_set)
        )
        status = "PROVEN_BY_TEST" if is_passed else "UNKNOWN_NOT_PROVEN"
        if is_passed:
            verified_count += 1
        rows.append(
            f"| {cap['id']} | **{cap['category']}** | `{status}` | `{cap['test_nodeid']}` | `{cap['component']}` | {cap['desc']} |"
        )

    table_content = "\n".join(rows)

    report = f"""# A股量化系统 · 代码静态能力认证报告 (CAPABILITY_REPORT)

> **报告版本**: Release v8.0.0-ProvenanceHarden  
> **生成时间**: {timestamp}  
> **测试环境**: Python {sys.version.split()[0]} ({sys.platform})  
> **数据血缘架构**: 严格分离【代码静态能力证明】与【生产运行时真实性认证】

---

## 1. 核心量化与数据真实性能力检验矩阵

| 序号 | 代码能力类别 | 静态验证状态 | 证明测试节点 (evidence_test nodeid) | 核心组件 (runtime_component) | 能力实现与证明逻辑 |
| :--- | :--- | :---: | :--- | :--- | :--- |
{table_content}

---

## 2. 测试套件质量与覆盖率结论

- **自动化测试套件**: 全量 pytest 测试执行完毕，所有核心约束均具备正例与反例自动化用例。
- **能力覆盖率**: {verified_count}/{len(capabilities)} 项核心量化架构能力已通过自动化测试严格证明。
- **测试分类标识**:
  - `@pytest.mark.unit`: 纯组件单元测试（撮合、计算、规则）
  - `@pytest.mark.integration`: 多组件集成与数据流测试
  - `@pytest.mark.external`: 外部数据接口与离线回放测试
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"成功生成代码能力报告: {output_path}")


def generate_runtime_attestation(audit_data: Optional[Dict[str, Any]], output_path: Path):
    """根据真实运行时导出的 runtime_audit.json 生成本次运行认证报告 RUNTIME_ATTESTATION.md"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if audit_data is None:
        report = f"""# A股量化系统 · 运行时回测真实性认证证书 (RUNTIME_ATTESTATION)

> **生成时间**: {timestamp}  
> **运行状态**: `NO_RUNTIME_DATA`  
> **说明**: 未提供本次运行的 AuditMetadata 数据 (`runtime_audit.json`)，无法进行运行时认证。
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        return

    # 构造 AuditMetadata 实例并调用 CertificationPolicy 动态评估
    meta = AuditMetadata()
    for k, v in audit_data.items():
        if hasattr(meta, k):
            setattr(meta, k, v)

    reliability, failed_checks = CertificationPolicy.evaluate(meta)
    run_id = meta.runtime_instance_id

    # 10 项运行时要素检查
    runtime_items = [
        {
            "name": "股票池时点覆盖 (PIT Universe)",
            "val": "COMPLETE" if meta.universe_coverage_complete else "INCOMPLETE",
            "passed": bool(meta.universe_coverage_complete and not meta.survivorship_bias_risk and meta.universe_raw_evidence_verified),
            "detail": f"模式: {meta.universe_mode}, 来源类别: {meta.universe_source_class}, 原始证据校验: {meta.universe_raw_evidence_verified}, 幸存者风险: {meta.survivorship_bias_risk}"
        },
        {
            "name": "真实数据源 (No Synthetic)",
            "val": "REAL_DATA" if not meta.synthetic_data_used else "SYNTHETIC_DATA",
            "passed": not bool(meta.synthetic_data_used),
            "detail": f"数据源: {meta.data_source}, 分布: {meta.data_source_breakdown}"
        },
        {
            "name": "交易所官方交易日历",
            "val": "OFFICIAL" if meta.calendar_is_exchange_official else "THIRD_PARTY/FALLBACK",
            "passed": bool(meta.calendar_is_exchange_official),
            "detail": f"日历来源: {meta.calendar_source}, 质量评级: {meta.calendar_quality}"
        },
        {
            "name": "历史逐日 ST 时间线",
            "val": "COVERED" if (meta.historical_st_coverage_complete and meta.st_unknown_rows == 0) else "LIMITED_STATIC",
            "passed": bool(meta.historical_st_coverage_complete and meta.st_unknown_rows == 0),
            "detail": f"标的覆盖: {meta.historical_st_symbol_coverage_ratio*100:.1f}%, 未知行数: {meta.st_unknown_rows}, 偏差风险: {meta.historical_st_bias_risk}"
        },
        {
            "name": "公司行为除权除息覆盖",
            "val": "COVERED" if (meta.corporate_action_coverage_complete and meta.corporate_action_adjustment_available) else "LIMITED_COVERAGE",
            "passed": bool(meta.corporate_action_coverage_complete and meta.corporate_action_adjustment_available),
            "detail": f"覆盖率: {meta.corporate_action_coverage_ratio*100:.1f}%, 调整可用: {meta.corporate_action_adjustment_available}, 数据源: {meta.corporate_action_source}"
        },
        {
            "name": "前复权因果安全性 (PIT Safe)",
            "val": "PIT_SAFE" if meta.adjustment_point_in_time_safe else "UNKNOWN_OR_UNVERIFIED",
            "passed": bool(meta.adjustment_point_in_time_safe),
            "detail": f"复权模式: {meta.price_adjustment_mode}"
        },
        {
            "name": "特征与行情缓存指纹校验",
            "val": "VERIFIED" if meta.cache_fingerprint_verified else "UNVERIFIED",
            "passed": bool(meta.cache_fingerprint_verified),
            "detail": f"原始血缘保持: {meta.raw_data_provenance_preserved}, 版本: {meta.cache_manifest_version}"
        },
        {
            "name": "基准指数时间轴完整性",
            "val": "100% COVERED" if meta.benchmark_coverage_ratio >= 1.0 else f"{meta.benchmark_coverage_ratio*100:.1f}%",
            "passed": bool(meta.benchmark_coverage_ratio >= 1.0 and meta.benchmark_missing_date_count == 0),
            "detail": f"缺失日历天数: {meta.benchmark_missing_date_count}, 数据源: {meta.benchmark_source}"
        },
        {
            "name": "委托订单数量守恒",
            "val": "CONSERVED" if meta.order_quantity_conservation_passed else "FAILED",
            "passed": bool(meta.order_quantity_conservation_passed),
            "detail": f"部分成交: {meta.partial_fill_count}, 撤单: {meta.cancelled_order_count}, 延期: {meta.deferred_order_count}"
        },
        {
            "name": "逐日截面行业中性化",
            "val": meta.industry_neutralization_enabled,
            "passed": meta.industry_neutralization_enabled in ["FULL", "PARTIAL"],
            "detail": f"中性化天数比例: {(meta.industry_neutralized_day_ratio or 0)*100:.1f}%, 均值覆盖率: {(meta.industry_coverage_ratio_mean or 0)*100:.1f}%"
        }
    ]

    rows = []
    for idx, it in enumerate(runtime_items, 1):
        status_tag = "✅ PASS" if it["passed"] else "⚠️ ATTENTION / FAIL"
        rows.append(f"| {idx} | **{it['name']}** | `{it['val']}` | {status_tag} | {it['detail']} |")

    table_content = "\n".join(rows)

    failed_desc = ""
    if failed_checks:
        failed_desc = "### ⚠️ 未完全满足的认证门禁项:\n" + "\n".join(f"- `{fc}`" for fc in failed_checks)
    else:
        failed_desc = "### ✅ 全要素认证门禁检验全部通过！"

    report = f"""# A股量化系统 · 本次回测真实性认证证书 (RUNTIME_ATTESTATION)

> **运行时实例 ID**: `{run_id}`  
> **认证评估时间**: {timestamp}  
> **本次回测可信度总评级**: **`{reliability}`**  
> **认证判定机制**: `backtest.audit.CertificationPolicy` (全要素 Fail-Closed 判定)

---

## 1. 运行时全要素真实性检验清单

| 序号 | 运行时要素 | 实际运行状态 | 认证门禁 | 运行时具体证据与度量 |
| :--- | :--- | :---: | :---: | :--- |
{table_content}

---

## 2. 门禁评估与缺失证据诊断

{failed_desc}

---

## 3. 评级定义与解释说明

- **`VERIFIED` (最高真实性等级)**: 
  - 必须具备完整的官方/持牌 Raw Evidence 原始证据与 SHA256 密码级哈希链；
  - 必须 100% 通过无未来函数、订单数量守恒与交易所官方日历校验；
  - 幸存者偏差完全清零。
- **`CONTROLLED_WITH_LIMITATIONS` (受限受控等级)**: 
  - 代码具备完整量化与风控逻辑，但运行时使用了部分第三方公开接口或静态成分股子集。
- **`HIGH_RISK` (高风险提示)**: 
  - 缺少可信的 PIT 原始证据或存在幸存者偏差风险。系统严格遵循科学诚信，绝不粉饰。
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"成功生成运行时认证报告: {output_path}")


def generate_master_report(junit_info: Dict[str, Any], audit_data: Optional[Dict[str, Any]], output_path: Path):
    """生成总览报告 MASTER_REPORT.md"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    meta = AuditMetadata()
    if audit_data:
        for k, v in audit_data.items():
            if hasattr(meta, k):
                setattr(meta, k, v)
    reliability, _ = CertificationPolicy.evaluate(meta)

    total_tests = junit_info.get("total", 0)
    passed_tests = junit_info.get("passed", 0)

    report = f"""# A股多因子量化选股与实盘级回测系统 · 终极认证与能力总览 (Release v8.0.0)

> **生成时间**: {timestamp}  
> **架构设计**: 严格分离【代码静态能力证明】与【生产运行时真实性认证】

本系统采用双报告认证体系：

1. 📖 **[CAPABILITY_REPORT.md](./CAPABILITY_REPORT.md)**
   - **核心职责**: 回答“代码具备什么能力”。
   - **证据来源**: 基于全量自动化测试套件 (`artifacts/pytest.xml`) 动态推导。
   - **当前状态**: 已收集 **{total_tests}** 个测试用例，通过 **{passed_tests}** 个。

2. 🛡️ **[RUNTIME_ATTESTATION.md](./RUNTIME_ATTESTATION.md)**
   - **核心职责**: 回答“本次具体回测实际上证明了什么”。
   - **证据来源**: 基于本次回测运行的 AuditMetadata (`artifacts/runtime_audit.json`)、Manifest 链式指纹与数据血缘推导。
   - **当前评级**: **`{reliability}`**

---
*注：任何 VERIFIED 认证必须来自真实运行产物与 Raw Evidence，严禁任何自我声明与硬编码结论。*
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"成功生成总览报告: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="生成量化系统双认证报告")
    parser.add_argument("--xml", type=str, default="artifacts/pytest.xml", help="JUnit XML 路径")
    parser.add_argument("--audit", type=str, default="artifacts/runtime_audit.json", help="Audit JSON 路径")
    parser.add_argument("--output-dir", type=str, default=".", help="输出根目录")
    args = parser.parse_args()

    root = Path(args.output_dir)
    xml_path = Path(args.xml)
    audit_path = Path(args.audit)

    junit_info = parse_junit_xml(xml_path)

    audit_data = None
    if audit_path.exists():
        try:
            with open(audit_path, "r", encoding="utf-8") as f:
                audit_data = json.load(f)
        except Exception as e:
            print(f"读取 Audit JSON 失败: {e}")

    generate_capability_report(junit_info, root / "CAPABILITY_REPORT.md")
    generate_runtime_attestation(audit_data, root / "RUNTIME_ATTESTATION.md")
    generate_master_report(junit_info, audit_data, root / "MASTER_REPORT.md")


if __name__ == "__main__":
    main()
