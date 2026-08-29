"""
诊断: 在当前网络下哪些 AKShare 财务/基本面接口可直连。
测试多个数据源，找出可替代被封东财源的接口。
"""
import sys, io, time, traceback
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import akshare as ak

SYM = "600519"


def try_call(name, fn, timeout_sleep=1.0):
    try:
        t0 = time.monotonic()
        df = fn()
        dt = time.monotonic() - t0
        if df is None:
            print(f"[FAIL] {name}: 返回 None")
        elif hasattr(df, "empty") and df.empty:
            print(f"[EMPTY] {name}: 返回空 DataFrame ({dt:.1f}s)")
        else:
            cols = list(df.columns)[:12]
            print(f"[OK] {name}: shape={getattr(df,'shape',None)} rows={len(df)} ({dt:.1f}s) cols={cols}")
    except Exception as e:
        print(f"[ERR] {name}: {type(e).__name__}: {str(e)[:160]}")
    time.sleep(timeout_sleep)


print("=== 财务/基本面接口可达性诊断 (Sina vs Eastmoney) ===")
try_call("stock_financial_abstract(Sina)", lambda: ak.stock_financial_abstract(symbol=SYM))
try_call("stock_financial_analysis_indicator(东财)", lambda: ak.stock_financial_analysis_indicator(symbol=SYM, start_year="2018"))
try_call("stock_financial_audit(东财?)", lambda: ak.stock_financial_audit(symbol=SYM))
try_call("stock_yjbb_em(东财业绩报表)", lambda: ak.stock_yjbb_em(date="20231231"))
try_call("stock_zh_a_spot_em(东财实时)", lambda: ak.stock_zh_a_spot_em())
print("=== 诊断结束 ===")
