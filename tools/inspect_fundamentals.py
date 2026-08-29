"""
检查两个可用财务接口的数据结构，为 fundamentals provider 设计解析器。
"""
import sys, io, pandas as pd
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import akshare as ak

print("########## stock_yjbb_em (东财业绩报表, 全市场按报告期) ##########")
df = ak.stock_yjbb_em(date="20231231")
print("shape:", df.shape)
print("columns:", list(df.columns))
print(df.head(3).to_string())
print()

print("########## stock_financial_abstract (Sina 主要财务指标, 单只) ##########")
da = ak.stock_financial_abstract(symbol="600519")
print("shape:", da.shape)
print("columns[:15]:", list(da.columns)[:15])
print("前10行(指标名 + 最近几列):", da.iloc[:10, [0,1,2,3,4]].to_string())
