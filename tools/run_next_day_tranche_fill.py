import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import settings
from execution.paper_broker import PaperBroker, OrderSide, OrderType, ExecutionStatus

print("=" * 80)
print(">>> [模拟跨日演练] 跨入 2026-09-04 9:30 开盘：解锁 T+1 限制并平滑补齐第二批建仓")
print("=" * 80)

# 1. 实例化实盘持久化 broker
broker = PaperBroker(persist=True)
broker.connect()
acc = broker.get_account()
print(f"[*] 当前账户初始总资产: {acc.total_equity:,.2f} 元 | 可用现金: {acc.cash:,.2f} 元")

# 2. 解锁昨日买入的 T+1 限制
broker.unlock_t1_shares()
print("[+] 已成功解锁昨日持仓，所有已持股标的均转为可随时卖出状态 (T+1 解锁成功)")

# 3. 读取待补齐建仓队列
queue_file = settings.BASE_DIR / "artifacts" / "pending_rebalance_queue.json"
if not queue_file.exists():
    print("[i] 待补齐建仓队列为空，无需补建。")
    sys.exit(0)

with open(queue_file, "r", encoding="utf-8") as f:
    queue_data = json.load(f)

print(f"[*] 检测到待平滑建仓标的数: {len(queue_data)} 支")

# 4. 逐一执行补齐挂单
filled_records = []
for sym, info in queue_data.items():
    shares = info.get("remaining_shares_to_fill", 0)
    ref_price = info.get("reference_price", 0.0)
    if shares <= 0 or ref_price <= 0:
        continue
    
    # 按照 100 股整手向下截断
    shares = (shares // 100) * 100
    if shares <= 0:
        continue

    order = broker.send_order(symbol=sym, side=OrderSide.BUY, shares=shares, price=ref_price, order_type=OrderType.LIMIT)
    if order.status == ExecutionStatus.FILLED:
        amt = order.filled_shares * order.avg_filled_price
        filled_records.append((sym, shares, ref_price, amt))
        print(f"  [买入成功] {sym} | 补齐股数: {shares:5d} 股 | 限价: {ref_price:6.2f} 元 | 成交金额: {amt:10.2f} 元")
    else:
        print(f"  [挂单未成] {sym} | 原因: {order.error_msg}")

# 5. 清理待补齐队列
if queue_file.exists():
    queue_file.unlink()
    print("\n[+] 待补齐建仓队列已全部平滑买满，队列已重置清空！")

# 6. 打印最新终极账户状态
acc_after = broker.get_account()
eq = acc_after.total_equity
cash = acc_after.cash
pos_cnt = len(broker.positions)
print("\n" + "=" * 80)
print("【2026-09-04 第二批建仓补齐后账户终极状态】:")
print("=" * 80)
print(f"  * 账户总资产:   {eq:,.2f} 元")
print(f"  * 账户可用现金: {cash:,.2f} 元 (保留现金垫: {cash/eq*100:.1f}%)")
print(f"  * 持仓标的总数: {pos_cnt} 支")
print("-" * 80)
print("  当前各标的持仓股数与市值分布:")
for sym, pos in sorted(broker.positions.items()):
    p_val = pos.total_shares * pos.avg_cost_price
    print(f"   - {sym:<10} | 总持股: {pos.total_shares:5d} 股 (可卖: {pos.available_shares:5d} 股) | 成本均价: {pos.avg_cost_price:6.2f} 元 | 持仓市值: {p_val:10.2f} 元")
print("=" * 80)
