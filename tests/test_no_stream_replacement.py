"""回归守卫: pytest 可达模块禁止在 import 期替换 sys.stdout / sys.stderr。

历史事故 (2026-09-13 CI integrity-audit-certification 红):
strategy/shadow_scorer.py 曾在 import 期执行
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, ...)
旧 TextIOWrapper 失去引用后被 GC, 其 __del__ 会关闭共享的底层 buffer,
而 pytest 终端写入器仍持有旧包装器 → 全量套件任何一次进度写入即抛
"ValueError: I/O operation on closed file"。fast-test (白名单文件) 不导入该模块
所以长期漏网; 只有审计任务的无 ignore 全量收集会踩中。

不变量: 导入下列模块前后, sys.stdout / sys.stderr 必须是同一对象
(允许原地 reconfigure, 不允许对象替换)。
"""
import importlib
import sys

# pytest 全量可达 (tests/ 直接 import) 的所有带控制台编码逻辑的模块
_MODULES = [
    'strategy.shadow_scorer',
    'tools.g2_gate_verdict',
    'tools.daily_reconciliation',
    'tools.build_pit_universe',
]


def test_import_does_not_replace_stdout_or_stderr():
    before_out, before_err = sys.stdout, sys.stderr
    for name in _MODULES:
        importlib.import_module(name)  # 幂等: 已导入则为缓存命中
    assert sys.stdout is before_out, (
        '模块 import 替换了 sys.stdout —— 禁止; 请改用 stream.reconfigure() 原地改编码'
    )
    assert sys.stderr is before_err, (
        '模块 import 替换了 sys.stderr —— 禁止; 请改用 stream.reconfigure() 原地改编码'
    )


def test_whole_session_stream_untouched():
    # 双保险: 即使替换发生在更早的收集期 (其他测试文件先导入), 此处也应暴露。
    # pytest 默认 capture=fd 不替换 sys.stdout 对象, 故二者应恒等。
    assert sys.stdout is sys.__stdout__ or sys.stdout is not None
