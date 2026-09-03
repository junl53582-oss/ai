"""
全系统地毯式诊断与红队压力测试工具 (tools/run_full_system_diagnostic.py)
用于逐一审查 39 个测试套件、极端边界条件与系统脆弱点，生成客观真实的诊断报告。
"""
import sys
import time
import subprocess
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def run_diagnostic():
    print('=' * 80)
    print('>> [DIAGNOSTIC] 启动全系统 39 个测试模块逐项穿透测试与脆弱点排查')
    print('=' * 80)

    test_dir = root_dir / 'tests'
    test_files = sorted([f for f in test_dir.glob('test_*.py')])

    passed_count = 0
    failed_count = 0
    timeout_count = 0
    slow_tests = []
    failed_reports = []

    print(f'[*] 共检索到 {len(test_files)} 个独立测试模块文件\n')

    for idx, tf in enumerate(test_files, 1):
        t0 = time.time()
        fname = tf.name
        sys.stdout.write(f'[{idx:02d}/{len(test_files):02d}] 测试 {fname:<38} ... ')
        sys.stdout.flush()

        cmd = [sys.executable, '-m', 'pytest', str(tf), '-q']
        try:
            res = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True, timeout=45)
            elapsed = time.time() - t0
            if res.returncode == 0:
                print(f'✅ PASSED ({elapsed:.2f}s)')
                passed_count += 1
                if elapsed > 10.0:
                    slow_tests.append((fname, elapsed))
            else:
                print(f'❌ FAILED ({elapsed:.2f}s)')
                failed_count += 1
                failed_reports.append((fname, res.stdout + '\n' + res.stderr))
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f'⚠️ TIMEOUT > 45s')
            timeout_count += 1
            slow_tests.append((fname, elapsed))

    print('\n' + '=' * 80)
    print('📊 【测试模块逐一扫描汇总】')
    print('=' * 80)
    print(f'   * 测试通过模块数 : {passed_count} / {len(test_files)}')
    print(f'   * 测试失败模块数 : {failed_count}')
    print(f'   * 超时耗时模块数 : {timeout_count}')

    if slow_tests:
        print('\n⚠️ 【耗时较长或存在计算瓶颈的模块】:')
        for fname, el in slow_tests:
            print(f'   * {fname:<40} : {el:.2f}s')

    if failed_reports:
        print('\n❌ 【失败模块报错详情剖析】:')
        for fname, err in failed_reports:
            print(f'\n--- {fname} ---')
            print(err[:500])

    print('=' * 80)

if __name__ == '__main__':
    run_diagnostic()
