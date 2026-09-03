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

def run_remaining():
    test_dir = root_dir / 'tests'
    test_files = sorted([f for f in test_dir.glob('test_*.py')])

    # 从 11 开始跑 (索引 10)
    remaining_files = test_files[10:]
    print(f'[*] 继续测试剩余 {len(remaining_files)} 个模块 (11 ~ 39)...\n')

    passed = 0
    failed = 0
    failed_reports = []
    slow_tests = []

    for idx, tf in enumerate(remaining_files, 11):
        t0 = time.time()
        fname = tf.name
        sys.stdout.write(f'[{idx:02d}/{len(test_files):02d}] 测试 {fname:<38} ... ')
        sys.stdout.flush()

        cmd = [sys.executable, '-m', 'pytest', str(tf), '-q']
        try:
            res = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True, timeout=120)
            elapsed = time.time() - t0
            if res.returncode == 0:
                print(f'✅ PASSED ({elapsed:.2f}s)')
                passed += 1
                if elapsed > 10.0:
                    slow_tests.append((fname, elapsed))
            else:
                print(f'❌ FAILED ({elapsed:.2f}s)')
                failed += 1
                failed_reports.append((fname, res.stdout + '\n' + res.stderr))
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f'⚠️ TIMEOUT > 120s')
            failed += 1
            failed_reports.append((fname, f'TIMEOUT > 120s'))

    print('\n' + '=' * 80)
    print(f'📊 剩余模块扫描结果: 通过 {passed} / 失败 {failed}')
    print('=' * 80)

    if failed_reports:
        print('\n❌ 【失败模块报错详情】:')
        for fname, err in failed_reports:
            print(f'\n--- {fname} ---')
            print(err[:800])

if __name__ == '__main__':
    run_remaining()
