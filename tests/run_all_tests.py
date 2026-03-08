"""
运行所有测试脚本
"""
import subprocess
import sys
import os

def run_test(test_file):
    """运行单个测试文件"""
    print(f"\n{'='*60}")
    print(f"运行测试: {test_file}")
    print('='*60)
    
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("错误输出:", result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    test_dir = os.path.dirname(__file__)
    
    tests = [
        os.path.join(test_dir, "test_ppo.py"),
        os.path.join(test_dir, "test_dqn.py"),
        os.path.join(test_dir, "test_a2c.py"),
        os.path.join(test_dir, "test_gpt2.py"),
    ]
    
    results = {}
    for test in tests:
        if os.path.exists(test):
            results[test] = run_test(test)
        else:
            print(f"警告: 测试文件不存在 {test}")
            results[test] = False
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for test, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{os.path.basename(test)}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("所有测试通过! ✓")
    else:
        print("部分测试失败，请检查错误信息")
    print("="*60)
    
    sys.exit(0 if all_passed else 1)
