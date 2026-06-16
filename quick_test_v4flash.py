"""快速验证V4-Flash CLI子进程调用"""
import sys, os
sys.path.insert(0, r"E:\zhongxing2")
from zhongxing_agent import call_v4flash_cli, V4FLASH_CLI, V4FLASH_MODEL

print(f"CLI: {V4FLASH_CLI}")
print(f"Model: {V4FLASH_MODEL}")
print(f"CLI exists: {os.path.exists(V4FLASH_CLI)}")
print(f"Model exists: {os.path.exists(V4FLASH_MODEL)}")
print()
print("发送测试...")
result = call_v4flash_cli("1+1等于几？", max_tokens=10, timeout=1200)
print(f"结果: {repr(result)}")
