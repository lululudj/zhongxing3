"""用Python cmake模块编译llama.cpp"""
import os
import subprocess
import sys

# cmake路径
cmake_exe = os.path.join(sys.prefix, "Scripts", "cmake.exe")
if not os.path.exists(cmake_exe):
    cmake_exe = "cmake"  # 尝试系统PATH

print(f"cmake路径: {cmake_exe}")

# 项目目录
project_dir = r"E:\zhongxing2\llama-cpp-v4flash"
build_dir = r"E:\zhongxing2\llama-cpp-v4flash\build"

# 创建build目录
os.makedirs(build_dir, exist_ok=True)

# cmake配置
print("=== cmake配置 ===")
config_cmd = [
    cmake_exe,
    "-B", build_dir,
    "-S", project_dir,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DCMAKE_BUILD_TYPE=Release",
]
print(f"命令: {' '.join(config_cmd)}")
result = subprocess.run(config_cmd, capture_output=True, text=True)
print(f"stdout: {result.stdout[:2000]}")
print(f"stderr: {result.stderr[:500]}")
print(f"exit_code: {result.returncode}")

if result.returncode != 0:
    print("cmake配置失败!")
    sys.exit(1)

# cmake编译
print("=== cmake编译 ===")
build_cmd = [
    cmake_exe,
    "--build", build_dir,
    "--config", "Release",
    "--parallel", "8",
]
print(f"命令: {' '.join(build_cmd)}")
result = subprocess.run(build_cmd, capture_output=True, text=True)
print(f"stdout: {result.stdout[:2000]}")
print(f"stderr: {result.stderr[:500]}")
print(f"exit_code: {result.returncode}")

if result.returncode == 0:
    print("=== 编译成功! ===")
    # 查找exe文件
    for root, dirs, files in os.walk(build_dir):
        for f in files:
            if f.endswith(".exe"):
                print(f"找到: {os.path.join(root, f)}")
else:
    print("编译失败!")
    sys.exit(1)