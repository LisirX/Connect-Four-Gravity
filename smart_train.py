# smart_train.py (改进版)
import os
import re
import shutil
import subprocess
import time
import traceback
from datetime import datetime

# 从集中配置文件中导入所有常量
from config import (
    PYTHON_EXE, MODEL_DIR, ARCHIVE_DIR, MODEL_BASENAME, VERSION_FILE,
    ARENA_WIN_RATE_THRESHOLD, MODEL_SAVE_PATH
)

def get_current_version():
    """从版本文件中读取当前模型版本。"""
    if not os.path.exists(VERSION_FILE):
        print(f"未找到版本文件。正在初始化 '{VERSION_FILE}'，版本号为1。")
        with open(VERSION_FILE, 'w') as f:
            f.write('1')
        return 1
    with open(VERSION_FILE, 'r') as f:
        try:
            return int(f.read().strip())
        except ValueError:
            print(f"警告：版本文件 '{VERSION_FILE}' 包含无效内容。重置为版本1。")
            with open(VERSION_FILE, 'w') as f:
                f.write('1')
            return 1

def set_current_version(version):
    """将新模型版本写入版本文件。"""
    with open(VERSION_FILE, 'w') as f:
        f.write(str(version))

def find_or_create_initial_champion(version):
    """
    为当前版本找到冠军模型。
    如果不存在，则尝试重命名默认模型。
    """
    champion_path = os.path.join(MODEL_DIR, f"{MODEL_BASENAME}_v{version}.pth")
    if os.path.exists(champion_path):
        return champion_path

    print(f"\n!!! 警告：未找到冠军模型: {champion_path}")
    
    # 回退：检查默认模型路径是否存在
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"找到 '{MODEL_SAVE_PATH}'。将其提升为冠军模型 v{version}。")
        shutil.move(MODEL_SAVE_PATH, champion_path)
        return champion_path
    
    # 尝试在存档中查找任何模型
    archive_models = [f for f in os.listdir(ARCHIVE_DIR) if f.endswith('.pth')]
    if archive_models:
        latest_model = max(archive_models, key=lambda x: os.path.getmtime(os.path.join(ARCHIVE_DIR, x)))
        print(f"从存档中找到模型 '{latest_model}'。将其作为冠军模型 v{version}。")
        shutil.copy(os.path.join(ARCHIVE_DIR, latest_model), champion_path)
        return champion_path
    
    print("\n!!! 错误：无法找到任何初始模型来启动训练循环。")
    print(f"!!! 请手动运行 'train.py' 一次以创建初始模型。")
    exit(1)

def clean_up_files(file_list):
    """清理临时文件"""
    for file_path in file_list:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"已清理临时文件: {file_path}")
            except Exception as e:
                print(f"清理文件 {file_path} 时出错: {e}")

def run_training_loop():
    """协调AI训练和评估的主函数。"""
    # 确保必要的目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    current_version = get_current_version()
    temp_files = []  # 用于跟踪需要清理的临时文件

    while True:
        champion_backup = None
        try:
            print("\n" + "="*75)
            print(f"== 启动AI训练与评估循环: 版本 {current_version} -> {current_version + 1}")
            print("="*75 + "\n")

            # --- [步骤0] 定义当前循环的路径 ---
            champion_model = find_or_create_initial_champion(current_version)
            champion_backup = os.path.join(MODEL_DIR, f"champion_backup_v{current_version}.pth")
            temp_files.append(champion_backup)

            print(f"当前冠军: v{current_version} ('{os.path.basename(champion_model)}')")

            # --- [步骤1] 备份当前冠军模型 ---
            print("\n--- [步骤1/4] 备份当前冠军模型... ---")
            shutil.copy(champion_model, champion_backup)
            print(f"冠军模型已备份到 '{os.path.basename(champion_backup)}'")

            # --- [步骤2] 运行自我对弈生成训练数据 ---
            print("\n--- [步骤2/4] 运行 self_play.py 生成新数据... ---")
            subprocess.run([PYTHON_EXE, "self_play.py"], check=True)

            # --- [步骤3] 训练新的挑战者模型 ---
            print("\n--- [步骤3/4] 运行 train.py 创建新挑战者... ---")
            subprocess.run([PYTHON_EXE, "train.py"], check=True)
            
            if not os.path.exists(MODEL_SAVE_PATH):
                raise FileNotFoundError(f"训练后未找到挑战者模型: {MODEL_SAVE_PATH}")
                
            print(f"挑战者模型已创建于 '{MODEL_SAVE_PATH}'")

            # --- [步骤4] 运行竞技场评估挑战者 ---
            print("\n--- [步骤4/4] 运行 arena.py 评估挑战者 vs. 冠军... ---")
            # 我们将备份作为 'old_model' 传递给竞技场
            process = subprocess.run(
                [PYTHON_EXE, "arena.py", "--old_model", champion_backup],
                capture_output=True,  # 捕获输出以便更好的错误处理
                text=True
            )
            
            # 打印竞技场输出
            print("\n竞技场输出:")
            print(process.stdout)
            if process.stderr:
                print("竞技场错误:")
                print(process.stderr)
                
            win_rate = process.returncode  # arena.py 返回 win_rate * 100 作为退出码

            # --- [步骤5] 分析结果并决定是否提升 ---
            print("\n" + "="*75)
            print("== 评估结果分析")
            print("="*75 + "\n")
            print(f"    挑战者净胜率 (胜场 / (胜场+负场)): {win_rate}%")
            print(f"    提升阈值:                         {ARENA_WIN_RATE_THRESHOLD}%")
            print("")

            if win_rate >= ARENA_WIN_RATE_THRESHOLD:
                new_version = current_version + 1
                print(f"*** 结论: 挑战成功! 新模型现为冠军 v{new_version}! ***\n")

                # 归档旧冠军（备份）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_name = f"model_v{current_version}_lost_to_v{new_version}_{timestamp}.pth"
                archive_path = os.path.join(ARCHIVE_DIR, archive_name)
                shutil.move(champion_backup, archive_path)
                print(f"旧冠军 (v{current_version}) 已归档至: {archive_name}")

                # 提升挑战者为新冠军
                new_champion_model = os.path.join(MODEL_DIR, f"{MODEL_BASENAME}_v{new_version}.pth")
                shutil.move(MODEL_SAVE_PATH, new_champion_model)
                print(f"新冠军命名为: '{os.path.basename(new_champion_model)}'")

                # 更新版本用于下一个循环
                set_current_version(new_version)
                current_version = new_version
            else:
                print("*** 结论: 挑战失败。原冠军保持不变。 ***\n")
                if os.path.exists(MODEL_SAVE_PATH):
                    os.remove(MODEL_SAVE_PATH)
                    print("已删除失败的挑战者模型。")
                print("清理冠军备份...")
                if os.path.exists(champion_backup):
                    os.remove(champion_backup)
                    print("备份已清理。")
                print(f"冠军 v{current_version} 保持不败。")

            print("\n" + "="*75)
            print(f"== 循环完成。10秒后开始下一个循环 v{current_version}... ==")
            print("="*75 + "\n")
            time.sleep(10)

        except subprocess.CalledProcessError as e:
            print(f"\n!!! 子进程发生关键错误: {e}")
            print("!!! 输出:")
            print(e.stdout)
            print(e.stderr)
            print("!!! 训练循环已暂停，10秒后重试...")
            clean_up_files(temp_files)
            time.sleep(10)
            
        except FileNotFoundError as e:
            print(f"\n!!! 文件未找到错误: {e}")
            print("!!! 训练循环已暂停，10秒后重试...")
            clean_up_files(temp_files)
            time.sleep(10)
            
        except Exception as e:
            print(f"\n!!! 发生未预期错误: {e}")
            traceback.print_exc()
            print("!!! 训练循环已暂停，30秒后重试...")
            clean_up_files(temp_files)
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n\n用户中断训练循环。正在优雅退出。")
            # 清理备份文件
            clean_up_files(temp_files)
            break

if __name__ == "__main__":
    run_training_loop()