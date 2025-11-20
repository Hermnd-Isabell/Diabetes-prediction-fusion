import subprocess
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.config import get_config


def run_script(script_name):
	print(f'[run] 执行 {script_name}')
	result = subprocess.run([sys.executable, str(Path(__file__).parent / f'{script_name}.py')], check=True)
	return result.returncode


def main():
	get_config()  # 确保输出目录存在
	run_script('data_cleaning')
	run_script('feature_processing')
	run_script('model_train')
	run_script('evaluate_visualize')
	print('[run] 全流程完成，结果见 outputs/')


if __name__ == '__main__':
	main()
