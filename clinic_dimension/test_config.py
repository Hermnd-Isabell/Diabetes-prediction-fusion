import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 重新导入配置
if 'code.config' in sys.modules:
    del sys.modules['code.config']

from code.config import get_config

cfg = get_config()
print('Data path:', cfg.data_path)
print('File exists:', cfg.data_path.exists())

# 列出原始数据目录的文件
data_dir = project_root / '原始数据'
print('Files in 原始数据:')
for f in data_dir.iterdir():
    print(f'  {f.name}')

