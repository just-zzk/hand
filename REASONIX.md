# Reasonix project memory

Notes the user pinned via the `#` prompt prefix. The whole file is
loaded into the immutable system prefix every session — keep it terse.

- 1. 安装依赖
pip install -r requirements.txt

# 2. 采集数据集
python scripts/collect_data.py --gesture 数字1 --count 400

# 3. 训练模型
python scripts/train.py --k 5 --feature all

# 4. 运行系统
python src/main.py
- 1. 安装依赖
pip install -r requirements.txt

# 2. 采集数据集
python scripts/collect_data.py --gesture 数字1 --count 400

# 3. 训练模型
python scripts/train.py --k 5 --feature all

# 4. 运行系统
python src/main.py
