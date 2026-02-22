"""
LSTM模型配置文件
"""

# 数据路径配置
DATA_CONFIG = {
    'processed_dir': 'Data/Processed',
    'weather_dir': 'Data/Processed/weather',
    'runoff_dir': 'Data/Processed/runoffs',
    'stations_file': 'Data/Processed/stations.csv'
}

# 模型保存路径
MODEL_DIR = 'model'
OUTPUT_DIR = 'output'

# 站点列表
STATIONS = [
    'Heishui River',
    'Meigu River',
    'Niulan River',
    'Pudu River',
    'Wudongde Ruku River',
    'Xining River',
    'Zhongdu River'
]

# LSTM模型参数
LSTM_CONFIG = {
    'input_size': None,  # 将根据数据自动设置
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.1,
    'output_size': 1,
    'bidirectional': False
}

# 训练参数
TRAIN_CONFIG = {
    'sequence_length': 30,  # 使用过去30天的数据预测
    'batch_size': 64,
    'epochs': 200,
    'learning_rate': 0.001,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'early_stopping_patience': 15,
    'device': 'cuda',  # 'cuda' 或 'cpu'
}

# 跨站点特征选择配置
FEATURE_SELECTION_CONFIG = {
    'use_cross_station_features': True,
    'cross_station_top_k': 10,
    'cross_station_use_abs': True,
}

# 评估指标配置
METRICS = ['NSE', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2']

# 可视化配置
VIS_CONFIG = {
    'figure_size': (15, 6),
    'dpi': 100,
    'save_format': 'png'
}
