"""
数据加载和预处理模块
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os


class RunoffDataset(Dataset):
    """径流预测数据集"""
    
    def __init__(self, sequences, targets):
        """
        初始化数据集
        
        Args:
            sequences: 输入序列 (n_samples, sequence_length, n_features)
            targets: 目标值 (n_samples, 1)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class DataProcessor:
    """数据处理器"""
    
    def __init__(
        self,
        station_name,
        weather_dir,
        sequence_length=30,
        stations=None,
        use_cross_station_features=False,
        cross_station_top_k=10,
        cross_station_use_abs=True
    ):
        """
        初始化数据处理器
        
        Args:
            station_name: 站点名称
            weather_dir: weather数据目录
            sequence_length: 序列长度
            stations: 全部站点列表
            use_cross_station_features: 是否启用跨站点特征
            cross_station_top_k: 从其它站点选择的特征数
            cross_station_use_abs: 是否按绝对相关系数排序
        """
        self.station_name = station_name
        self.weather_dir = weather_dir
        self.sequence_length = sequence_length
        self.stations = stations or []
        self.use_cross_station_features = use_cross_station_features
        self.cross_station_top_k = cross_station_top_k
        self.cross_station_use_abs = cross_station_use_abs
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.data = None
        self.feature_names = None
        self.selected_cross_features = []

    def _select_cross_station_features(self, base_df):
        """基于Spearman相关系数，从其它站点选择Top-K特征"""
        if not self.stations or self.cross_station_top_k <= 0:
            return base_df

        if 'runoff' not in base_df.columns:
            raise ValueError(f"{self.station_name} 数据中缺少目标列 runoff")

        target_series = base_df['runoff']
        candidate_records = []

        for other_station in self.stations:
            if other_station == self.station_name:
                continue

            other_file = os.path.join(self.weather_dir, f"{other_station}.csv")
            if not os.path.exists(other_file):
                print(f"警告: 找不到其它站点数据文件，跳过: {other_file}")
                continue

            other_df = pd.read_csv(other_file)
            if other_df.isnull().any().any():
                other_df = other_df.fillna(method='ffill').fillna(method='bfill')

            min_len = min(len(base_df), len(other_df))
            aligned_target = target_series.iloc[:min_len]
            aligned_other = other_df.iloc[:min_len]

            for col in aligned_other.columns:
                series = aligned_other[col]
                corr = aligned_target.corr(series, method='spearman')

                if pd.isna(corr):
                    continue

                candidate_records.append({
                    'target_station': self.station_name,
                    'source_station': other_station,
                    'source_feature': col,
                    'spearman_corr': float(corr),
                    'abs_spearman_corr': float(abs(corr)),
                    'selected': False,
                    'prefixed_feature_name': f"{other_station}__{col}"
                })

        if not candidate_records:
            print(f"警告: {self.station_name} 未找到可用的跨站点候选特征")
            self.selected_cross_features = []
            return base_df

        candidates_df = pd.DataFrame(candidate_records)

        sort_col = 'abs_spearman_corr' if self.cross_station_use_abs else 'spearman_corr'
        selected_df = candidates_df.sort_values(by=sort_col, ascending=False).head(self.cross_station_top_k)

        selected_set = set(selected_df['prefixed_feature_name'].tolist())
        candidates_df['selected'] = candidates_df['prefixed_feature_name'].isin(selected_set)
        candidates_df = candidates_df.sort_values(by=['selected', sort_col], ascending=[False, False])

        merged_df = base_df.copy()
        for _, row in selected_df.iterrows():
            other_station = row['source_station']
            source_feature = row['source_feature']
            prefixed_name = row['prefixed_feature_name']
            other_file = os.path.join(self.weather_dir, f"{other_station}.csv")
            other_df = pd.read_csv(other_file)
            if other_df.isnull().any().any():
                other_df = other_df.fillna(method='ffill').fillna(method='bfill')

            min_len = min(len(merged_df), len(other_df))
            merged_df = merged_df.iloc[:min_len].copy()
            merged_df[prefixed_name] = other_df[source_feature].iloc[:min_len].values

        self.selected_cross_features = candidates_df.to_dict('records')

        print(
            f"{self.station_name} 已选择 {int(candidates_df['selected'].sum())} 个跨站点特征 "
            f"(候选总数: {len(candidates_df)})"
        )

        return merged_df
        
    def load_data(self):
        """加载站点数据"""
        weather_file = os.path.join(self.weather_dir, f"{self.station_name}.csv")
        
        if not os.path.exists(weather_file):
            raise FileNotFoundError(f"找不到文件: {weather_file}")
        
        # 读取数据
        self.data = pd.read_csv(weather_file)
        
        # 检查是否有缺失值
        if self.data.isnull().any().any():
            print(f"警告: {self.station_name} 数据中存在缺失值，将进行前向填充")
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')

        if self.use_cross_station_features:
            self.data = self._select_cross_station_features(self.data)
        
        # 保存特征名称
        self.feature_names = self.data.columns.tolist()
        
        print(f"成功加载 {self.station_name} 数据: {len(self.data)} 条记录, {len(self.feature_names)} 个特征")
        print(f"特征: {self.feature_names}")
        
        return self.data

    def save_selected_features_to_csv(self, save_path):
        """保存跨站点变量筛选信息到CSV"""
        if not self.selected_cross_features:
            print(f"{self.station_name} 无跨站点变量筛选信息可保存")
            return None

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        selection_df = pd.DataFrame(self.selected_cross_features)
        selection_df.to_csv(save_path, index=False)
        print(f"跨站点变量选择信息已保存至: {save_path}")
        return save_path
    
    def create_sequences(self, data, train_ratio=0.7, val_ratio=0.15):
        """
        创建时间序列数据
        
        Args:
            data: 原始数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        # 分离特征和目标（目标是runoff）
        features = data.values
        targets = data['runoff'].values.reshape(-1, 1)
        
        # 标准化
        features_scaled = self.feature_scaler.fit_transform(features)
        targets_scaled = self.target_scaler.fit_transform(targets)
        
        # 创建序列
        sequences = []
        seq_targets = []
        
        for i in range(len(features_scaled) - self.sequence_length):
            # 输入序列
            seq = features_scaled[i:i + self.sequence_length]
            # 目标值（下一个时间步的runoff）
            target = targets_scaled[i + self.sequence_length]
            
            sequences.append(seq)
            seq_targets.append(target)
        
        sequences = np.array(sequences)
        seq_targets = np.array(seq_targets)
        
        # 划分数据集
        n_samples = len(sequences)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        # 训练集
        train_sequences = sequences[:train_size]
        train_targets = seq_targets[:train_size]
        
        # 验证集
        val_sequences = sequences[train_size:train_size + val_size]
        val_targets = seq_targets[train_size:train_size + val_size]
        
        # 测试集
        test_sequences = sequences[train_size + val_size:]
        test_targets = seq_targets[train_size + val_size:]
        
        # 创建数据集
        train_dataset = RunoffDataset(train_sequences, train_targets)
        val_dataset = RunoffDataset(val_sequences, val_targets)
        test_dataset = RunoffDataset(test_sequences, test_targets)
        
        print(f"数据集划分 - 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def inverse_transform_target(self, scaled_target):
        """
        反标准化目标值
        
        Args:
            scaled_target: 标准化后的目标值
            
        Returns:
            original_target: 原始尺度的目标值
        """
        if isinstance(scaled_target, torch.Tensor):
            scaled_target = scaled_target.cpu().numpy()
        
        return self.target_scaler.inverse_transform(scaled_target.reshape(-1, 1)).flatten()
    
    def get_input_size(self):
        """获取输入特征维度"""
        if self.data is not None:
            return len(self.feature_names)
        else:
            raise ValueError("请先加载数据")


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    创建数据加载器
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        test_dataset: 测试数据集
        batch_size: 批次大小
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader
