import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def _read_station_df(weather_dir: str, station: str) -> pd.DataFrame:
    file_path = os.path.join(weather_dir, f"{station}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到站点数据: {file_path}")
    df = pd.read_csv(file_path)
    if df.isnull().any().any():
        df = df.fillna(method="ffill").fillna(method="bfill")
    return df


def _read_selected_features(selection_file: str) -> List[Tuple[str, str, str]]:
    if not os.path.exists(selection_file):
        return []
    selection_df = pd.read_csv(selection_file)
    if "selected" not in selection_df.columns:
        return []
    selected_df = selection_df[selection_df["selected"] == True]
    rows = []
    for _, row in selected_df.iterrows():
        rows.append((row["source_station"], row["source_feature"], row["prefixed_feature_name"]))
    return rows


class MultiStationDataset(Dataset):
    def __init__(
        self,
        weather_dir: str,
        station_order: List[str],
        history_len: int = 30,
        horizon: int = 1,
        selection_dir: str = "output",
        train_ratio: float = 0.7,
    ) -> None:
        self.station_order = station_order
        self.history_len = history_len
        self.horizon = horizon

        base_dfs = {station: _read_station_df(weather_dir, station) for station in station_order}
        min_len = min(len(df) for df in base_dfs.values())
        base_dfs = {k: v.iloc[:min_len].reset_index(drop=True) for k, v in base_dfs.items()}

        station_feature_names: Dict[str, List[str]] = {}
        station_feature_frames: Dict[str, pd.DataFrame] = {}

        for station in station_order:
            station_df = base_dfs[station].copy()
            selection_file = os.path.join(selection_dir, station, f"{station}_selected_cross_station_features.csv")
            selected_rows = _read_selected_features(selection_file)

            for source_station, source_feature, prefixed_feature_name in selected_rows:
                if source_station not in base_dfs:
                    continue
                source_df = base_dfs[source_station]
                if source_feature not in source_df.columns:
                    continue
                station_df[prefixed_feature_name] = source_df[source_feature].values

            station_feature_names[station] = station_df.columns.tolist()
            station_feature_frames[station] = station_df

        global_feature_names = []
        for station in station_order:
            for name in station_feature_names[station]:
                if name not in global_feature_names:
                    global_feature_names.append(name)

        self.global_feature_names = global_feature_names
        self.station_feature_indices = {
            station: [global_feature_names.index(name) for name in station_feature_names[station]]
            for station in station_order
        }
        self.input_size_map = {station: len(station_feature_names[station]) for station in station_order}

        feature_tensors = []
        runoff_tensors = []
        for station in station_order:
            station_df = station_feature_frames[station]
            full_df = pd.DataFrame(0.0, index=station_df.index, columns=global_feature_names)
            full_df[station_df.columns] = station_df.values
            feature_tensors.append(full_df.values.astype(np.float32))
            runoff_tensors.append(station_df["runoff"].values.astype(np.float32))

        features = np.stack(feature_tensors, axis=0)
        runoffs = np.stack(runoff_tensors, axis=0)

        train_end = int(min_len * train_ratio)
        self.feature_scalers = []
        self.runoff_scalers = []
        for node_idx in range(len(station_order)):
            f_scaler = StandardScaler()
            r_scaler = StandardScaler()
            f_scaler.fit(features[node_idx, :train_end, :])
            r_scaler.fit(runoffs[node_idx, :train_end].reshape(-1, 1))

            features[node_idx] = f_scaler.transform(features[node_idx])
            runoffs[node_idx] = r_scaler.transform(runoffs[node_idx].reshape(-1, 1)).reshape(-1)
            self.feature_scalers.append(f_scaler)
            self.runoff_scalers.append(r_scaler)

        self.features = features
        self.runoffs = runoffs
        self.total_len = min_len
        self.sample_len = self.total_len - history_len - horizon + 1
        if self.sample_len <= 0:
            raise ValueError("数据长度不足以构建历史窗口")

    def __len__(self) -> int:
        return self.sample_len

    def __getitem__(self, idx: int):
        t = idx + self.history_len
        x = self.features[:, t - self.history_len:t, :]
        y = self.runoffs[:, t + self.horizon - 1]
        prev_y = self.runoffs[:, t - 1]
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(prev_y).float(),
        )
