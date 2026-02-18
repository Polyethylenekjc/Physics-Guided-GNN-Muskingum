import csv
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_csv(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])
    return header, np.asarray(rows, dtype=np.float32)


def _load_station_series(root_dir: str, station: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    runoff_path = os.path.join(root_dir, "runoffs", f"{station}.csv")
    weather_path = os.path.join(root_dir, "weather", f"{station}.csv")

    runoff_header, runoff_data = _read_csv(runoff_path)
    weather_header, weather_data = _read_csv(weather_path)

    if runoff_data.ndim == 1:
        runoff_data = runoff_data[:, None]

    runoff = runoff_data[:, 0]

    if "runoff" in weather_header:
        idx = weather_header.index("runoff")
        weather_features = np.delete(weather_data, idx, axis=1)
        weather_names = [c for i, c in enumerate(weather_header) if i != idx]
    else:
        weather_features = weather_data
        weather_names = weather_header

    if len(runoff) != len(weather_features):
        raise ValueError(f"Length mismatch for station {station}.")

    return runoff, weather_features, weather_names


class MultiStationDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        station_order: List[str],
        history_len: int = 30,
        horizon: int = 1,
        selection_map: Optional[List[List[Tuple[int, int]]]] = None,
        selection_names: Optional[List[str]] = None,
    ) -> None:
        self.station_order = station_order
        self.history_len = history_len
        self.horizon = horizon

        runoffs = []
        weathers = []
        weather_names = None

        for station in station_order:
            runoff, weather, names = _load_station_series(root_dir, station)
            runoffs.append(runoff)
            weathers.append(weather)
            if weather_names is None:
                weather_names = names

        runoff_arr = np.stack(runoffs, axis=0)
        weather_arr = np.stack(weathers, axis=0)

        runoff_feat = runoff_arr[:, :, None]
        features = np.concatenate([runoff_feat, weather_arr], axis=2)

        self.features = features
        self.runoffs = runoff_arr
        self.weather = weather_arr
        self.feature_names = ["runoff"] + (weather_names or [])
        self.selection_map = selection_map
        self.selection_mean = None
        self.selection_std = None
        self.feature_mean = self.features.mean(axis=(0, 1))
        self.feature_std = self.features.std(axis=(0, 1))
        self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        self.runoff_mean = self.runoffs.mean(axis=1)
        self.runoff_std = self.runoffs.std(axis=1)
        self.runoff_std = np.where(self.runoff_std < 1e-6, 1.0, self.runoff_std)
        if selection_map is not None:
            if selection_names is None:
                selection_names = [f"selected_{i}" for i in range(len(selection_map[0]))]
            self.feature_names = selection_names
            self._set_selection_stats(selection_map)

        self.total_len = self.runoffs.shape[1]
        self.sample_len = self.total_len - history_len - horizon + 1
        if self.sample_len <= 0:
            raise ValueError("Not enough data for the given history and horizon.")

    def __len__(self) -> int:
        return self.sample_len

    def set_selection_map(
        self, selection_map: List[List[Tuple[int, int]]], selection_names: List[str]
    ) -> None:
        self.selection_map = selection_map
        self.feature_names = selection_names
        
        expected_len = len(selection_names)
        padded_map = []
        for i, mapping in enumerate(selection_map):
            m = list(mapping)
            while len(m) < expected_len:
                m.append((i, 0))
            padded_map.append(m)
        self.selection_map = padded_map
        self._set_selection_stats(padded_map)

    def _set_selection_stats(self, selection_map: List[List[Tuple[int, int]]]) -> None:
        means = []
        stds = []
        for mapping in selection_map:
            means.append([self.feature_mean[feat_idx] for _, feat_idx in mapping])
            stds.append([self.feature_std[feat_idx] for _, feat_idx in mapping])
        self.selection_mean = np.asarray(means, dtype=np.float32)
        self.selection_std = np.asarray(stds, dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = idx + self.history_len
        if self.selection_map is None:
            x = self.features[:, t - self.history_len : t, :]
            x = (x - self.feature_mean[None, None, :]) / self.feature_std[None, None, :]
        else:
            selected = []
            for node_idx, mapping in enumerate(self.selection_map):
                node_feats = []
                for src_idx, feat_idx in mapping:
                    node_feats.append(
                        self.features[src_idx, t - self.history_len : t, feat_idx]
                    )
                selected.append(np.stack(node_feats, axis=1))
            x = np.stack(selected, axis=0)
            if self.selection_mean is None or self.selection_std is None:
                self._set_selection_stats(self.selection_map)
            x = (x - self.selection_mean[:, None, :]) / self.selection_std[:, None, :]
        y = self.runoffs[:, t : t + self.horizon]
        prev_y = self.runoffs[:, t - 1]

        y = (y - self.runoff_mean[:, None]) / self.runoff_std[:, None]
        prev_y = (prev_y - self.runoff_mean) / self.runoff_std

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(prev_y).float(),
        )
