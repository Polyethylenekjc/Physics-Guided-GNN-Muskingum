import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

RAW_DIR = Path("Data/Raw")
OUTPUT_DIR = Path("Data/Processed")
RUNOFF_DIR = OUTPUT_DIR / "runoffs"
WEATHER_DIR = OUTPUT_DIR / "weather"
STATIONS_CSV = OUTPUT_DIR / "stations.csv"

RIVER_NAME_MAP = {
    "中都河": "Zhongdu River",
    "乌东德入库": "Wudongde Ruku River",
    "普渡河": "Pudu River",
    "牛栏河": "Niulan River",
    "美姑河": "Meigu River",
    "西宁河": "Xining River",
    "黑水河": "Heishui River",
}

STATION_NAME_MAP = {
    "中都镇": "Zhongdu Zhen",
    "东村镇": "Dongcun Zhen",
    "玉龙镇": "Yulong Zhen",
    "九口乡": "Jiukou Xiang",
    "西宁镇": "Xining Zhen",
    "宁南县": "Ningnan Xian",
    "乌东德": "Wudongde",
    "中都河（中都镇）": "Zhongdu Zhen",
    "普渡河（东村镇）": "Dongcun Zhen",
    "牛栏河（玉龙镇）": "Yulong Zhen",
    "美姑河（九口乡）": "Jiukou Xiang",
    "西宁河（西宁镇）": "Xining Zhen",
    "黑水河（宁南县）": "Ningnan Xian",
}

PHRASE_NAME_MAP = {
    "逐日径流量": "DailyRunoff",
    "逐日径流": "DailyRunoff",
    "逐日气象数据": "DailyWeather",
    "逐日气象": "DailyWeather",
    "水电站": "",
}

DATE_HINTS = ["日期", "时间", "date", "Date", "年", "月", "日"]
RUNOFF_HINTS = ["径流", "流量", "runoff", "discharge", "q", "dailyrunoff"]
WEATHER_HINTS = ["气象", "weather", "meteo", "dailyweather"]
META_HINTS = [
    "经度",
    "纬度",
    "海拔",
    "站名",
    "站点",
    "经纬度",
    "longitude",
    "latitude",
    "elevation",
    "station",
    "备注",
]

RUNOFF_FIRST_WITHOUT_DATE = True

WEATHER_COLUMN_MAP = {
    "地面气压(hPa)": "surface_pressure_hpa",
    "平均气温(℃)": "mean_temp_c",
    "最高气温2m(℃)": "max_temp_2m_c",
    "最低气温2m(℃)": "min_temp_2m_c",
    "降水量(mm)": "precipitation_mm",
    "露点温度(℃)": "dew_point_c",
    "平均风速(m/s)": "mean_wind_speed_ms",
    "经向风速(V,m/s)": "wind_v_ms",
    "纬向风速(U,m/s)": "wind_u_ms",
    "太阳辐射净强度(net,J/m2)": "solar_net_radiation_j_m2",
    "太阳辐射总强度(down,J/m2)": "solar_down_radiation_j_m2",
}


def combined_name_map() -> Dict[str, str]:
    combined = {}
    combined.update(RIVER_NAME_MAP)
    combined.update(STATION_NAME_MAP)
    combined.update(PHRASE_NAME_MAP)
    return combined


def replace_names(text: str, mapping: Dict[str, str]) -> str:
    for cn, en in mapping.items():
        text = text.replace(cn, en)
    return text


def rename_river_dirs(raw_dir: Path) -> None:
    for child in raw_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name not in RIVER_NAME_MAP:
            continue
        new_name = RIVER_NAME_MAP[child.name]
        new_path = child.with_name(new_name)
        if new_path.exists() or new_path == child:
            continue
        child.rename(new_path)


def rename_files_in_dir(dir_path: Path, mapping: Dict[str, str]) -> None:
    for file_path in dir_path.iterdir():
        if not file_path.is_file():
            continue
        new_name = replace_names(file_path.name, mapping)
        if new_name == file_path.name:
            continue
        new_path = file_path.with_name(new_name)
        if new_path.exists():
            continue
        file_path.rename(new_path)


def read_excel_any(file_path: Path, header=None, nrows: Optional[int] = None) -> pd.DataFrame:
    try:
        return pd.read_excel(file_path, header=header, nrows=nrows, engine="xlrd")
    except Exception:
        return pd.read_excel(file_path, header=header, nrows=nrows)


def detect_header_row(file_path: Path) -> int:
    preview = read_excel_any(file_path, header=None, nrows=10)
    for i in range(min(10, len(preview))):
        row_values = preview.iloc[i].astype(str).str.strip().tolist()
        joined = " ".join(row_values)
        if any(hint in joined for hint in DATE_HINTS):
            return i
    return 0


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def build_date_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    if {"年", "月", "日"}.issubset(df.columns):
        df = df.copy()
        df["date"] = pd.to_datetime(
            dict(year=df["年"], month=df["月"], day=df["日"]),
            errors="coerce",
        )
        return df, "date"
    return df, None


def parse_date_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if not non_null.empty:
            if non_null.between(19000101, 21000101).all():
                return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    text = series.astype(str).str.strip()
    digits_only = text.str.fullmatch(r"\d{8}")
    if digits_only.any() and digits_only.mean() > 0.6:
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def find_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if any(hint in col for hint in DATE_HINTS):
            return col
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    return None


def find_runoff_column(df: pd.DataFrame, date_col: str) -> Optional[str]:
    for col in df.columns:
        if col == date_col:
            continue
        if any(hint in col for hint in RUNOFF_HINTS):
            return col
    numeric_cols = [
        col
        for col in df.columns
        if col != date_col and pd.api.types.is_numeric_dtype(df[col])
    ]
    return numeric_cols[0] if numeric_cols else None


def is_runoff_file(file_name: str) -> bool:
    lowered = file_name.lower()
    return any(hint in lowered for hint in RUNOFF_HINTS)


def is_weather_file(file_name: str) -> bool:
    lowered = file_name.lower()
    return any(hint in lowered for hint in WEATHER_HINTS)


def load_runoff(file_path: Path) -> pd.DataFrame:
    header_row = detect_header_row(file_path)
    df = read_excel_any(file_path, header=header_row)
    df = normalize_columns(df)

    df, date_from_ymd = build_date_column(df)
    date_col = date_from_ymd or find_date_column(df)
    if not date_col:
        raise ValueError(f"No date column found in {file_path}")

    df[date_col] = parse_date_series(df[date_col])
    df = df.dropna(subset=[date_col])

    runoff_col = find_runoff_column(df, date_col)
    if not runoff_col:
        raise ValueError(f"No runoff column found in {file_path}")

    out = df[[date_col, runoff_col]].rename(columns={date_col: "date", runoff_col: "runoff"})
    out = out.sort_values("date")
    return out


def load_weather(file_path: Path) -> pd.DataFrame:
    header_row = detect_header_row(file_path)
    df = read_excel_any(file_path, header=header_row)
    df = normalize_columns(df)

    df, date_from_ymd = build_date_column(df)
    date_col = date_from_ymd or find_date_column(df)
    if not date_col:
        raise ValueError(f"No date column found in {file_path}")

    df[date_col] = parse_date_series(df[date_col])
    df = df.dropna(subset=[date_col])

    keep_cols = []
    for col in df.columns:
        if col == date_col:
            continue
        if any(hint in col for hint in META_HINTS):
            continue
        keep_cols.append(col)

    out = df[[date_col] + keep_cols].rename(columns={date_col: "date"})
    out = out.rename(columns=WEATHER_COLUMN_MAP)
    out = out.sort_values("date")
    return out


def merge_by_date(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame(columns=["date"])
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["date"], keep="first")
    df = df.sort_values("date")
    return df


def drop_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    return df.drop(columns=["date"])


def find_value_near_keyword(df: pd.DataFrame, keyword: str) -> Optional[float]:
    rows, cols = df.shape
    for r in range(rows):
        for c in range(cols):
            cell = df.iat[r, c]
            if isinstance(cell, str) and keyword in cell:
                for dr, dc in [(0, 1), (1, 0), (0, 2), (2, 0)]:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        val = df.iat[rr, cc]
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            return float(val)
                match = re.search(r"(-?\d+(?:\.\d+)?)", cell)
                if match:
                    return float(match.group(1))
    return None


def extract_station_info(file_path: Path) -> Dict[str, Optional[float]]:
    header_row = detect_header_row(file_path)
    df = read_excel_any(file_path, header=header_row)
    df = normalize_columns(df)

    lon_col = next((c for c in df.columns if "经度" in c or "lon" in c.lower()), None)
    lat_col = next((c for c in df.columns if "纬度" in c or "lat" in c.lower()), None)
    elev_col = next((c for c in df.columns if "海拔" in c or "elev" in c.lower()), None)
    remark_col = next((c for c in df.columns if "备注" in c), None)

    info: Dict[str, Optional[float]] = {
        "longitude": None,
        "latitude": None,
        "elevation": None,
        "station_name": None,
    }

    if lon_col and not df[lon_col].dropna().empty:
        info["longitude"] = float(df[lon_col].dropna().iloc[0])
    if lat_col and not df[lat_col].dropna().empty:
        info["latitude"] = float(df[lat_col].dropna().iloc[0])
    if elev_col and not df[elev_col].dropna().empty:
        info["elevation"] = float(df[elev_col].dropna().iloc[0])
    if remark_col and not df[remark_col].dropna().empty:
        info["station_name"] = str(df[remark_col].dropna().iloc[0]).strip()

    if info["longitude"] is None or info["latitude"] is None:
        raw = read_excel_any(file_path, header=None)
        info["longitude"] = info["longitude"] or find_value_near_keyword(raw, "经度")
        info["latitude"] = info["latitude"] or find_value_near_keyword(raw, "纬度")
        info["elevation"] = info["elevation"] or find_value_near_keyword(raw, "海拔")

    return info


def station_name_from_file(file_name: str) -> Optional[str]:
    match = re.search(r"\(([^)]+)\)", file_name)
    if match:
        return match.group(1)
    match = re.search(r"（([^）]+)）", file_name)
    if match:
        return match.group(1)
    return None


def format_station_info(
    river_cn: str,
    river_en: str,
    station_cn: Optional[str],
    station_en: Optional[str],
    info: Dict[str, Optional[float]],
) -> Dict[str, Optional[str]]:
    return {
        "river_cn": river_cn,
        "river_en": river_en,
        "station_cn": station_cn,
        "station_en": station_en,
        "longitude": info.get("longitude"),
        "latitude": info.get("latitude"),
        "elevation": info.get("elevation"),
    }


def process_river_dir(river_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, Optional[str]]]]:
    runoff_files = sorted(
        [p for p in river_dir.iterdir() if p.is_file() and is_runoff_file(p.name)]
    )
    weather_files = sorted(
        [p for p in river_dir.iterdir() if p.is_file() and is_weather_file(p.name)]
    )

    runoff_frames = [load_runoff(p) for p in runoff_files]
    weather_frames = [load_weather(p) for p in weather_files]

    runoff_df = merge_by_date(runoff_frames)
    weather_df = merge_by_date(weather_frames)

    merged_df = None
    if not runoff_df.empty and not weather_df.empty:
        merged_df = pd.merge(runoff_df, weather_df, on="date", how="left")

        weather_cols = [col for col in merged_df.columns if col not in ("date", "runoff")]
        merged_df = merged_df[["runoff"] + weather_cols]

    station_info = None
    if weather_files or runoff_files:
        river_cn = next((k for k, v in RIVER_NAME_MAP.items() if v == river_dir.name), river_dir.name)
        river_en = river_dir.name
        info = extract_station_info(weather_files[0]) if weather_files else {}

        if runoff_files and (
            info.get("longitude") is None
            or info.get("latitude") is None
            or info.get("elevation") is None
            or not info.get("station_name")
        ):
            runoff_info = extract_station_info(runoff_files[0])
            for key in ("longitude", "latitude", "elevation", "station_name"):
                if info.get(key) is None and runoff_info.get(key) is not None:
                    info[key] = runoff_info.get(key)

        station_cn = info.get("station_name") if info.get("station_name") else None
        if not station_cn and weather_files:
            station_cn = station_name_from_file(weather_files[0].name)
        if not station_cn and runoff_files:
            station_cn = station_name_from_file(runoff_files[0].name)

        station_en = STATION_NAME_MAP.get(station_cn, None) if station_cn else None
        station_info = format_station_info(river_cn, river_en, station_cn, station_en, info)

    return runoff_df, merged_df, station_info


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory not found: {RAW_DIR}")

    rename_river_dirs(RAW_DIR)

    name_map = combined_name_map()
    for river_dir in RAW_DIR.iterdir():
        if river_dir.is_dir():
            rename_files_in_dir(river_dir, name_map)

    RUNOFF_DIR.mkdir(parents=True, exist_ok=True)
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)

    station_rows: List[Dict[str, Optional[str]]] = []

    for river_dir in RAW_DIR.iterdir():
        if not river_dir.is_dir():
            continue

        runoff_df, merged_df, station_info = process_river_dir(river_dir)

        if runoff_df is not None and not runoff_df.empty:
            runoff_output = RUNOFF_DIR / f"{river_dir.name}.csv"
            drop_date_column(runoff_df).to_csv(runoff_output, index=False)

        if merged_df is not None and not merged_df.empty:
            merged_output = WEATHER_DIR / f"{river_dir.name}.csv"
            drop_date_column(merged_df).to_csv(merged_output, index=False)

        if station_info:
            station_rows.append(station_info)

    if station_rows:
        pd.DataFrame(station_rows).to_csv(STATIONS_CSV, index=False)


if __name__ == "__main__":
    main()
