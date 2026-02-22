"""
评估指标计算模块
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_nse(y_true, y_pred):
    """
    计算Nash-Sutcliffe效率系数(NSE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        nse: NSE值
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if denominator == 0:
        return np.nan
    
    nse = 1 - (numerator / denominator)
    
    return nse


def calculate_mse(y_true, y_pred):
    """计算均方误差(MSE)"""
    return mean_squared_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """计算均方根误差(RMSE)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    """计算平均绝对误差(MAE)"""
    return mean_absolute_error(y_true, y_pred)


def calculate_mape(y_true, y_pred):
    """
    计算平均绝对百分比误差(MAPE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        mape: MAPE值(%)
    """
    # 避免除以零
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return mape


def calculate_r2(y_true, y_pred):
    """计算R²决定系数"""
    return r2_score(y_true, y_pred)


def calculate_all_metrics(y_true, y_pred):
    """
    计算所有评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        metrics: 包含所有评估指标的字典
    """
    metrics = {
        'NSE': calculate_nse(y_true, y_pred),
        'MSE': calculate_mse(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred)
    }
    
    return metrics


def print_metrics(metrics, dataset_name='Test'):
    """
    打印评估指标
    
    Args:
        metrics: 评估指标字典
        dataset_name: 数据集名称
    """
    print(f"\n{dataset_name} 评估指标:")
    print("=" * 50)
    for metric_name, value in metrics.items():
        if np.isnan(value):
            print(f"{metric_name}: NaN")
        elif metric_name in ['MAPE']:
            print(f"{metric_name}: {value:.2f}%")
        else:
            print(f"{metric_name}: {value:.4f}")
    print("=" * 50)
