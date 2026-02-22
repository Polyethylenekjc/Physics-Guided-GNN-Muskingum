"""
可视化模块
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import json


def plot_predictions(y_true, y_pred, station_name, save_dir, 
                    metrics=None, show_recent=500):
    """
    绘制预测结果对比图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        station_name: 站点名称
        save_dir: 保存目录
        metrics: 评估指标字典
        show_recent: 显示最近多少个数据点
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 图1: 全部数据
    axes[0].plot(y_true, label='True Runoff', linewidth=1.5, alpha=0.8)
    axes[0].plot(y_pred, label='Predicted Runoff', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Runoff')
    axes[0].set_title(f'{station_name} - Full Prediction vs True Values')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 图2: 最近的数据（看细节）
    if len(y_true) > show_recent:
        recent_true = y_true[-show_recent:]
        recent_pred = y_pred[-show_recent:]
        x_range = range(len(y_true) - show_recent, len(y_true))
    else:
        recent_true = y_true
        recent_pred = y_pred
        x_range = range(len(y_true))
    
    axes[1].plot(x_range, recent_true, label='True Runoff', 
                linewidth=1.5, alpha=0.8, marker='o', markersize=3)
    axes[1].plot(x_range, recent_pred, label='Predicted Runoff', 
                linewidth=1.5, alpha=0.8, marker='s', markersize=3)
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Runoff')
    axes[1].set_title(f'{station_name} - Recent {len(recent_true)} Predictions (Detail View)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 添加指标文本
    if metrics:
        metrics_text = '\n'.join([f'{k}: {v:.4f}' if k != 'MAPE' 
                                 else f'{k}: {v:.2f}%' 
                                 for k, v in metrics.items() if not np.isnan(v)])
        axes[0].text(0.02, 0.98, metrics_text, 
                    transform=axes[0].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10, family='monospace')
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, f'{station_name}_predictions.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"预测对比图已保存至: {save_path}")
    plt.close()


def plot_scatter(y_true, y_pred, station_name, save_dir, metrics=None):
    """
    绘制散点图（真实值 vs 预测值）
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        station_name: 站点名称
        save_dir: 保存目录
        metrics: 评估指标字典
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    
    # 散点图
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # 绘制理想线 (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Ideal (y=x)')
    
    plt.xlabel('True Runoff')
    plt.ylabel('Predicted Runoff')
    plt.title(f'{station_name} - Prediction Scatter Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加R²值
    if metrics and 'R2' in metrics:
        r2_text = f'R² = {metrics["R2"]:.4f}'
        plt.text(0.05, 0.95, r2_text, 
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=12)
    
    # 保存图片
    save_path = os.path.join(save_dir, f'{station_name}_scatter.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"散点图已保存至: {save_path}")
    plt.close()


def plot_training_history(train_losses, val_losses, station_name, save_dir):
    """
    绘制训练历史
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        station_name: 站点名称
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'{station_name} - Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 标记最佳验证损失
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    plt.plot(best_epoch, best_val_loss, 'r*', markersize=15, 
            label=f'Best (Epoch {best_epoch})')
    plt.legend()
    
    # 保存图片
    save_path = os.path.join(save_dir, f'{station_name}_training_history.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"训练历史图已保存至: {save_path}")
    plt.close()


def save_metrics_to_file(metrics, station_name, save_dir):
    """
    保存评估指标到文件
    
    Args:
        metrics: 评估指标字典
        station_name: 站点名称
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存为JSON
    json_path = os.path.join(save_dir, f'{station_name}_metrics.json')
    with open(json_path, 'w') as f:
        # 将numpy类型转换为Python类型
        metrics_serializable = {k: float(v) if not np.isnan(v) else None 
                               for k, v in metrics.items()}
        json.dump(metrics_serializable, f, indent=4)
    
    print(f"评估指标已保存至: {json_path}")
    
    # 保存为文本
    txt_path = os.path.join(save_dir, f'{station_name}_metrics.txt')
    with open(txt_path, 'w') as f:
        f.write(f"站点: {station_name}\n")
        f.write("=" * 50 + "\n")
        f.write("评估指标:\n")
        f.write("=" * 50 + "\n")
        for metric_name, value in metrics.items():
            if np.isnan(value):
                f.write(f"{metric_name}: NaN\n")
            elif metric_name == 'MAPE':
                f.write(f"{metric_name}: {value:.2f}%\n")
            else:
                f.write(f"{metric_name}: {value:.4f}\n")
        f.write("=" * 50 + "\n")
    
    print(f"评估指标已保存至: {txt_path}")


def save_predictions_to_file(y_true, y_pred, station_name, save_dir):
    """
    保存预测结果到CSV文件
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        station_name: 站点名称
        save_dir: 保存目录
    """
    import pandas as pd
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'True_Runoff': y_true,
        'Predicted_Runoff': y_pred,
        'Error': y_true - y_pred,
        'Absolute_Error': np.abs(y_true - y_pred),
        'Percentage_Error': np.abs((y_true - y_pred) / y_true) * 100
    })
    
    # 保存为CSV
    csv_path = os.path.join(save_dir, f'{station_name}_predictions.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"预测结果已保存至: {csv_path}")
