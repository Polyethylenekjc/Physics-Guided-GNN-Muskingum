"""
LSTM径流预测管线 - 主执行脚本
支持单个或多个站点的训练和评估
"""
import os
import sys
import argparse
import torch

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.lstm_model import LSTMModel
from src.data_loader import DataProcessor, create_dataloaders
from src.trainer import Trainer, Evaluator
from src.metrics import calculate_all_metrics, print_metrics
from src.visualization import (plot_predictions, plot_scatter, 
                               plot_training_history, save_metrics_to_file,
                               save_predictions_to_file)
from config.lstm_config import (LSTM_CONFIG, TRAIN_CONFIG, DATA_CONFIG,
                                MODEL_DIR, OUTPUT_DIR, STATIONS,
                                FEATURE_SELECTION_CONFIG)


def run_station_pipeline(station_name, config=None):
    """
    运行单个站点的完整管线
    
    Args:
        station_name: 站点名称
        config: 配置字典（可选，使用默认配置）
    
    Returns:
        success: 是否成功
        metrics: 评估指标
    """
    print("\n" + "=" * 80)
    print(f"开始处理站点: {station_name}")
    print("=" * 80)
    
    # 使用默认配置或自定义配置
    if config is None:
        config = {
            'lstm': LSTM_CONFIG.copy(),
            'train': TRAIN_CONFIG.copy(),
            'data': DATA_CONFIG.copy(),
            'feature_selection': FEATURE_SELECTION_CONFIG.copy()
        }
    
    try:
        # ========== 步骤1: 加载数据 ==========
        print("\n[步骤 1/5] 加载数据...")
        data_processor = DataProcessor(
            station_name=station_name,
            weather_dir=config['data']['weather_dir'],
            sequence_length=config['train']['sequence_length'],
            stations=STATIONS,
            use_cross_station_features=config['feature_selection']['use_cross_station_features'],
            cross_station_top_k=config['feature_selection']['cross_station_top_k'],
            cross_station_use_abs=config['feature_selection']['cross_station_use_abs']
        )
        
        data = data_processor.load_data()
        
        # ========== 步骤2: 创建数据集 ==========
        print("\n[步骤 2/5] 创建数据集...")
        train_dataset, val_dataset, test_dataset = data_processor.create_sequences(
            data,
            train_ratio=config['train']['train_ratio'],
            val_ratio=config['train']['val_ratio']
        )
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=config['train']['batch_size']
        )
        
        # ========== 步骤3: 创建和训练模型 ==========
        print("\n[步骤 3/5] 创建并训练LSTM模型...")
        
        # 设置输入大小
        config['lstm']['input_size'] = data_processor.get_input_size()
        
        # 创建模型
        model = LSTMModel(
            input_size=config['lstm']['input_size'],
            hidden_size=config['lstm']['hidden_size'],
            num_layers=config['lstm']['num_layers'],
            output_size=config['lstm']['output_size'],
            dropout=config['lstm']['dropout'],
            bidirectional=config['lstm']['bidirectional']
        )
        
        print(f"\n模型结构:")
        print(model)
        print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=config['train']['learning_rate'],
            device=config['train']['device'],
            patience=config['train']['early_stopping_patience']
        )
        
        # 训练模型
        train_losses, val_losses = trainer.train(epochs=config['train']['epochs'])
        
        # 保存模型
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"{station_name}.pt")
        trainer.save_model(model_path)
        
        # ========== 步骤4: 评估模型 ==========
        print("\n[步骤 4/5] 评估模型...")
        
        # 在测试集上评估
        evaluator = Evaluator(
            model=model,
            data_loader=test_loader,
            data_processor=data_processor,
            device=config['train']['device']
        )
        
        y_true, y_pred = evaluator.predict()
        
        # 计算评估指标
        metrics = calculate_all_metrics(y_true, y_pred)
        print_metrics(metrics, dataset_name='Test')
        
        # ========== 步骤5: 可视化和保存结果 ==========
        print("\n[步骤 5/5] 保存结果和可视化...")
        
        station_output_dir = os.path.join(OUTPUT_DIR, station_name)
        os.makedirs(station_output_dir, exist_ok=True)

        # 保存跨站点变量选择信息
        selection_csv_path = os.path.join(
            station_output_dir,
            f"{station_name}_selected_cross_station_features.csv"
        )
        data_processor.save_selected_features_to_csv(selection_csv_path)
        
        # 保存训练历史
        plot_training_history(train_losses, val_losses, station_name, station_output_dir)
        
        # 保存预测对比图
        plot_predictions(y_true, y_pred, station_name, station_output_dir, metrics)
        
        # 保存散点图
        plot_scatter(y_true, y_pred, station_name, station_output_dir, metrics)
        
        # 保存评估指标
        save_metrics_to_file(metrics, station_name, station_output_dir)
        
        # 保存预测结果
        save_predictions_to_file(y_true, y_pred, station_name, station_output_dir)
        
        print("\n" + "=" * 80)
        print(f"站点 {station_name} 处理完成!")
        print(f"模型保存路径: {model_path}")
        print(f"结果保存目录: {station_output_dir}")
        print("=" * 80)
        
        return True, metrics
        
    except Exception as e:
        print(f"\n处理站点 {station_name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LSTM径流预测管线')
    parser.add_argument('--stations', nargs='+', 
                       help='要处理的站点名称（可以指定多个），留空则处理所有站点',
                       default=None)
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小（覆盖配置文件）')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='训练设备（覆盖配置文件）')
    
    args = parser.parse_args()
    
    # 确定要处理的站点
    if args.stations is None:
        stations_to_process = STATIONS
        print(f"未指定站点，将处理所有站点: {stations_to_process}")
    else:
        # 验证站点名称
        stations_to_process = []
        for station in args.stations:
            if station in STATIONS:
                stations_to_process.append(station)
            else:
                print(f"警告: 站点 '{station}' 不在配置的站点列表中，将跳过")
        
        if not stations_to_process:
            print("错误: 没有有效的站点需要处理")
            sys.exit(1)
    
    # 准备配置（应用命令行参数覆盖）
    config = {
        'lstm': LSTM_CONFIG.copy(),
        'train': TRAIN_CONFIG.copy(),
        'data': DATA_CONFIG.copy(),
        'feature_selection': FEATURE_SELECTION_CONFIG.copy()
    }
    
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
    if args.device is not None:
        config['train']['device'] = args.device
    
    # 打印配置信息
    print("\n" + "=" * 80)
    print("配置信息:")
    print(f"  要处理的站点数: {len(stations_to_process)}")
    print(f"  站点列表: {stations_to_process}")
    print(f"  训练轮数: {config['train']['epochs']}")
    print(f"  批次大小: {config['train']['batch_size']}")
    print(f"  序列长度: {config['train']['sequence_length']}")
    print(f"  设备: {config['train']['device']}")
    print(f"  隐藏层大小: {config['lstm']['hidden_size']}")
    print(f"  LSTM层数: {config['lstm']['num_layers']}")
    print(f"  启用跨站点特征: {config['feature_selection']['use_cross_station_features']}")
    print(f"  跨站点Top-K: {config['feature_selection']['cross_station_top_k']}")
    print(f"  按绝对Spearman排序: {config['feature_selection']['cross_station_use_abs']}")
    print("=" * 80)
    
    # 处理每个站点
    results = {}
    success_count = 0
    
    for i, station in enumerate(stations_to_process, 1):
        print(f"\n\n{'#' * 80}")
        print(f"进度: [{i}/{len(stations_to_process)}] 处理站点: {station}")
        print(f"{'#' * 80}")
        
        success, metrics = run_station_pipeline(station, config)
        results[station] = {
            'success': success,
            'metrics': metrics
        }
        
        if success:
            success_count += 1
    
    # 打印总结
    print("\n\n" + "=" * 80)
    print("处理完成! 总结:")
    print("=" * 80)
    print(f"总站点数: {len(stations_to_process)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(stations_to_process) - success_count}")
    print("\n各站点结果:")
    print("-" * 80)
    
    for station, result in results.items():
        if result['success']:
            nse = result['metrics'].get('NSE', 'N/A')
            r2 = result['metrics'].get('R2', 'N/A')
            print(f"✓ {station:30s} - NSE: {nse:.4f}, R²: {r2:.4f}")
        else:
            print(f"✗ {station:30s} - 处理失败")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
