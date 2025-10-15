from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import csv
import os
def regression_metrics(y_true, y_pred):
    """
    计算回归模型的多种评估指标
    :param y_true: 真实值 (array-like)
    :param y_pred: 预测值 (array-like)
    :return: 指标字典
    """
    # 转为 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 均方误差（MSE）
    mse = mean_squared_error(y_true, y_pred)

    # 均方根误差（RMSE）
    rmse = np.sqrt(mse)

    # 平均绝对误差（MAE）
    mae = mean_absolute_error(y_true, y_pred)

    # 决定系数（R²）
    r2 = r2_score(y_true, y_pred)

    # 平均绝对百分比误差（MAPE）
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # 对称平均绝对百分比误差（sMAPE）
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

    # 对数误差（Log Error）
    log_error = np.mean((np.log(y_true + 1) - np.log(y_pred + 1)) ** 2)

    bias = np.mean(y_pred - y_true)

    # 指标字典
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape,
        'sMAPE (%)': smape,
        'Log Error': log_error,
        'bias': bias
    }
    return metrics



def process_metrics(y_true, y_pred, use_time):
    """
    计算并返回评估指标
    :param y_true: 真实值
    :param y_pred: 预测值
    :param use_time: 时间消耗
    :param bias: 偏差
    :return: 指标字典
    """
    metrics = regression_metrics(y_true, y_pred)
    return {
        "MSE": metrics["MSE"],
        "RMSE": metrics["RMSE"],
        "R²": metrics["R²"],
        "MAPE": metrics["MAPE (%)"],
        "SMAPE": metrics["sMAPE (%)"],
        "LOG": metrics["Log Error"],
        "MAE": metrics["MAE"],
        "Bias": metrics["bias"],
        "Time": use_time,
    }

def write_results_to_csv(csv_file, results):
    """
    将结果写入 CSV 文件
    """
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # 写入标题行
        writer.writerow(["Scenario", "Number of Obs."] +
                        ["MSE_train", "RMSE_train", "R2_train", "MAPE_train", "SMAPE_train",
                         "LOG_train", "MAE_train", "Bias_train", "Time_train"] +
                        ["MSE_test", "RMSE_test", "R2_test", "MAPE_test", "SMAPE_test",
                         "LOG_test", "MAE_test", "Bias_test", "Time_test"])
        # 写入数据
        for scenario, obs_data in results.items():
            for num_obs, values in obs_data.items():
                writer.writerow([scenario, num_obs] + values)



def save_metrics_to_csv(train_metrics_list, test_metrics_list, model, scien, num, file_dir="results"):
    """
    将 train_metrics_list 和 test_metrics_list 写入 CSV 文件，并为不同的 scien 和 num 创建不同的文件

    :param train_metrics_list: 训练阶段的指标列表
    :param test_metrics_list: 测试阶段的指标列表
    :param scien: 当前的科学分类
    :param num: 当前的样本数
    :param file_dir: 保存文件的目录
    """
    # 创建存储目录
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    # 创建文件路径
    file_path = os.path.join(file_dir, f"{model}_{scien}_num_{num}.csv")

    # 写入 CSV 文件
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # 写入标题行
        metric_keys = list(train_metrics_list[0].keys())
        writer.writerow(["Run", "Phase"] + metric_keys)

        # 写入训练数据
        for i, metrics in enumerate(train_metrics_list):
            writer.writerow([i + 1, "Train"] + [metrics[key] for key in metric_keys])

        # 写入训练平均值
        avg_train_metrics = {key: np.round(np.mean([m[key] for m in train_metrics_list]), 5) for key in metric_keys}
        writer.writerow(["Average", "Train"] + [avg_train_metrics[key] for key in metric_keys])

        # 写入测试数据
        for i, metrics in enumerate(test_metrics_list):
            writer.writerow([i + 1, "Test"] + [metrics[key] for key in metric_keys])

        # 写入测试平均值
        avg_test_metrics = {key: np.round(np.mean([m[key] for m in test_metrics_list]), 5) for key in metric_keys}
        writer.writerow(["Average", "Test"] + [avg_test_metrics[key] for key in metric_keys])

    print(f"Metrics for {scien} with num {num} saved to {file_path}")
