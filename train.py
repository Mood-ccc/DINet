import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from lstm_gnn.train import fold_train
from lstm_gnn.tool import set_log, set_train_argument, mkdir
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings('ignore')

from rdkit import rdBase
rdBase.DisableLog('rdApp.*')


if __name__ == '__main__':
    # 输入实验名称
    comment = input("Experiment name: ")
    args = set_train_argument()
    log = set_log('train', args.log_path)

    num_runs = 100
    top_k = 10
    primary_metric = 'r2'  # 按 test_r2 排序，越大越好

    original_save_path = args.save_path
    original_log_path = args.log_path
    mkdir(original_save_path)

    all_results = []  # 存储 (test_r2, result_dict)

    # 固定随机种子（全局一致）
    fixed_seed = args.seed
    np.random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fixed_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(fixed_seed)

    # 调试信息：确认 metrics
    print("=== DEBUG INFO ===")
    print(f"args.metric: {args.metric}")
    print(f"args.metrics: {args.metrics}")
    print(f"args.metrics type: {type(args.metrics)}")
    print("=== END DEBUG INFO ===")

    log.info(f"Starting up to {num_runs} runs with fixed seed = {fixed_seed}")
    log.info(f"Results will be saved to: {original_save_path}")
    log.info(f"Top-{top_k} will be selected based on test_{primary_metric} (higher is better)")

    # === 检查已完成的 Run_X ===
    completed_runs = []
    for run_idx in range(1, num_runs + 1):
        run_path = os.path.join(original_save_path, f"Run_{run_idx}")
        result_file = os.path.join(run_path, "test_scores.npy")
        val_file = os.path.join(run_path, "val_scores.npy")
        train_file = os.path.join(run_path, "train_scores.npy")

        if os.path.exists(result_file):
            try:
                # 加载所有分数文件
                test_score = np.load(result_file, allow_pickle=True).item()
                val_score = np.load(val_file, allow_pickle=True).item()
                train_score = np.load(train_file, allow_pickle=True).item()

                if primary_metric in test_score:
                    test_r2 = np.nanmean(test_score[primary_metric])
                    val_r2 = np.nanmean(val_score[primary_metric]) if primary_metric in val_score else float('nan')

                    # 构建 result 字典
                    result = {}
                    for metric in args.metrics:
                        result[f'train_{metric}'] = np.nanmean(train_score[metric]) if metric in train_score else float('nan')
                        result[f'val_{metric}'] = np.nanmean(val_score[metric]) if metric in val_score else float('nan')
                        result[f'test_{metric}'] = np.nanmean(test_score[metric]) if metric in test_score else float('nan')

                    # >>>>>>>>>> 新增：获取 RMSE <<<<<<<<<<
                    test_rmse = result.get('test_rmse', float('nan'))
                    val_rmse = result.get('val_rmse', float('nan'))

                    completed_runs.append((test_r2, result))
                    # >>>>>>>>>> 修改日志：同时打印 r2 和 rmse <<<<<<<<<<
                    log.info(f"Run {run_idx} already completed. Test r2: {test_r2:.6f}, Test rmse: {test_rmse:.6f}, Val r2: {val_r2:.6f}, Val rmse: {val_rmse:.6f}")
                else:
                    log.info(f"Run {run_idx} missing '{primary_metric}', will re-run.")
            except Exception as e:
                log.info(f"Run {run_idx} result file corrupted: {e}. Will re-run.")
        else:
            log.info(f"Run {run_idx} not found or incomplete. Will run.")

    # 将已完成的结果加入 all_results
    all_results.extend(completed_runs)
    num_completed = len(completed_runs)
    log.info(f"Found {num_completed} completed runs. Need to run {num_runs - num_completed} more.")

    # === 运行未完成的 runs ===
    for run_idx in range(1, num_runs + 1):
        run_path = os.path.join(original_save_path, f"Run_{run_idx}")
        result_file = os.path.join(run_path, "test_scores.npy")

        # 检查是否已完成
        if os.path.exists(result_file):
            try:
                test_score = np.load(result_file, allow_pickle=True).item()
                if primary_metric in test_score:
                    continue  # 已完成，跳过
            except:
                pass  # 文件损坏，重新运行

        # 创建目录
        mkdir(run_path)

        # 设置参数
        args.save_path = run_path
        args.log_path = original_log_path
        args.seed = fixed_seed  # 固定 seed

        log.info(f"Starting Run {run_idx}/{num_runs}")

        try:
            writer = SummaryWriter(log_dir=run_path)
            train_score, val_score, test_score, _, _ = fold_train(args, log, writer)
            writer.close()

            # 保存分数
            np.save(result_file, test_score)
            np.save(os.path.join(run_path, "val_scores.npy"), val_score)
            np.save(os.path.join(run_path, "train_scores.npy"), train_score)

            # 构建 result 字典
            result = {}
            for metric in args.metrics:
                result[f'train_{metric}'] = np.nanmean(train_score[metric])
                result[f'val_{metric}'] = np.nanmean(val_score[metric])
                result[f'test_{metric}'] = np.nanmean(test_score[metric])

            # >>>>>>>>>> 新增：提取 r2 和 rmse <<<<<<<<<<
            test_r2 = result.get('test_r2', float('nan'))
            test_rmse = result.get('test_rmse', float('nan'))
            val_r2 = result.get('val_r2', float('nan'))
            val_rmse = result.get('val_rmse', float('nan'))

            all_results.append((test_r2, result))
            # >>>>>>>>>> 修改日志：同时打印 r2 和 rmse <<<<<<<<<<
            log.info(f"Run {run_idx} completed. Test r2: {test_r2:.6f}, Test rmse: {test_rmse:.6f}, Val r2: {val_r2:.6f}, Val rmse: {val_rmse:.6f}")

        except Exception as e:
            log.error(f"Run {run_idx} failed with error: {str(e)}", exc_info=True)
            continue

    # === 所有 runs 完成，开始分析 ===
    log.info(f"All runs completed. Total valid results: {len(all_results)}")

    # --- 分析 1: 基于 test_r2 的 Top-10 ---
    all_results.sort(key=lambda x: x[0], reverse=True)  # test_r2 越大越好
    top10_test = all_results[:top_k]

    summary_test_top = {}
    for dataset in ['train', 'val', 'test']:
        for metric in args.metrics:
            key = f'{dataset}_{metric}'
            values = [res[key] for _, res in top10_test if key in res and not np.isnan(res[key])]
            if len(values) == 0:
                log.warning(f"No values found for {key} in test-based top-{top_k}.")
                summary_test_top[key] = (float('nan'), float('nan'))
            else:
                mean_val = np.mean(values)
                std_val = np.std(values)
                summary_test_top[key] = (mean_val, std_val)
                log.info(f"Top-10 (test_{primary_metric}) {key}: {mean_val:.6f} ± {std_val:.6f}")

    # --- 分析 2: 基于 val_r2 的 Top-10 ---
    log.info("Now selecting top-10 based on val_r2...")
    val_primary_key = f'val_{primary_metric}'
    val_based_list = []

    for _, result_dict in all_results:
        if val_primary_key in result_dict and not np.isnan(result_dict[val_primary_key]):
            val_based_list.append((result_dict[val_primary_key], result_dict))

    val_based_list.sort(key=lambda x: x[0], reverse=True)
    top10_val = val_based_list[:top_k]

    log.info(f"Selected {len(top10_val)} runs for val-based top-{top_k} analysis.")

    summary_val_top = {}
    for dataset in ['train', 'val', 'test']:
        for metric in args.metrics:
            key = f'{dataset}_{metric}'
            values = [res[key] for _, res in top10_val if key in res and not np.isnan(res[key])]
            if len(values) == 0:
                log.warning(f"No values found for {key} in val-based top-{top_k}.")
                summary_val_top[key] = (float('nan'), float('nan'))
            else:
                mean_val = np.mean(values)
                std_val = np.std(values)
                summary_val_top[key] = (mean_val, std_val)
                log.info(f"Top-10 (val_{primary_metric}) {key}: {mean_val:.6f} ± {std_val:.6f}")

    # === 保存 dual summary 到文件 ===
    final_summary_path = os.path.join(original_save_path, 'dual_top10_summary.txt')
    with open(final_summary_path, 'w') as f:
        f.write(f"Dual Top-10 Summary (from {num_runs} runs, seed={fixed_seed})\n")
        f.write(f"Selection metrics: test_{primary_metric} and val_{primary_metric} (higher is better)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"== [1] Top-10 Selected by test_{primary_metric} ==\n")
        for key, (m, s) in summary_test_top.items():
            f.write(f"{key}: {m:.6f} ± {s:.6f}\n")
        f.write("\n")

        f.write(f"== [2] Top-10 Selected by val_{primary_metric} ==\n")
        for key, (m, s) in summary_val_top.items():
            f.write(f"{key}: {m:.6f} ± {s:.6f}\n")

    log.info(f"Dual top-10 summary saved to {final_summary_path}")
    print(f" Experiment completed. Results saved in: {original_save_path}")