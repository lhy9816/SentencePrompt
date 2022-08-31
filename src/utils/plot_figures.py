"""
The util file for plotting metrics/loss
Author: Hangyu Li
Date: 02/06/2022
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set()
font_size = 13
label_size = 1


def draw_plot(pickle_file, metrics, metric_name, title_name, save_file, step_epoch=1):
    '''
    draw plot from pickle files or directly from matrices
    '''
    if pickle_file is None and metrics is None:
        raise ValueError("Please either provide a pickle file storing all eval metrics or the metric dict!")
    elif metrics is None:
        with open(pickle_file, 'rb') as f:
            metrics = pickle.load(f)

    metric_val = metrics[metric_name]
    x = np.arange(step_epoch, step_epoch * len(metric_val) + 1, step_epoch)
    plt.figure()
    plt.title(title_name)
    plt.xlabel('epoch', fontsize=font_size)
    plt.ylabel(metric_name, fontsize=font_size)
    plt.plot(x, metric_val, linewidth=2.0, color='b')
    plt.savefig(save_file)
    plt.close()


def draw_plot_mean_std_fill_between(pickle_file, metrics, mean_name, std_name, title_name, save_file, step_epoch=1):
    
    if pickle_file is None and metrics is None:
        raise ValueError("Please either provide a pickle file storing all eval metrics or the metric dict!")
    elif metrics is None:
        with open(pickle_file, 'rb') as f:
            metrics = pickle.load(f)

    mean_metric = np.asarray(metrics[mean_name])[:500]
    std_metric = np.asarray(metrics[std_name])[:500]
    x = np.arange(step_epoch, step_epoch * len(mean_metric) + 1, step_epoch)

    upper_metric = mean_metric + std_metric
    lower_metric = mean_metric - std_metric

    plt.figure()
    plt.title(title_name)
    plt.xlabel("epoch", fontsize=font_size)
    plt.ylabel(mean_name, fontsize=font_size)
    plt.plot(x, mean_metric, linewidth=2.0, color='b', label=mean_name)
    plt.fill_between(x, lower_metric, upper_metric, alpha=0.2, label=std_name)
    plt.legend()
    plt.savefig(save_file)
    plt.close()


def draw_plot_mean_std_subplot(pickle_file, metrics, mean_names, std_names, title_name, save_file, step_epoch=1):
    
    if pickle_file is None and metrics is None:
        raise ValueError("Please either provide a pickle file storing all eval metrics or the metric dict!")
    elif metrics is None:
        with open(pickle_file, 'rb') as f:
            metrics = pickle.load(f)
    
    n = len(mean_names)
    idx = 1
    plt.figure(figsize=(9, 9))
    plt.tick_params(labelsize=label_size)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
    for mean_name in mean_names:
        mean_metric = np.asarray(metrics[mean_name])
        x = np.arange(step_epoch, step_epoch * len(mean_metric) + 1, step_epoch)
        plt.subplot(2, n, idx)
        plt.xlabel("epoch", fontsize=font_size)
        plt.ylabel(mean_name, fontsize=font_size)
        plt.title(f"MLM eval mean {mean_name} vs epoch", fontsize=14)
        plt.plot(x, mean_metric, linewidth=2.0, color='b')
        idx += 1
    for std_name in std_names:
        std_metric = np.asarray(metrics[std_name])
        x = np.arange(1, len(std_metric) + 1)
        plt.subplot(2, n, idx)
        plt.xlabel("epoch", fontsize=font_size)
        plt.ylabel(std_name, fontsize=font_size)
        plt.title(f"MLM eval std {mean_name} vs epoch", fontsize=14)
        plt.plot(x, std_metric, linewidth=2.0, color='r')
        idx += 1
    
    plt.savefig(save_file)
    plt.close()


# def merge_two_bin(file1_path, file2_path, tgt_path):

#     with open(file1_path, 'rb') as f:
#         metrics1 = pickle.load(f)
#     with open(file2_path, 'rb') as f:
#         metrics2 = pickle.load(f)
    
#     for key in metrics1:
#         if key in metrics2:
#             metrics1[key] = metrics1[key][:500]
#             metrics1[key] += metrics2[key][:1000]

#     with open(tgt_path, 'wb') as f:
#         pickle.dump(metrics1, f)


# def draw_plot_multiple(pickle_files, dataset_names, metric_name, title_name, save_file, step_epoch=1):
#     # load all metrics
#     dataset_metrics = []
#     for pickle_file in pickle_files:
#         with open(pickle_file, 'rb') as f:
#             dataset_metrics.append(pickle.load(f))

#     plt.figure()
#     plt.title(title_name)
#     # eval metrics
#     for dataset_name, metrics in zip(dataset_names, dataset_metrics):
#         metric_val = metrics[metric_name][:500]
#         x = np.arange(step_epoch, step_epoch * len(metric_val) + 1, step_epoch)
#         plt.xlabel('epoch', fontsize=font_size)
#         plt.ylabel(metric_name, fontsize=font_size)
#         plt.plot(x, metric_val, linewidth=2.0, label=dataset_name)
#     plt.legend()
#     plt.savefig(save_file)
#     plt.close()


# '''
# draw plot from huggingface saved training states
# '''
# def draw_plot_trainer_state(trainer_state_file, metrics, metric_name, title_name, save_file, step_epoch=1):

#     if trainer_state_file is None and metrics is None:
#         raise ValueError("Please either provide a pickle file storing all eval metrics or the metric dict!")
#     elif metrics is None:
#         with open(trainer_state_file, 'r') as f:
#             metrics = json.load(f)
    
#     # get the metrics we need
#     metric_val = [history[metric_name] for history in metrics['log_history'] if 'eval_accuracy' in history]
#     x = np.arange(step_epoch, step_epoch * len(metric_val) + 1, step_epoch)
#     plt.figure()
#     plt.title(title_name)
#     plt.xlabel('epoch', fontsize=font_size)
#     plt.ylabel(metric_name, fontsize=font_size)
#     plt.plot(x, metric_val, linewidth=2.0, color='b')
#     plt.savefig(save_file)
#     plt.close()


# def draw_plot_multiple_trainer_state(trainer_state_files, diff_names, metric_name, title_name, save_file, start_epoch=0, step_epoch=1):
#     # load all metrics
#     dataset_metrics = []
#     for trainer_state_file in trainer_state_files:
#         with open(trainer_state_file, 'r') as f:
#             dataset_metrics.append(json.load(f))

#     plt.figure()
#     plt.title(title_name)
#     # eval metrics
#     for diff_name, metrics in zip(diff_names, dataset_metrics):
#         metric_val = [history[metric_name] for history in metrics['log_history'] if metric_name in history]
#         x = np.arange(start_epoch + step_epoch, start_epoch + step_epoch * len(metric_val) + 1, step_epoch)
#         plt.xlabel('epoch', fontsize=font_size)
#         plt.ylabel(metric_name.remove("reparam_"), fontsize=font_size)
#         plt.plot(x, metric_val, linewidth=2.0, label=diff_name)
#     plt.legend()
#     plt.savefig(save_file)
#     plt.close()


# def draw_plot_multiple_trainer_state_customized_epoch(trainer_state_files, diff_names, metric_name, title_name, save_file):
#     # load all metrics
#     dataset_metrics = []
#     for trainer_state_file in trainer_state_files:
#         with open(trainer_state_file, 'r') as f:
#             dataset_metrics.append(json.load(f))

#     plt.figure()
#     plt.title(title_name)
#     # eval metrics
#     for diff_name, metrics in zip(diff_names, dataset_metrics):
#         metric_items = [(int(history['epoch']), history[metric_name]) for history in metrics['log_history'] if metric_name in history and history[metric_name] > 0 and int(history['epoch']) % 5 == 0 and 200 <= int(history['epoch']) <= 300]
#         x = [item[0] for item in metric_items]
#         metric_val = [item[1] for item in metric_items]

#         plt.xlabel('epoch', fontsize=font_size)
#         plt.ylabel(metric_name.replace("reparam_", "").replace("eval_", "").replace("_fast", ""), fontsize=font_size)
#         plt.plot(x, metric_val, linewidth=2.0, label=diff_name)
#     plt.legend()
#     plt.savefig(save_file)
#     plt.close()


# # 0630
# def draw_three_acc():
#     # load lr 2e-3 and 1e-4 train states
#     diff_names = ['zero_init', 'random_init']
#     trainer_state_files = [
#         '../../checkpoints/roberta-base_CR/test_init_zero_eval-my-mask_scheme-random_mask-lossacc-trainepoch-500-prompt_length-2-lr-1e-4-scheduler-linear-weightdecay-0.01-batch_size-32-mlm_prob-0.25/trainer_state.json',
#         '../../checkpoints/roberta-base_CR/ttttt-eval-my-mask_scheme-random_mask-lossacc-trainepoch-500-prompt_length-2-lr-1e-4-scheduler-linear-weightdecay-0.01-batch_size-32-mlm_prob-0.25/trainer_state.json',
#     ]
#     dataset_metrics = []
#     for trainer_state_file in trainer_state_files:
#         with open(trainer_state_file, 'r') as f:
#             dataset_metrics.append(json.load(f))

#     fig, ax1 = plt.subplots(figsize=(8, 6))
#     ax2 = ax1.twinx()
#     plt.xlabel('epoch', fontsize=font_size)
#     plt.title("results on CR task")
#     colors = ['coral', 'teal']
#     # draw mlm accuracy
#     metric_name = "eval_accuracy"
#     ax2.set_yticks(np.arange(0.3, 0.9, 0.1)) 
#     ax2.set_ylim([0.3, 0.9])
#     ax2.set_ylabel('mlm_eval_mean_accuracy', fontsize=font_size)    
#     for i, (diff_name, metrics) in enumerate(zip(diff_names, dataset_metrics)):
#         metric_val = [history[metric_name] for history in metrics['log_history'] if metric_name in history]
#         x = np.arange(1, len(metric_val) + 1)   
#         ax2.plot(x, metric_val, linewidth=2.0, label=diff_name + ', mlmacc', color=colors[i], linestyle='-')
#     ax2.legend(loc=4)

#     ax1.set_yticks(np.arange(0.3, 0.9, 0.1)) 
#     ax1.set_ylim([0.25, 0.85])
#     ax1.set_ylabel('CR_devacc', fontsize=font_size)   
#     # draw only sentence-vector accuracy
#     metric_name = "eval_CR_reparam_devacc_onlysent_fast"
#     for i, (diff_name, metrics) in enumerate(zip(diff_names, dataset_metrics)):
#         metric_items = [(int(history['epoch']), history[metric_name]) for history in metrics['log_history'] if metric_name in history and history[metric_name] > 0 and int(history['epoch']) % 3 == 0]
#         x = [item[0] for item in metric_items]
#         metric_val = [item[1] / 100. for item in metric_items]
#         ax1.plot(x, metric_val, linewidth=2.0, label=diff_name + ', sentvector', color=colors[i], linestyle='--')
   
#     ax1.legend(loc=2)

#     plt.savefig("CR_testacc_result.png")
#     plt.close()


# metric_type = "loss"
# model_path = "../../checkpoints/roberta-base_CoordinationInversion/eval-my-mask_scheme-random_mask-lossacc-trainepoch-300-prompt_length-2-lr-2e-3-scheduler-linear-weightdecay-0.01-batch_size-128-mlm_prob-0.25"
# pickle_file = os.path.join(model_path, "merged_eval_result_per_epoch.bin")
# trainer_state_file = os.path.join(model_path, 'trainer_state.json')
# mean_metric = f"eval_{metric_type}_mean"
# std_metric = f"eval_{metric_type}_std"
# mean_title_name = f'mlm eval {metric_type} mean vs epoch'
# std_title_name = f'mlm eval {metric_type} std vs epoch'
# title_name = f"mlm eval {metric_type} mean+std vs epoch"
# save_file = os.path.join(model_path, f"eval_{metric_type}_mean_std_per_epoch_noscale.png")


# if __name__ == "__main__":
#     # draw_plot(pickle_file, None, mean_metric, title_name, save_file)
#     # draw_plot_mean_std_fill_between(pickle_file, None, mean_metric, std_metric, title_name, os.path.join(model_path, f"eval_{metric_type}_mean_std_per_epoch_noscale.png"))
#     # draw_plot_mean_std_subplot(pickle_file, None, 
#     #         ["eval_loss_mean", "eval_accuracy_mean"], ["eval_loss_std", "eval_accuracy_std"], "mlm eval loss and accuracy vs epoch", 
#     #         os.path.join(model_path, "lmerged_eval_loss_acc_mean_std_per_epoch_subplot.png")
#     #     )
#     # draw_plot_trainer_state(trainer_state_file, None, std_metric, std_title_name, 
#     #     os.path.join(model_path, f'eval_{metric_type}_std_per_epoch_trainerstate.png')
#     # )
#     # metrics1_file = "../../checkpoints/bert-base_CR/eval-mlmmy-lossacc-all_mask_input-trainepoch-1500-prompt_length-2-lr-5e-4-scheduler--weightdecay-0.00-batch_size-32-mlm_prob-1.0/eval_result_per_epoch.bin"
#     # metrics2_file = "../../checkpoints/bert-base_CR/eval-mlmmy-lossacc-all_mask_input-trainepoch-500-prompt_length-2-lr-5e-4-scheduler--weightdecay-0.00-batch_size-32-mlm_prob-1.0/eval_result_per_epoch.bin"
#     # tgt_file = "../../checkpoints/bert-base_CR/eval-mlmmy-lossacc-all_mask_input-trainepoch-500-prompt_length-2-lr-5e-4-scheduler--weightdecay-0.00-batch_size-32-mlm_prob-1.0/merged_eval_result_per_epoch.bin"
#     # merge_two_bin(metrics1_file, metrics2_file, tgt_file)
#     draw_three_acc()

#     # diff_names = ['init lr=2e-3', 'init lr=1e-4']
#     # trainer_state_files = [
#     #     '../../checkpoints/roberta-base_CoordinationInversion/eval-my-mask_scheme-random_mask-lossacc-trainepoch-300-prompt_length-2-lr-2e-3-scheduler-linear-weightdecay-0.01-batch_size-128-mlm_prob-0.25/trainer_state.json',
#     #     '../../checkpoints/roberta-base_CoordinationInversion/eval-my-mask_scheme-random_mask-lossacc-trainepoch-500-prompt_length-2-lr-1e-4-scheduler-linear-weightdecay-0.01-batch_size-32-mlm_prob-0.25/checkpoint-1294095/trainer_state.json',
#     # ]
#     # diff_item = 'illustrate_diff_init_lr'
#     # mean_metric = 'eval_accuracy' # "eval_CoordinationInversion_reparam_testacc"
#     # mean_title_name = 'mlm_mean_accuracy vs epoch'
#     # metric_type = 'mlm_mean_acc'
#     # draw_plot_multiple_trainer_state_customized_epoch(trainer_state_files, diff_names, mean_metric, mean_title_name, f"MLM_eval_{metric_type}_per_epoch_{diff_item}.png")
#     # mean_metric = 'eval_CoordinationInversion_reparam_testacc_onlysent_fast' # "eval_CoordinationInversion_reparam_testacc"
#     # mean_title_name = 'CoordinationInversion_testacc_only_sentvector vs epoch'
#     # metric_type = 'sentvector_only_testset_acc'
#     # draw_plot_multiple_trainer_state_customized_epoch(trainer_state_files, diff_names, mean_metric, mean_title_name, f"MLM_eval_{metric_type}_per_epoch_{diff_item}.png")
#     # mean_metric = 'eval_CoordinationInversion_reparam_testacc' # "eval_CoordinationInversion_reparam_testacc"
#     # mean_title_name = 'CoordinationInversion_testacc vs epoch'
#     # metric_type = 'concat_plmavg_testset_acc'
#     # draw_plot_multiple_trainer_state_customized_epoch(trainer_state_files, diff_names, mean_metric, mean_title_name, f"MLM_eval_{metric_type}_per_epoch_{diff_item}.png")
#     # # draw_plot_multiple(pickle_files, dataset_names, "eval_accuracy_std", "MLM eval std accuracy vs epoch", "MLM_eval_std_accuracy_per_epoch.png")