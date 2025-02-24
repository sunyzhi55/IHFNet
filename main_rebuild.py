import argparse
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold
from Dataset import MriPetDataset, MriPetDatasetNew
import torch.utils.data
from model_object import models
from Config import parse_args
from observer import Runtime_Observer
from Net.api import *
import numpy as np
#99480885
#
def prepare_to_train(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_index,
                     seed, device, data_parallel, n_splits, others_params):
    global experiment_settings
    assert torch.cuda.is_available(), "Please ensure codes are executed on cuda."
    try:
        experiment_settings = models[model_index]
    except KeyError:
        print('model not in model_object!')
    torch.cuda.empty_cache()

    # 初始化数据集
    # dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file, valid_group=("pMCI", "sMCI"))
    dataset = MriPetDatasetNew(mri_dir, pet_dir, cli_dir, csv_file, valid_group=("pMCI", "sMCI"))
    torch.manual_seed(seed)

    # K折交叉验证
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = [data[3] for data in dataset]  # 假设dataset[i]的第3项是label
    # 存储每个fold的评估指标
    metrics = {
        'accuracy': [],
        'auc': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'Specificity': []
    }

    # for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
    for fold, (train_index, test_index) in enumerate(skf.split(dataset, labels), 1):
        print(f'Fold {fold + 1}/{5}')
        train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
        val_sampler = torch.utils.data.SubsetRandomSampler(test_index)
        trainDataLoader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size,
                                                      num_workers=4, drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(dataset, sampler=val_sampler, batch_size=batch_size,
                                                     num_workers=4, drop_last=True)

        # 分割数据集
        # train_dataset = torch.utils.data.Subset(dataset, train_index)
        # test_dataset = torch.utils.data.Subset(dataset, test_index)
        #
        # trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
        #                                               num_workers=4, drop_last=True)
        # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
        #                                              num_workers=4)

        # 训练日志和监控
        target_dir = Path('./logs/')
        target_dir.mkdir(exist_ok=True)
        target_dir = target_dir.joinpath('classification')
        target_dir.mkdir(exist_ok=True)
        current_time = str(datetime.now().strftime('%Y-%m-%d_%H-%M'))
        target_dir = target_dir.joinpath(experiment_settings['Name'] + f'_{current_time}_fold_{fold + 1}')
        target_dir.mkdir(exist_ok=True)
        checkpoints_dir = target_dir.joinpath('checkpoints')
        checkpoints_dir.mkdir(exist_ok=True)
        log_dir = target_dir.joinpath('logs')
        log_dir.mkdir(exist_ok=True)

        observer = Runtime_Observer(log_dir=log_dir, device=device, name=experiment_settings['Name'], seed=seed)
        observer.log(f'[DEBUG] Observer init successfully, program start @{current_time}\n')

        # 模型加载
        _model = experiment_settings['Model']
        print(f"The name of model will run {_model}")
        model = _model()

        # 冻结Resnet层的参数
        # for name, param in model.named_parameters():
        #     if "Resnet" in name:
        #         param.requires_grad = False

        # 使用 DataParallel 进行多GPU训练
        if torch.cuda.device_count() > 1 and data_parallel == 1:
            observer.log("Using " + str(torch.cuda.device_count()) + " GPUs for training.\n")
            model = torch.nn.DataParallel(model)

        observer.log(f'Use model : {str(experiment_settings)}\n')
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        observer.log("\n===============================================\n")
        observer.log("model parameters: " + str(num_params))
        observer.log("\n===============================================\n")

        # 超参数设置
        optimizer = experiment_settings['Optimizer'](model.parameters(), experiment_settings['Lr'])
        scheduler = experiment_settings['Scheduler'](optimizer, others_params)
        # 定义一个filter，只传入requires_grad=True的模型参数
        # optimizer = experiment_settings['Optimizer'](filter(lambda p: p.requires_grad, model.parameters()),
        #                                              experiment_settings['Lr'])

        if 'w1' in experiment_settings:
            criterion = experiment_settings['Loss'](w1=experiment_settings['w1'], w2=experiment_settings['w2'])
        else:
            criterion = experiment_settings['Loss']()

        print("Prepare completed for fold {}! Launch training!\U0001F680".format(fold))

        # 启动训练
        _run = experiment_settings['Run']
        _run(observer, experiment_settings['Epoch'], trainDataLoader, testDataLoader, model, device,
             optimizer, criterion, scheduler)

        # 收集评估指标
        metrics['accuracy'].append(observer.best_dicts['acc'])
        metrics['auc'].append(observer.best_dicts['auc'])
        metrics['f1'].append(observer.best_dicts['f1'])
        metrics['precision'].append(observer.best_dicts['p'])
        metrics['recall'].append(observer.best_dicts['recall'])
        metrics['Specificity'].append(observer.best_dicts['spe'])

    # for key in metrics:
    #     mean_value = np.mean(metrics[key].cpu().numpy())
    #     std_value = np.std(metrics[key].cpu().numpy())
    #     print(f'{key.capitalize()} - Mean: {mean_value:.4f}, Std: {std_value:.4f}')
    print(f'ACC:{metrics}')

    print("Cross-validation training completed for all folds.")

if __name__ == "__main__":

    args = parse_args()
    print(args)
    # prepare_to_train(model_index=args.model, mri_dir=args.mri_dir, pet_dir=args.pet_dir, cli_dir=args.cli_dir, csv_file=args.csv_file, batch_size=args.batch_size, seed=args.seed , device=args.device, fold=args.fold, data_parallel=args.data_parallel)
    prepare_to_train(model_index=args.model, mri_dir=args.mri_dir, pet_dir=args.pet_dir,
                     cli_dir=args.cli_dir,csv_file=args.csv_file, batch_size=args.batch_size,
                     seed=args.seed, device=args.device, data_parallel=args.data_parallel,
                     n_splits=args.n_splits, others_params=args)