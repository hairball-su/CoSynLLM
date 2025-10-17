import os
import shutil 
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as Data
from sklearn.model_selection import KFold
from dataset import *
from torch.utils.data import DataLoader
from metrics import compute_cls_metrics, compute_reg_metrics

from model import *
from early_stop import EarlyStopping
from prettytable import PrettyTable
import pickle

from metrics import AverageMeter
import warnings
from sklearn.model_selection import train_test_split
from FlagEmbedding import BGEM3FlagModel
import torch.optim.lr_scheduler as lr_scheduler
from nitrous_ema import PostHocEMA
from pretrained_model.muon import *
warnings.filterwarnings("ignore")

import random


def ptable_to_csv(table, filename, headers=True):
    """Save PrettyTable results to a CSV file.

    Adapted from @AdamSmith https://stackoverflow.com/questions/32128226

    :param PrettyTable table: Table object to get data from.
    :param str filename: Filepath for the output CSV.
    :param bool headers: Whether to include the header row in the CSV.
    :return: None
    """
    raw = table.get_string()
    data = [tuple(filter(None, map(str.strip, splitline)))
            for line in raw.splitlines()
            for splitline in [line.split('|')] if len(splitline) > 1]
    if table.title is not None:
        data = data[1:]
    if not headers:
        data = data[1:]
    with open(filename, 'w') as f:
        for d in data:
            f.write('{}\n'.format(','.join(d)))

def dy_tukey_loss(input, target, c):
    """
    Dynamic Tukey loss function

    """

    n = torch.abs(input - target)
    cond = n <= c
    loss = torch.where(cond, ((c** 2)/6) * (1- (1 - (n /c)**2) **3 )  , torch.tensor((c** 2)/6).to('cuda'))

    return loss.mean()

def dy_huber_loss(inputs, targets, beta):
    """
    Dynamic Huber loss function

    """
    n = torch.abs(inputs - targets)
    cond = n <= beta
    loss = torch.where(cond, 0.5 * n ** 2, beta*n - 0.5 * beta**2)

    return loss.mean()

def dy_smooth_l1_loss(inputs, targets, beta):
    """
    Dynamic ParamSmoothL1 loss function

    """
    n = torch.abs(inputs - targets)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2, n + 0.5 * beta**2 - beta)

    return loss.mean()

def run_a_train_epoch(device, epoch, model, data_loader, loss_criterion, optimizer, scheduler=None):
    model.train()
    tbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for id,  (*x, y) in tbar:
        for i in range(len(x)):
            x[i] = x[i].to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output   = model(*x)
        loss  = loss_criterion(output.view(-1), y.view(-1))
        loss.backward()
        optimizer.step()
        tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item()  :.3f}')
    if scheduler is not None:
        scheduler.step()

def run_an_eval_epoch(device, model, data_loader, task_name, loss_criterion):
    model.eval()
    running_loss = AverageMeter()

    with torch.no_grad():
        preds =  torch.Tensor()
        trues = torch.Tensor()
        for batch_id, (*x, y) in tqdm(enumerate(data_loader)):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)

            logits  =  model(*x)


            loss = loss_criterion(logits.view(-1), y.view(-1))

            if task_name == 'classification':
                logits = torch.sigmoid(logits)
            preds = torch.cat((preds, logits.cpu()), 0)
            trues = torch.cat((trues, y.view(-1, 1).cpu()), 0)
            running_loss.update(loss.item(), y.size(0))
        preds, trues = preds.numpy().flatten(), trues.numpy().flatten()
    val_loss =  running_loss.get_average()
    return preds, trues, val_loss


def data_split(synergy, test_size, rd_seed=42):
    synergy = np.array(synergy)

    train_data, test_data = train_test_split(synergy, test_size=test_size, random_state=rd_seed)

    return train_data, test_data


def process_data(synergy, drug2smile, cline2exp, task_name='regression'):
    processed_synergy = []
    for row in synergy:
        processed_synergy.append([drug2smile[int(row[0])], drug2smile[int(row[1])],
                                  row[2],  float(row[3])])


    if task_name == 'classification':
        threshold = 30
        for row in processed_synergy:
            row[3] = 1 if row[3] >= threshold else 0

    return np.array(processed_synergy, dtype=object)

if __name__ == '__main__':
    import argparse
    # python fp_main.py -d ONEIL -g 2
    # python fp_main.py -d ALMANAC -g 0

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-d', '--datasetname', default= 'ONEIL', type=str,
                      help='dataset name')
    parser.add_argument('-g', '--gpuid', default= '0', type=str,
                      help='GPU device')
    args = parser.parse_args()
    print(args)

    dataset_name =  args.datasetname #'ALMANAC'  # ONEIL or ALMANAC

    task_name = 'regression'  # regression or classification

    # random, leave_cline, leave_comb
    cv_mode_ls = [1,2,3]

    device = torch.device("cuda:"+args.gpuid if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    BATCH_SIZE = 512
    num_epochs = 200

    seed = 42

    lr = 5e-5

    synergy,   cline2gene  = get_data_with_clid(dataset_name)
    with open(f'{dataset_name}_drugfp.pkl', 'rb') as f:
        drug_graph_dict = pickle.load(f)

    with open(f'{dataset_name}_drugcaption_qw.pkl', 'rb') as f:
    # with open(f'{dataset_name}_drugcaption_gpt.pkl', 'rb') as f:
    # with open(f'{dataset_name}_drugcaption_ge.pkl', 'rb') as f:
    # with open(f'{dataset_name}_drugcaption_dsv3.pkl', 'rb') as f:
    # with open(f'{dataset_name}_drugcaption_dsr1.pkl', 'rb') as f:
        drug_caption_dict = pickle.load(f)

    model = SentenceTransformer('pretrained_model/sentence-transformers/all-mpnet-base-v2', device = device)
    drug2caption_emb = {}
    for key, value in drug_caption_dict.items():
        drug2caption_emb[key] =    model.encode( [ value ]).flatten()



    for cv_mode in cv_mode_ls:


        if task_name == 'classification':
            # k-fold val
            val_tables = PrettyTable(['Method', 'AUC', 'AUPR', 'F1', 'ACC'])
            # k-fold test
            t_tables = PrettyTable(['Method', 'AUC', 'AUPR', 'F1', 'ACC'])
            # 独立测试结果
            ind_tables = PrettyTable(['Method', 'AUC', 'AUPR', 'F1', 'ACC'])
        else:
            val_tables = PrettyTable(['Method', 'RMSE', 'R2', 'Pearson r', 'MAE'])
            t_tables = PrettyTable(['Method', 'RMSE', 'R2', 'Pearson r', 'MAE'])
            ind_tables = PrettyTable(['Method', 'RMSE', 'R2', 'Pearson r', 'MAE'])

        ind_tables.float_format = '.3'
        val_tables.float_format = '.3'
        t_tables.float_format = '.3'


        synergy_data, independent_test = data_split(synergy, test_size=0.1, rd_seed=seed)


        if cv_mode == 1:  # random split
            cv_data = synergy_data
        elif cv_mode == 2:  # leave_cline
            cv_data = np.unique(synergy_data[:, 2])
        else:  # leave_comb
            cv_data = np.unique(np.vstack([synergy_data[:, 0], synergy_data[:, 1]]), axis=1).T

        # 记录最终的五次平均
        test_mean = np.array([0., 0., 0., 0.])
        ind_mean = np.array([0., 0., 0., 0.])
        # leave_out操作在测试集上进行
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for test_fold, (cv_index, test_index) in enumerate(kf.split(cv_data)):
            if cv_mode == 1:
                synergy_cv, synergy_test = cv_data[cv_index], cv_data[test_index]
            elif cv_mode == 2:
                cline_cv, cline_test = cv_data[cv_index], cv_data[test_index]
                synergy_cv = np.array([i for i in synergy_data if i[2] in cline_cv])

                synergy_test = np.array([i for i in synergy_data if i[2] in cline_test])
            else:
                pair_cv, pair_test = cv_data[cv_index], cv_data[test_index]
                synergy_cv = np.array(
                    [j for i in pair_cv for j in synergy_data if (i[0] == j[0]) and (i[1] == j[1])])
                synergy_test = np.array(
                    [j for i in pair_test for j in synergy_data if (i[0] == j[0]) and (i[1] == j[1])])




            synergy_train, synergy_validation = data_split(synergy_cv, test_size=0.1, rd_seed=seed)


            trn_ds = FPCaptionDSDataset(synergy_train, drug_graph_dict,  drug2caption_emb, cline2gene)
            val_ds = FPCaptionDSDataset(synergy_validation, drug_graph_dict,  drug2caption_emb, cline2gene)
            test_ds = FPCaptionDSDataset(synergy_test, drug_graph_dict,  drug2caption_emb, cline2gene)



            train_loader = DataLoader(trn_ds,  batch_size= BATCH_SIZE, shuffle=True,  num_workers=4 )

            valid_loader = DataLoader(val_ds, batch_size= BATCH_SIZE, shuffle=False, num_workers=4 )
            test_loader = DataLoader(test_ds, batch_size= BATCH_SIZE, shuffle=False, num_workers=4 )


            test_preds, test_ys = [], []
            for n_ensembles in range(1):
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                model = FPCapDSModel().to(device)

                optimizer = get_optimizer('muon', model, lr=1e-4)

                model_path =   'model/fold_' + str(test_fold)+ '-' + dataset_name + '_model.pth'
                stopper = EarlyStopping(mode='lower', patience=25, filename=model_path)

                if task_name == 'classification':
                    loss_criterion = nn.BCEWithLogitsLoss()
                else:
                    loss_criterion = nn.MSELoss()


                for epoch in range(num_epochs):

                    # Train
                    run_a_train_epoch(device, epoch, model, train_loader, loss_criterion, optimizer, scheduler=None)
            #         # Validation and early stop
                    val_pred, val_true, val_loss = run_an_eval_epoch(device, model, valid_loader, task_name, loss_criterion)

                    if task_name == 'classification':
                        e_tables = PrettyTable(['Epoch', 'AUC', 'AUPR', 'F1', 'ACC'])
                        auc, aupr, f1_score, acc = compute_cls_metrics(val_true,val_pred)
                        row = [epoch, auc, aupr, f1_score, acc]
                    else:
                        e_tables = PrettyTable(['Epoch', 'RMSE', 'R2', 'Pearson r', 'MAE'])
                        rmse, r2, r, mae =  compute_reg_metrics(val_true,val_pred)
                        row = [epoch, rmse, r2, r, mae]

                    early_stop = stopper.step(val_loss, model)
                    e_tables.float_format = '.3'

                    e_tables.add_row(row)
                    print(e_tables)
                    if early_stop:
                        break
                stopper.load_checkpoint(model)
                test_pred, test_y, test_loss= run_an_eval_epoch(device, model, test_loader,task_name, loss_criterion)
                rmse, r2, r, mae =  compute_reg_metrics(test_y,test_pred)
                test_mean += np.array([rmse, r2, r, mae])
                row_test = [ 'test', rmse, r2, r, mae]
                t_tables.add_row(row_test)
                print(t_tables)


            print('---------------------------------------------------Test---------------------------------------------------')


        print('--------------------------------Final Results-----------------------------------')
        test_mean /= 5
        test_mean_row = ['mean', test_mean[0], test_mean[1], test_mean[2], test_mean[3]]
        t_tables.add_row(test_mean_row)
        print(t_tables)
        ptable_to_csv(t_tables, 'result_' +  dataset_name + str(cv_mode)+'.csv')
