import torch
from torch import nn
import torch.nn.functional as F

import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score,matthews_corrcoef
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.metrics import confusion_matrix,root_mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr
import numpy as np 


class ContrastiveLoss(nn.Module):                   
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature  
        
    def forward(self, emb_i, emb_j):		
        device = emb_i.get_device()
        batch_size = emb_i.shape[0]
        z_i = F.normalize(emb_i, dim=1)     
        z_j = F.normalize(emb_j, dim=1)     

        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device) 
        representations = torch.cat([z_i, z_j], dim=0)          
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)     
        
        sim_ij = torch.diag(similarity_matrix, batch_size)      
        sim_ji = torch.diag(similarity_matrix, -batch_size)      
        positives = torch.cat([sim_ij, sim_ji], dim=0)                
        
        nominator = torch.exp(positives / self.temperature)          
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)       
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))      
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss


# Concordance Correlation Coefficient loss ver 1
# ====================================================
class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()
        self.sum = torch.sum
        self.mean = torch.mean
        self.var = torch.var
        self.std = torch.std

    def forward(self, pred, label):
        # pred = torch.squeeze(pred).t().type(torch.float64)
        # label = torch.squeeze(label).t().type(torch.float64)
        pred = pred.type(torch.float64)
        label = label.type(torch.float64)
        pred_mean = self.mean(pred)
        label_mean = self.mean(label)
        pred_var = self.var(pred, unbiased=True)
        label_var = self.var(label, unbiased=True)
        pred_std = self.std(pred, unbiased=True)
        label_std = self.std(label, unbiased=True)
        cov = self.sum((pred - pred_mean) * (label - label_mean)) / (len(pred) - 1)  
        pcc = cov / (pred_std * label_std)                                          
        ccc = (2 * cov) / (pred_var + label_var + (pred_mean - label_mean) ** 2 + 1e-6)   
        # return (1 - ccc).cuda() if torch.cuda.is_available() else (1 - ccc)
        return 1 - ccc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)       #避免除零

        return self.avg
    

# def compute_cls_metrics(yt, yp):
#     precision, recall, _, = precision_recall_curve(yt, yp)
#     aupr = -np.trapz(precision, recall)
#     auc = roc_auc_score(yt, yp)
#     # ---f1,acc,recall, specificity, precision
#     real_score = np.mat(yt)
#     predict_score = np.mat(yp)
#     sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
#     sorted_predict_score_num = len(sorted_predict_score)
#     thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
#     thresholds = np.mat(thresholds)
#     thresholds_num = thresholds.shape[1]
#     predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
#     negative_index = np.where(predict_score_matrix < thresholds.T)
#     positive_index = np.where(predict_score_matrix >= thresholds.T)
#     predict_score_matrix[negative_index] = 0
#     predict_score_matrix[positive_index] = 1
#     TP = predict_score_matrix.dot(real_score.T)
#     FP = predict_score_matrix.sum(axis=1) - TP
#     FN = real_score.sum() - TP
#     TN = len(real_score.T) - TP - FP - FN
#     tpr = TP / (TP + FN)
#     recall_list = tpr
#     precision_list = TP / (TP + FP)
#     f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
#     accuracy_list = (TP + TN) / len(real_score.T)
#     specificity_list = TN / (TN + FP)
#     max_index = np.argmax(f1_score_list)
#     f1_score = f1_score_list[max_index]
#     accuracy = accuracy_list[max_index]
#     specificity = specificity_list[max_index]
#     recall = recall_list[max_index]
#     precision = precision_list[max_index]
#     return auc, aupr, f1_score[0, 0], accuracy[0, 0]


def compute_reg_metrics(ytrue, ypred):                  #回归

    rmse = root_mean_squared_error(y_true=ytrue, y_pred=ypred)    
    r2 = r2_score(y_true=ytrue, y_pred=ypred)
    r, _ = pearsonr(ytrue, ypred)
    mae = mean_absolute_error(ytrue, ypred)
    return rmse, r2, r, mae

def compute_cls_metrics(y_true, y_prob):                #分类
    
    y_pred = np.array(y_prob) > 0.5
   
    auc = roc_auc_score(y_true, y_prob)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    aupr = -np.trapz(precision, recall)
   
    F1 = f1_score(y_true, y_pred, average = 'binary')

    acc = accuracy_score(y_true, y_pred)
   
    mcc = matthews_corrcoef(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return   auc, aupr, F1, acc


# def compute_reg_metrics(y_true, y_prob):
#     y_true = y_true.flatten().astype(float)
#     y_prob = y_prob.flatten().astype(float)
#     tau, p_value = stats.kendalltau(y_true, y_prob)
#     rho, pval =stats.spearmanr(y_true, y_prob)
#     r, _ =stats.pearsonr(y_true, y_prob)
#     rmse = mean_squared_error(y_true, y_prob, squared=False)
#     mae = mean_absolute_error(y_true, y_prob)
#     r2 = r2_score(y_true=y_true, y_pred=y_prob)
#     return tau, rho, r, rmse, mae, r2


# def compute_cls_metrics(y_true, y_prob):
    
#     y_pred = np.array(y_prob) > 0.5
   
#     roc_auc = roc_auc_score(y_true, y_prob)
   
#     F1 = f1_score(y_true, y_pred, average = 'binary')
   
#     mcc = matthews_corrcoef(y_true, y_pred)
    
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    

#     return   F1, roc_auc, mcc,  tn, fp, fn, tp


# def compute_reg_metrics(y_true, y_prob):
#     y_true = y_true.flatten().astype(float)
#     y_prob = y_prob.flatten().astype(float)
     
#     r2 = r2_score(y_true, y_prob)
#     r, _ =stats.pearsonr(y_true, y_prob)
#     rmse = mean_squared_error(y_true, y_prob, squared=False)
#     mae = mean_absolute_error(y_true, y_prob)
#     return  r2, r, rmse, mae