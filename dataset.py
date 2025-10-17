import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from torch_geometric.data import Data, Batch
import pandas as pd 
import numpy as np 
from sentence_transformers import SentenceTransformer
import pickle
import torch_geometric.transforms as T
import copy 
import os
 


class FPCaptionDSDataset(Dataset):                                      
    def __init__(self, synergy_data, drug_fp, drug_captions, cline2gene):
        
        self.drug1_ids = [row[0] for row in synergy_data]
        self.drug2_ids = [row[1] for row in synergy_data]
        self.cline = [row[2] for row in synergy_data]
        self.labels = [row[3] for row in synergy_data]
        self.cline2gene = cline2gene
        self.length = len(self.labels)

        self.drug_fp = drug_fp
        self.drug_captions = drug_captions
       
          

    def __getitem__(self, idx):
        # print(self.labels[idx])
        label = torch.tensor( float(self.labels[idx]), dtype=torch.float32)
        cline_gene = torch.tensor( self.cline2gene[self.cline[idx]], dtype=torch.float32)

        fp1 = self.drug_fp[int(self.drug1_ids[idx])]
        fp2 = self.drug_fp[int(self.drug2_ids[idx])]

        drug1_caption = torch.tensor(self.drug_captions[ int(self.drug1_ids[idx]) ])
        drug2_caption = torch.tensor(self.drug_captions[ int(self.drug2_ids[idx]) ])
        
 
        return  fp1, fp2,  cline_gene, drug1_caption, drug2_caption, label

    def __len__(self):
        return self.length


def get_data_with_clid(dataset):
    base_dir = os.path.dirname(__file__) 
    data_dir = os.path.join(base_dir, "Data")
    

    if dataset == 'ONEIL':
        drug_smiles_file = os.path.join(data_dir, "ONEIL-COSMIC", "drug_smiles.csv")
        drug_synergy_file = os.path.join(data_dir, "ONEIL-COSMIC", "drug_synergy.csv")
        gene_file = os.path.join(data_dir, "ONEIL-COSMIC", "cell line_gene_expression.csv")
    else:
        drug_smiles_file = os.path.join(data_dir, "ALMANAC-COSMIC", "drug_smiles.csv")
        drug_synergy_file = os.path.join(data_dir, "ALMANAC-COSMIC", "drug_synergy.csv")
        gene_file = os.path.join(data_dir, "ALMANAC-COSMIC", "cell line_gene_expression.csv")

         

             
    drug = pd.read_csv(drug_smiles_file, sep=',', header=0)             #逗号分隔，第一行用作列名
    drugid2smile = dict(zip(drug['pubchemid'], drug['isosmiles']))      #创建字典，前键后值
    drugsmile2name = dict(zip(drug['isosmiles'], drug['name']))

     

    synergy_load = pd.read_csv(drug_synergy_file, sep=',', header=0)
    gene_data = pd.read_csv(gene_file, sep=',', header=0, index_col=[0])
     
    cline_required = list(set(synergy_load['cell_line']))
    cline_num = len(cline_required)
    
    cline2id = dict(zip(cline_required, range(cline_num))) ##给每个细胞系编号
    id2cline = {value: key for key, value in cline2id.items()}                  #值为新键，键为新值

    cline2gene = {}
    for cline, cline_id in cline2id.items():
        cline2gene[cline] = np.array(gene_data.loc[cline].values, dtype='float32')
    gene_dim = gene_data.shape[1]
     

    synergy = [[  row[0], row[1], row[2], float(row[3]) ] for _, row in                #_用于忽略每一行的索引
               synergy_load.iterrows()]       #按行迭代，返回每一行的索引和行数据
   
   
    return synergy,   cline2gene
