import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from typing import Optional, Tuple
from torch import Tensor

import torch.nn as nn
import torch_geometric
from collections import OrderedDict
from einops.layers.torch import Rearrange, Reduce
from torch_geometric.nn import (
  TransformerConv,
  global_mean_pool as gap,
  global_add_pool as gsp,
  global_max_pool as gmp,
  TopKPooling
)
import torch.nn.functional as F

import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import math
import numpy as np
from einops import rearrange, repeat, reduce, pack, unpack
# from modules import BatchNorm, LayerNorm
# from eqgat import EQGATConv, EQGATConvNoCross, EQGATNoFeatAttnConv
from transformers import AutoTokenizer, AutoModel
import re



class PointWiseGateBlock(nn.Module):
    def __init__(self, hidden_size, dropout_prob=0.1):
        super(PointWiseGateBlock, self).__init__()

        self.inter_proj = nn.Linear(6 * hidden_size, hidden_size)  # 增强交互维度
        self.gate_proj = nn.Linear(hidden_size, hidden_size)

        # 轻量级前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, g):
        # 多种交互信息融合
        interaction = torch.cat([
            x, g,
            x * g,
            x - g,
            (x - g) ** 2,  # 新增非线性特征
            torch.abs(x - g)  # 新增绝对差
        ], dim=-1)

        interaction = self.inter_proj(interaction)
        interaction = F.gelu(interaction)

        gate = torch.sigmoid(self.gate_proj(interaction))

        mixed = gate * x + (1 - gate) * g

        # 残差增强+轻量FFN
        mixed = mixed + self.dropout(self.ffn(mixed))

        return self.layer_norm(mixed)


    
class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GLU, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        Y = self.W(X) * self.sigmoid(self.V(X))
        return Y

from torch import  einsum


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x, gate_res=None):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v) #+  gate_res
        out = u * v
        return out


class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        # self.attn =   Attention(d_model, d_ffn, d_model)
    def forward(self, x):
        # gate_res = self.attn(x)
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out



class graph11_new(nn.Module):
    def __init__(self, in_size, dropout=0.5):
        super(graph11_new, self).__init__()

        self.bifusion =  PointWiseGateBlock(in_size)
        self.bifusion2 =  PointWiseGateBlock(in_size)
        self.bifusion3 =  PointWiseGateBlock(in_size)

        depth = 1
        self.trifusion = nn.Sequential(
            *[gMLPBlock(in_size, in_size*4, 3) for _ in range(depth)]
        )

        self.attention_drug1 = GLU(in_size, in_size)
        self.attention_drug2 = GLU(in_size, in_size)
        self.attention_cell = GLU(in_size, in_size)

    def forward(self,cell_features, drug1_feature, drug2_feature):
        ###################### SFE layer  ##########################
        cell_features = self.attention_cell(cell_features)
        drug1_feature = self.attention_drug1(drug1_feature)
        drug2_feature = self.attention_drug1(drug2_feature)

        ##################### DFF layer ############################
        mafu_c_d1 =   self.bifusion( cell_features, drug1_feature )
        mafu_c_d2 =   self.bifusion( cell_features, drug2_feature)
        mafu_d1_d2 =   self.bifusion3( drug1_feature, drug2_feature)
        mafu_output = torch.stack([mafu_c_d1, mafu_c_d2, mafu_d1_d2], axis=1)


        ##################### TFI layer ############################
        tri_fusion_out = self.trifusion(mafu_output)
        tri_fusion_out = Reduce('b n d -> b d', reduction='mean')(tri_fusion_out)

        return  tri_fusion_out



class ResNet(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=4096, dropout=0.1, n_layers = 7):
        super().__init__()
        self.mlps = nn.ModuleList()
        for l in range(n_layers):
            self.mlps.append(
                nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                           nn.BatchNorm1d(hidden_dim),
                                      nn.GELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, embed_dim)))
        self.lin = nn.Linear(embed_dim, 1)
    def forward(self, x):
        for l in self.mlps:
            x = l(x) + x
        return self.lin(x)


class SimpleMLPBlock(nn.Module):
    def __init__(self, dim):
        super(SimpleMLPBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class SwiGLUBlock(nn.Module):
    def __init__(self, dim):
        super(SwiGLUBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x_proj, gate = self.fc1(x).chunk(2, dim=-1)
        x = x_proj * torch.nn.functional.silu(gate)
        x = self.fc2(x)
        return residual + x
    
class ChannelMixBlock(nn.Module):
    def __init__(self, dim):
        super(ChannelMixBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return x + self.fc(self.norm(x))

class DeepHybridFCResNet(nn.Module):
    def __init__(self, hidden_dim, num_blocks_stage1=2, num_blocks_stage2=2, num_blocks_stage3=2):
        super(DeepHybridFCResNet, self).__init__()

        self.initial = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1024)
        )

        stage1 = []
        for i in range(num_blocks_stage1):
            stage1.append(SimpleMLPBlock(1024))
            stage1.append(ChannelMixBlock(1024))  # 交替结构

        stage2 = []
        for i in range(num_blocks_stage2):
            stage2.append(SwiGLUBlock(1024))
            stage2.append(ChannelMixBlock(1024))

        stage3 = []
        for i in range(num_blocks_stage3):
            stage3.append(SwiGLUBlock(1024))
            stage3.append(SimpleMLPBlock(1024))

        self.stage1 = nn.Sequential(*stage1)
        self.stage2 = nn.Sequential(*stage2)
        self.stage3 = nn.Sequential(*stage3)


    def forward(self, x):
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        return x



class FPCapDSModel(nn.Module):
    def __init__(self,):
        super(FPCapDSModel, self).__init__()
        hidden_dim = 1024
        self.drug_model1 =  DeepHybridFCResNet(1024)
        self.drug_fc_cap1 =  DeepHybridFCResNet(768) 
        self.cell_fc = DeepHybridFCResNet(651) 
        self.top_fc = ResNet(hidden_dim)
        self.fusion_layer = graph11_new(hidden_dim)
        self.fusion_layer2 = graph11_new(hidden_dim)

    def forward(self, fp1, fp2, gene_exp, drug1_cap_emb, drug2_cap_emb):
        drug1_x = self.drug_model1(fp1)
        drug2_x = self.drug_model1(fp2)
        drug1_x_cap = self.drug_fc_cap1(drug1_cap_emb.float())
        drug2_x_cap = self.drug_fc_cap1(drug2_cap_emb.float())
        cl_f = self.cell_fc(gene_exp)


        fusion_features = self.fusion_layer(cl_f, drug1_x, drug2_x)
        fusion_features2 = self.fusion_layer2(cl_f, drug1_x_cap, drug2_x_cap)
        fusion_features =  fusion_features + fusion_features2

        pred  =  self.top_fc(fusion_features )

        return pred

