# Tokenizer module to convert feature maps into visual tokens.

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models

class Projector(nn.Module):
    def __init__(self, CT, C, head=16, groups=16):
        super(Projector , self).__init__()
        self.proj_value_conv = nn.Conv1d(CT, C, kernel_size=1)
        self.proj_key_conv = nn.Conv1d(CT, C, kernel_size=1)
        self.proj_query_conv = nn.Conv2d(C, CT, kernel_size=1,groups=1)
        self.head = head

    def forward(self, feature, token):
        N, L, CT = token.shape
        token = token.view(N, CT, L)
        h = self.head
        proj_v = self.proj_value_conv(token).view(N, h, -1, L)
        proj_k = self.proj_key_conv(token).view(N, h, -1, L)
        proj_q = self.proj_query_conv(feature)
        N, C, H, W = proj_q.shape
        proj_q = proj_q.view(N, h, C // h, H * W).permute(0, 1, 3, 2)
        proj_coef = F.softmax(torch.Tensor.matmul(proj_q, proj_k) / np.sqrt(C / h), dim=3)
        proj = torch.Tensor.matmul(proj_v, proj_coef.permute(0, 1, 3, 2))
        _, _, H, W = feature.shape
        return feature + proj.view(N, -1, H, W), token


class Tokenizer(nn.Module):
    def __init__(self, L, CT, C, head=16, groups=16, dynamic=False, input_channels=512):
        super(Tokenizer , self).__init__()
         # Code for adjusting the channel sizes in case C is not equal to CT
        self.feature = nn.Conv2d(input_channels, C, kernel_size=1)
        if not dynamic :
            # use static weights to compute token coefficients.
            self.conv_token_coef = nn.Conv2d(C, L, kernel_size=1)
        else:
            # use previous tokens to compute a query weight, which is
            # then used to compute token coefficients.
            self.conv_query = nn.Conv1d(CT, C, kernel_size=1)
            self.conv_key = nn.Conv2d(C, C, kernel_size=1, groups=groups)
        self.conv_value = nn.Conv2d(C, C,kernel_size=1, groups=groups)
        self.head = head
        self.dynamic = dynamic
        self.C = C
        # self.transformer = nn.Transformer(nhead=16, num_encoder_layers=5, num_decoder_layers = 0, dim_feedforward=2048, activation='relu', dropout=0.5)


    def forward(self, feature, tokens=0):
        N, C, H, W = feature.shape
        if C != self.C:
            feature = self.feature(feature)
            # print("feature:",feature.shape)
        # compute token coefficients
        #feature: N, C, H, W, token: N, CT, L
        if not self.dynamic :
            token_coef = self.conv_token_coef(feature)
            N, L, H, W = token_coef.shape
            token_coef = token_coef.view(N, 1, L, H * W)
            token_coef = token_coef.permute(0, 1, 3, 2) # N, 1, HW, L
            token_coef = token_coef / np.sqrt(feature.shape[1])
        else:
            L = tokens.shape[2]
            # Split input tokens
            T_a, T_b = tokens[:, :, :L // 2], tokens[:, :, L // 2:]
            query = self.conv_query(T_a)
            N, C, L_a = query.shape
            query = query.view(N, self.head, C // self.head, L_a)
            N, C, H, W = feature.shape
            key = self.conv_key(feature).view(N, self.head, C // self.head, H * W) # N, h, C//h, HW
            # Compute token coefficients.
            # N, h, HW, L_a
            token_coef = torch.Tensor.matmul(key.permute(0, 1, 3, 2), query)
            token_coef = token_coef / np.sqrt(C / self.head)
        N, C, H, W = feature.shape
        token_coef = F.softmax(token_coef , dim=2)
        value = self.conv_value(feature).view(N, self.head, C // self.head, H * W) # N, h, C//h, HW
        # extract tokens from the feature map
        # static tokens: N, C, L. dynamic tokens: N, C, L_a
        tokens = torch.Tensor.matmul(value, token_coef).view(N, C, -1)
        tokens = tokens.view(N, L, C)
        # print(tokens.shape)
        # tokens = self.transformer(tokens, tokens)
        # print(tokens.shape)
        return tokens


class ViT(nn.Module):
    # Constructor
    def __init__(self, L, CT, C):   # C=token数，CT=维度，L
        super(ViT, self).__init__()
        self.bn = nn.BatchNorm2d(512)
        self.tokenizer = Tokenizer(L=L,CT=CT, C=C)
        self.transformer = nn.Transformer(d_model=768,nhead=16, num_encoder_layers=5, num_decoder_layers = 0, dim_feedforward=1024, activation='relu', dropout=0.5)
        # self.projector = Projector(CT=CT, C=C)
        # self.resnet = models.resnet34(pretrained=True)
        # self.feature_extractor = nn.Sequential(*list(self.projector.children())[:-1])

    def forward(self, x):
        import pdb
        x = self.bn(x)
        token = self.tokenizer(x)    # x.shape(bsz,C,H,W);token.shape(bsz,L,C)
        token = self.transformer(token, token)  # token维度不变,seqlen,bsz,dim
        token=token.permute(1,0,2)  # bsz,seqlen,dim
        # out, token = self.projector(x,token)
        return token


'''
图片先经过VT,再将feature经过vit
'''
class VT(nn.Module):
    # Constructor
    def __init__(self):   # C=token数，CT=维度，L
        super(VT, self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.resnet18_feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        import pdb
        feature = self.resnet18_feature_extractor(x)
        # out, token = self.projector(x,token)
        return feature
