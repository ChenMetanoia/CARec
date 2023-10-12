# @Time   : 2022/3/8
# @Author : Lanling Xu
# @Email  : xulanling_sherry@163.com

r"""
LightGCN
################################################
Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import torch
import pdb

from torch.nn import ModuleList
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.layers import LightGCNConv, FeedForward

from model.init import xavier_uniform_initialization
from utils.util import get_user_emb

class LightGCN(GeneralGraphRecommender):
    r"""LightGCN is a GCN-based recommender model, implemented via PyG.
    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly 
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.
    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.num_layers = config['num_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']  # bool type: whether to require pow when regularization
            
        if config['PLM'] is not None:
            print('item embedding is loaded from PLM')
            self.item_embedding = torch.nn.Embedding.from_pretrained(dataset.item_feat.item_emb) # default is not trainable
            self.latent_dim = dataset.item_feat.item_emb.shape[1]
        else:
            print('item embedding is random initilized and trainable')
            self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
            
        if config['fix_user_emb']:
            user_emb = get_user_emb(dataset)
            self.user_embedding = torch.nn.Embedding.from_pretrained(user_emb)
            self.latent_dim = dataset.item_feat.item_emb.shape[1]
            self.fix_user_emb = True
        else:            
            self.fix_user_emb = False
            self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # FFN layer
        if config['FFN'] == 'multi':
            self.FFN = ModuleList()
            for i in range(self.num_layers):
                self.FFN.append(FeedForward(
                hidden_size=self.latent_dim,
                inner_size=self.latent_dim*2,
                hidden_dropout_prob=0.5,
                hidden_act='gelu',
                layer_norm_eps=1e-12,
                ))
        elif config['FFN'] == 'single':
            self.FFN = FeedForward(
                hidden_size=self.latent_dim,
                inner_size=self.latent_dim*2,
                hidden_dropout_prob=0.5,
                hidden_act='gelu',
                layer_norm_eps=1e-12,
                )
        else:
            self.FFN = None
            
        # parameters initialization
        self.apply(xavier_uniform_initialization)
        # skip item embedding initialization if using PLM
        if not config['fix_item_emb'] and config['PLM'] is not None:
            print('skip text item embedding initialization, and make item text embedding trainable')
            self.item_embedding.weight.requires_grad = True # make it trainable
            
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']
        
        
    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for layer_idx in range(self.num_layers):
            if self.FFN is not None and isinstance(self.FFN, torch.nn.ModuleList):
                user_all_embeddings = self.FFN[layer_idx](user_all_embeddings)
                all_embeddings = torch.cat([user_all_embeddings, item_all_embeddings], dim=0)
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        if self.FFN is not None and not isinstance(self.FFN, torch.nn.ModuleList):
            user_all_embeddings = self.FFN(user_all_embeddings)
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)