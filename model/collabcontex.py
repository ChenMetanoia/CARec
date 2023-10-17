# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils.enum_type import InputType
from recbole_gnn.model.layers import LightGCNConv
from recbole.model.init import xavier_normal_initialization


class CollabContex(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # load parameters info
        self.gamma = 1
        self.encoder_name = config['encoder']

        # define layers and loss
        self.encoder = LGCNEncoder(dataset, config, device=self.device)

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def set_item_tutoring_phase(self):
        self.calculate_item_uniformity = False
        self.calculate_user_uniformity = True
        self.encoder.not_apply_mlp = True
        self.restore_item_e = None
        self.restore_user_e = None
        # make sure the mlps are not updated
        for param in self.encoder.mlp.parameters():
            param.requires_grad = False
        # make sure the user embedding is updated
        self.encoder.user_embedding.weight.requires_grad = True
    
    def set_user_tutoring_phase(self):
        self.calculate_item_uniformity = True
        self.calculate_user_uniformity = False
        self.encoder.not_apply_mlp = False
        self.restore_item_e = None
        self.restore_user_e = None
        # make sure the mlps are updated
        for param in self.encoder.mlp.parameters():
            param.requires_grad = True
        # make sure the user embedding is not updated
        self.encoder.user_embedding.weight.requires_grad = False
        
    def forward(self, user, item):
        user_e, item_e = self.encoder(user, item)
        return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)

    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e, item_e = self.forward(user, item)
        align = self.alignment(user_e, item_e)
        
        item_uniform = torch.tensor(0.0, device=user_e.device)
        user_uniform = torch.tensor(0.0, device=user_e.device)
        
        divisor = 0

        if self.calculate_item_uniformity:
            # get unique set of item
            item, item_idx = np.unique(item.cpu().numpy(), return_index=True)
            # extract unique item embedding
            item_e_set = item_e[item_idx]
            # item uniformity loss
            item_uniform = self.gamma * self.uniformity(item_e_set)
            divisor += 1
        if self.calculate_user_uniformity:
            # get unique set of user
            user, user_idx = np.unique(user.cpu().numpy(), return_index=True)
            # extract unique user embedding
            user_e_set = user_e[user_idx]
            # user uniformity loss
            user_uniform = self.gamma * self.uniformity(user_e_set)
            divisor += 1

        if divisor == 0:
            return align
        elif divisor == 1:
            return align + item_uniform + user_uniform
        else:
            return align + (item_uniform + user_uniform) / 2
        
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.encoder.get_all_embeddings()
        user_e = self.restore_user_e[user]
        all_item_e = self.restore_item_e
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
    

class LGCNEncoder(nn.Module):
    def __init__(self, dataset, config, device):
        super().__init__()
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.num_layers = config['num_layers']
        self.latent_dim = config['embedding_size']
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index, self.edge_weight = self.get_norm_adj_mat(dataset, device)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        
        self.not_apply_mlp = True
        self.mlp = MLP(
            config['mlp']['input_dim'], config['mlp']['hidden_dims'], 
            config['mlp']['output_dim'], config['mlp']['num_layers'],
            config['mlp']['dropout'])
        self.device = device
        self.item_embedding = dataset.item_feat.item_emb.to(device)
        self.latent_dim = dataset.item_feat.item_emb.shape[1]
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        
    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        if self.not_apply_mlp:
            item_embeddings = self.item_embedding
        else:
            item_embeddings = self.mlp(self.item_embedding)
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def get_all_embeddings(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.num_layers):
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings
        
    def forward(self, user_id, item_id):
        user_all_embeddings, item_all_embeddings = self.get_all_embeddings()
        u_embed = user_all_embeddings[user_id]
        i_embed = item_all_embeddings[item_id]
        return u_embed, i_embed
    
    def get_norm_adj_mat(self, dataset, device):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        row = dataset.inter_feat[dataset.uid_field]
        col = dataset.inter_feat[dataset.iid_field] + dataset.user_num
        # if the graph is bidirectional, we need to add the reverse direction
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        deg = degree(edge_index[0], dataset.user_num + dataset.item_num)

        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]
        
        return edge_index.to(device), edge_weight.to(device)
    
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_layers, dropout=0.2):
        super().__init__()
        
        # Initialize a list to hold the linear layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Combine all layers into a sequential module
        self.mlp = nn.Sequential(*layers)
        self.apply(xavier_normal_initialization)
    
    def forward(self, x):
        return self.mlp(x)