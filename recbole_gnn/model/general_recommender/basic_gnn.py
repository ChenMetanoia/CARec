from typing import Tuple, Union
import pdb
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv import (
    EdgeConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    MessagePassing,
    PNAConv,
    SAGEConv,
    TransformerConv,
)
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils.trim_to_layer import TrimToLayer
from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole.model.loss import BPRLoss, EmbLoss
from model.init import xavier_uniform_initialization


class BasicGNN(GeneralGraphRecommender):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    def __init__(self, config, dataset, **kwargs):
        super().__init__(config, dataset)
        self.in_channels = config['in_channels']
        self.hidden_channels = config['hidden_channels']
        self.dropout = config['dropout']
        self.num_layers = config['num_layers']  # int type:the layer num
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']  # bool type: whether to require pow when regularization
        self.act = activation_resolver(config['activation_function'])
        self.loss_tyep = config['loss_type']
        if config['loss_type'] == 'BPR':
            self.mf_loss = BPRLoss()
            self.reg_loss = EmbLoss()
        
        if config['out_channels'] is not None:
            self.out_channels = config['out_channels']
        else:
            self.out_channels = config['hidden_channels']

        # define the user and item embedding
        if config['PLM'] is not None:
            print('item embedding is loaded from PLM')
            self.item_embedding = torch.nn.Embedding.from_pretrained(dataset.item_feat.item_emb) # default is not trainable
            self.latent_dim = dataset.item_feat.item_emb.shape[1]
        else:
            print('item embedding is random initilized and trainable')
            self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.in_channels)
        
        try:
            self.user_embedding = torch.nn.Embedding.from_pretrained(dataset.user_feat.user_emb)
            print('user embedding is loaded from average item embedding as initialization')
        except AttributeError:
            print('user embedding is random initilized and trainable')
            self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.in_channels)
        if config['fix_item_emb']:
            self.calculate_item_uniformity = False
        else:
            self.calculate_item_uniformity = True
        if config['fix_user_emb']:
            self.calculate_user_uniformity = False
        else:
            self.calculate_user_uniformity = True
        
        self.convs = ModuleList()
        if config['num_layers'] > 1:
            self.convs.append(
                self.init_conv(self.in_channels, self.hidden_channels, **kwargs))
            if isinstance(self.in_channels, (tuple, list)):
                self.in_channels = (self.hidden_channels, self.hidden_channels)
            else:
                self.in_channels = self.hidden_channels
        for _ in range(config['num_layers'] - 2):
            self.convs.append(
                self.init_conv(self.in_channels, self.hidden_channels, **kwargs))
            if isinstance(self.in_channels, (tuple, list)):
                self.in_channels = (self.hidden_channels, self.hidden_channels)
            else:
                self.in_channels = self.hidden_channels
        self.convs.append(
            self.init_conv(self.in_channels, self.hidden_channels, **kwargs))

        self.in_channels = self.hidden_channels
        self.lin = Linear(self.in_channels, self.out_channels)

        self.apply(xavier_uniform_initialization)
        # skip item embedding initialization if using PLM
        if not config['fix_item_emb'] and config['PLM'] is not None:
            print('skip text item embedding initialization, and make item text embedding trainable')
            self.item_embedding.weight.requires_grad = True # make it trainable
        elif not config['fix_item_emb']:
            print('item embedding is trainable')
        else:
            print('item embedding is fixed')
            
        if not config['fix_user_emb']:
            try:
                if dataset.user_feat.user_emb is not None:
                    print('skip user embedding initialization, and make user embedding trainable')
                    self.user_embedding.weight.requires_grad = True
            except AttributeError:
                print('user embedding is trainable')
        else:
            print('user embedding is fixed')
        
        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # TODO:
        self.norms = None
        self.jk_mode = None

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
    
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    @staticmethod
    def alignment(x, y, alpha=2):
        results = (x - y).norm(p=2, dim=1).pow(alpha)
        return results.mean()

    @staticmethod
    def uniformity(x, t=2):
        results = torch.pdist(x, p=2).pow(2).mul(-t).exp()
        return results.mean().log()
    
    def forward(self) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
        """
        x = self.get_ego_embeddings()
        embeddings_list = [x]
        for i in range(self.num_layers):
            if self.supports_edge_weight and self.supports_edge_attr:
                x = self.convs[i](x, self.edge_index, edge_weight=self.edge_weight)
            elif self.supports_edge_weight:
                x = self.convs[i](x, self.edge_index, edge_weight=self.edge_weight)
            else:
                x = self.convs[i](x, self.edge_index)
            if i == self.num_layers - 1:
                break
            if self.norms is not None:
                x = self.norms[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            embeddings_list.append(x)
        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
            
        user_all_embeddings, item_all_embeddings = torch.split(x, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings
    
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]

        if self.loss_tyep == 'BPR':
            pos_item = interaction[self.ITEM_ID]
            neg_item = interaction[self.NEG_ITEM_ID]
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
            
        else:
            item = interaction[self.ITEM_ID]
            align = self.alignment(user_all_embeddings, item_all_embeddings)
            if self.calculate_item_uniformity:
                # get unique set of item
                item, item_idx = np.unique(item.cpu().numpy(), return_index=True)
                # extract unique item embedding
                item_e_set = item_e[item_idx]
                # item uniformity loss
                item_uniform = self.gamma * self.uniformity(item_e_set)
            if self.calculate_user_uniformity:
                # get unique set of user
                user, user_idx = np.unique(user.cpu().numpy(), return_index=True)
                # extract unique user embedding
                user_e_set = user_e[user_idx]
                # user uniformity loss
                user_uniform = self.gamma * self.uniformity(user_e_set)
            if item_uniform is None and user_uniform is not None:
                return align + user_uniform
            elif item_uniform is not None and user_uniform is None:
                return align + item_uniform
            else:
                unifrom = (item_uniform + user_uniform) / 2
                loss = align + unifrom
        return loss
    
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        
        return scores.view(-1)
        
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class GCN(BasicGNN):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(in_channels, out_channels, **kwargs)


class GraphSAGE(BasicGNN):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """
    supports_edge_weight = False
    supports_edge_attr = False

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, **kwargs)


class GIN(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """
    supports_edge_weight = False
    supports_edge_attr = False

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=None,
        )
        return GINConv(mlp, **kwargs)


class GAT(BasicGNN):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout, **kwargs)


class Transformer(BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)
        beta = kwargs.pop('beta', False)
        dropout = kwargs.pop('dropout', 0.0)
        edge_dim = kwargs.pop('edge_dim', None)
        bias = kwargs.pop('bias', True)
        root_weight = kwargs.pop('root_weight', True)
        
        return TransformerConv(in_channels, out_channels, heads=heads, concat=concat,
                               beta=beta, dropout=dropout, edge_dim=edge_dim, bias=bias,
                               root_weight=root_weight, **kwargs)


class PNA(BasicGNN):
    r"""The Graph Neural Network from the `"Principal Neighbourhood Aggregation
    for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
    :class:`~torch_geometric.nn.conv.PNAConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.PNAConv`.
    """
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return PNAConv(in_channels, out_channels, **kwargs)


class EdgeCNN(BasicGNN):
    r"""The Graph Neural Network from the `"Dynamic Graph CNN for Learning on
    Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper, using the
    :class:`~torch_geometric.nn.conv.EdgeConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.EdgeConv`.
    """
    supports_edge_weight = False
    supports_edge_attr = False

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [2 * in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            # norm=self.norm,
            # norm_kwargs=self.norm_kwargs,
        )
        return EdgeConv(mlp, **kwargs)


__all__ = ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'PNA', 'EdgeCNN', 'Transformer']