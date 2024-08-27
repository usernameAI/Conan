import torch.nn as nn
import torch
from torch_geometric.nn import TransformerConv
import inspect
from typing import Any, Dict, Optional
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential, BatchNorm1d, GELU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import normalization_resolver
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
from mamba_ssm import Mamba
from torch_geometric.utils import degree, sort_edge_index
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType
import numpy as np


class gnn_affinity_soft_attention(nn.Module):
    def __init__(self, emb_dim):
        super(gnn_affinity_soft_attention, self).__init__()
        self.emb_dim = emb_dim
        self.linear1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_alpha = nn.Linear(self.emb_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.long_middle_short = fusion_triple_feature(self.emb_dim)

    def forward(self, mask, short_feature, seq_feature, affinity_feature):
        q1 = self.linear1(short_feature).view(short_feature.size(0), 1, short_feature.size(1))
        q2 = self.linear2(seq_feature)
        q3 = self.linear3(affinity_feature).view(short_feature.size(0), 1, short_feature.size(1))
        alpha = self.sigmoid(q1 + q2 + q3)
        alpha = self.linear_alpha(alpha)
        long_feature = torch.sum(alpha * seq_feature * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.long_middle_short(long_feature, short_feature, affinity_feature)
        return seq_output


class fusion_triple_feature(nn.Module):
    def __init__(self, emb_dim):
        super(fusion_triple_feature, self).__init__()
        self.emb_dim = emb_dim
        self.linear1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.softmax = nn.Softmax(dim=1)
        self.linear_final = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, seq_hidden, pos_emb, category_seq_hidden):
        seq_hidden = seq_hidden.unsqueeze(dim=1)
        pos_emb = pos_emb.unsqueeze(dim=1)
        category_seq_hidden = category_seq_hidden.unsqueeze(dim=1)
        seq_hidden = self.linear1(seq_hidden)
        pos_emb = self.linear2(pos_emb)
        category_seq_hidden = self.linear3(category_seq_hidden)
        fusion_feature = torch.cat((seq_hidden, pos_emb, category_seq_hidden), dim=1)
        attn_weight = self.softmax(fusion_feature)
        fusion_feature = torch.sum(attn_weight * fusion_feature, dim=1)
        fusion_feature = self.linear_final(fusion_feature)
        return fusion_feature


class seq_affinity_soft_attention(nn.Module):
    def __init__(self, emb_dim):
        super(seq_affinity_soft_attention, self).__init__()
        self.emb_dim = emb_dim
        self.linear_1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_alpha = nn.Linear(self.emb_dim, 1, bias=False)
        self.long_middle_short = fusion_triple_feature(self.emb_dim)

    def forward(self, mask, short_feature, seq_feature, affinity_feature):
        q1 = self.linear_1(seq_feature)
        q2 = self.linear_2(short_feature)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q3 = self.linear_3(affinity_feature)
        q3_expand = q3.unsqueeze(1).expand_as(q1)
        alpha = self.linear_alpha(mask * torch.sigmoid(q1 + q2_expand + q3_expand))
        long_feature = torch.sum(alpha.expand_as(seq_feature) * seq_feature, 1)
        seq_output = self.long_middle_short(long_feature, short_feature, affinity_feature)
        return seq_output


class NaiveFourierKANLayer(nn.Module):
    # 参数gridsize调整范围[1, 2, 4, 8]
    def __init__(self, inputdim, outdim, gridsize):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        # Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        # This should be fused to avoid materializing memory
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)

        y = y.view(outshape)
        return y


class Bi_Mamba_block(nn.Module):
    def __init__(self, emb_dim, d_state, d_conv, expand, drop_ratio, gridsize, shared_parameter: bool = True):
        super(Bi_Mamba_block, self).__init__()
        self.emb_dim = emb_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.drop_ratio = drop_ratio
        self.gridsize = gridsize

        self.forward_mamba_block = Mamba(
            d_model=self.emb_dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand
        )
        self.backward_mamba_block = Mamba(
            d_model=self.emb_dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand
        )
        self.FourierKAN = NaiveFourierKANLayer(self.emb_dim, self.emb_dim, self.gridsize)
        self.ln1 = nn.LayerNorm(self.emb_dim, eps=1e-12)
        self.ln = nn.LayerNorm(self.emb_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=self.drop_ratio)
        if shared_parameter:  # 是否共享正向 & 反向的参数
            self.backward_mamba_block.in_proj.weight = self.forward_mamba_block.in_proj.weight
            self.backward_mamba_block.in_proj.bias = self.forward_mamba_block.in_proj.bias
            self.backward_mamba_block.out_proj.weight = self.forward_mamba_block.out_proj.weight
            self.backward_mamba_block.out_proj.bias = self.forward_mamba_block.out_proj.bias

    def forward(self, x):
        out = self.ln1(x)
        out_forward = self.forward_mamba_block(out)
        out_backward = self.backward_mamba_block(out.flip(dims=(1,)))  # 注意对输入进行flip，dim=1为seq len
        out_backward = out_backward.flip(dims=(1,))  # 再翻回来，不然还是flip后的
        out = out_forward + out_backward
        out = self.ln(out)
        out = self.dropout(out)
        out = self.FourierKAN(out)
        out = out + x
        return out


class my_MambaFormer(nn.Module):
    def __init__(self, emb_dim, layer_num, d_state, d_conv, expand, drop_prob, gridsize):
        super(my_MambaFormer, self).__init__()
        self.emb_dim = emb_dim
        self.layer_num = layer_num
        self.drop_prob = drop_prob
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.gridsize = gridsize

        self.mamba_list = nn.ModuleList(
            Bi_Mamba_block(
                emb_dim=self.emb_dim,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                drop_ratio=self.drop_prob,
                gridsize=self.gridsize
            ) for _ in range(self.layer_num))

    def forward(self, x):
        feature_list = []
        for layer in range(self.layer_num):
            x = self.mamba_list[layer](x)
            feature_list.append(x)
        x = sum(feature_list) / len(feature_list)
        return x


def permute_within_batch(x, batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()

        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices


class GraphMambaConv(nn.Module):
    def __init__(
            self,
            channels: int,
            conv: Optional[MessagePassing],
            dropout: float,
            d_state: int,
            d_conv: int,
            expand: int,
            order_by_degree: bool = False,
            shuffle_ind: int = 0,
            norm: Optional[str] = 'batch_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.dropout = dropout
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        assert (self.order_by_degree == True and self.shuffle_ind == 0) or (
                self.order_by_degree == False), f'order_by_degree={self.order_by_degree} and shuffle_ind={self.shuffle_ind}'

        self.self_attn = Mamba(
            d_model=channels,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand
        )

        # stable mlp
        self.mlp = Sequential(
            BatchNorm1d(channels),
            Linear(channels, channels * 2),
            GELU(),
            BatchNorm1d(channels * 2),
            Dropout(dropout),
            Linear(channels * 2, channels),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            batch: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        ### Global attention transformer-style model.
        if self.order_by_degree:
            deg = degree(edge_index[0], x.shape[0]).to(torch.long)
            order_tensor = torch.stack([batch, deg], 1).T
            _, x = sort_edge_index(order_tensor, edge_attr=x)

        if self.shuffle_ind == 0:
            h, mask = to_dense_batch(x, batch)
            h = self.self_attn(h)[mask]
        else:
            mamba_arr = []
            for _ in range(self.shuffle_ind):
                h_ind_perm = permute_within_batch(x, batch)
                h_i, mask = to_dense_batch(x[h_ind_perm], batch)
                h_i = self.self_attn(h_i)[mask][h_ind_perm]
                mamba_arr.append(h_i)
            h = sum(mamba_arr) / self.shuffle_ind
        ###

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    # def __repr__(self) -> str:
    #     return (f'{self.__class__.__name__}({self.channels}, '
    #             f'conv={self.conv}, heads={self.heads})')


class GNN(nn.Module):
    def __init__(self, emb_dim, gnn_layer_num, head_num, drop_prob, d_state, d_conv, expand):
        super(GNN, self).__init__()
        self.gnn_layer_num = gnn_layer_num
        self.emb_dim = emb_dim
        self.head_num = head_num
        self.drop_prob = drop_prob
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.GnnConv_list = nn.ModuleList(
            GraphMambaConv(
                emb_dim,
                conv=TransformerConv(
                    in_channels=emb_dim,
                    out_channels=emb_dim,
                    heads=self.head_num,
                    beta=True,
                    dropout=self.drop_prob,
                ),
                dropout=self.drop_prob,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand
            )
            for _ in range(self.gnn_layer_num))

    def forward(self, x, edge_index):
        gnn_feature_list = []
        for gnn_layer_num in range(self.gnn_layer_num):
            x = self.GnnConv_list[gnn_layer_num](x=x, edge_index=edge_index)
            gnn_feature_list.append(x)
        x = torch.stack(gnn_feature_list, dim=1)
        x = torch.mean(x, dim=1)
        return x


class session_infonce(nn.Module):
    def __init__(self, temperature):
        super(session_infonce, self).__init__()
        self.temperature = temperature

    def forward(self, sess4pre, sess4aux):
        raw_shuffled_index = torch.randperm(sess4pre.shape[0]).to(sess4pre.device)
        neg_session4predict = sess4pre[raw_shuffled_index]
        col_shuffled_index = torch.randperm(sess4pre.shape[0]).to(sess4pre.device)
        neg_session4predict = neg_session4predict[col_shuffled_index]
        sess4pre, sess4aux = F.normalize(sess4pre, dim=-1), F.normalize(sess4aux, dim=-1)
        pos_score = (sess4pre * sess4aux).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(neg_session4predict, sess4aux.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).sum()


class Conan(SequentialRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(Conan, self).__init__(config, dataset)
        self.drop_prob = config['drop_prob']
        self.embedding_size = config['embedding_size']
        self.device = config['device']
        self.max_seq_length = dataset.field2seqlen[self.ITEM_SEQ]
        self.mask_token = self.n_items
        self.temperature_parameter = config['temperature_parameter']
        self.gnn_layer_num = config['gnn_layer_num']
        self.gnn_head_num = config['gnn_head_num']
        self.seq_conv_list = config['seq_conv_list']
        self.seq_expand = config['seq_expand']
        self.seq_state = config["seq_state"]
        self.san_layer_num = config['san_layer_num']
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]
        self.cl_loss_weight = config['cl_loss_weight']
        self.gridsize = config['gridsize']
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.GNN = GNN(
            emb_dim=self.embedding_size,
            gnn_layer_num=self.gnn_layer_num,
            head_num=self.gnn_head_num,
            drop_prob=self.drop_prob,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand
        )
        self.my_MambaFormer = my_MambaFormer(
            emb_dim=self.embedding_size,
            layer_num=self.san_layer_num,
            drop_prob=self.drop_prob,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            gridsize=self.gridsize
        )
        self.gnn_soft_attention = gnn_affinity_soft_attention(emb_dim=self.embedding_size)
        self.time_soft_attention = seq_affinity_soft_attention(emb_dim=self.embedding_size)
        self.gnn_dropout_layer = nn.Dropout(p=self.drop_prob)
        self.loss_dropout_layer = nn.Dropout(p=self.drop_prob)
        self.ce_loss = nn.CrossEntropyLoss()
        self.nce_loss1 = session_infonce(self.temperature_parameter)

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, x, edge_index, alias_inputs, item_seq_len, item_seq):
        hidden = self.item_embedding(x)
        hidden = self.gnn_dropout_layer(hidden)
        gnn_feature = self.GNN(hidden, edge_index)
        gnn_seq = gnn_feature[alias_inputs]
        time = self.item_embedding(item_seq)
        time_seq = self.my_MambaFormer(time)
        time_seq_mask = item_seq.gt(0).unsqueeze(2).expand_as(time_seq)
        time_seq_mean = torch.mean(time_seq_mask * time_seq, dim=1)
        gnn_seq_mask = alias_inputs.gt(0)
        gnn_seq_mean = torch.mean(gnn_seq * gnn_seq_mask.view(gnn_seq_mask.size(0), -1, 1).float(), 1)
        gnn_short = self.gather_indexes(gnn_seq, item_seq_len - 1)
        gnn_session = self.gnn_soft_attention(
            mask=gnn_seq_mask,
            short_feature=gnn_short,
            seq_feature=gnn_seq,
            affinity_feature=gnn_seq_mean
        )
        time_short = self.gather_indexes(time_seq, item_seq_len - 1)
        time_session = self.time_soft_attention(
            mask=time_seq_mask,
            short_feature=time_short,
            seq_feature=time_seq,
            affinity_feature=time_seq_mean
        )
        gnn_session = F.normalize(gnn_session, dim=-1)
        time_session = F.normalize(time_session, dim=-1)
        return time_session, gnn_session

    def calculate_loss(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = interaction[self.ITEM_SEQ]
        time_session, gnn_session = self.forward(x, edge_index, alias_inputs, item_seq_len, item_seq)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.item_embedding.weight
        test_item_emb = self.loss_dropout_layer(test_item_emb)
        test_item_emb = F.normalize(test_item_emb, dim=-1)
        gnn_logits = torch.matmul(gnn_session, test_item_emb.transpose(0, 1)) / self.temperature_parameter
        gnn_ce_loss = self.ce_loss(gnn_logits, pos_items)
        cl_loss1 = self.nce_loss1(sess4pre=gnn_session, sess4aux=time_session)
        return gnn_ce_loss + self.cl_loss_weight * cl_loss1

    def predict(self, interaction):
        test_item = interaction[self.ITEM_ID]
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = interaction[self.ITEM_SEQ]
        _, gnn_session = self.forward(x, edge_index, alias_inputs, item_seq_len, item_seq)
        test_item_emb = self.item_embedding(test_item)
        test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = torch.mul(gnn_session, test_item_emb).sum(dim=1) / self.temperature_parameter  # [B]
        return scores

    def full_sort_predict(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = interaction[self.ITEM_SEQ]
        _, gnn_session = self.forward(x, edge_index, alias_inputs, item_seq_len, item_seq)
        test_items_emb = self.item_embedding.weight
        test_items_emb = F.normalize(test_items_emb, dim=-1)
        scores = torch.matmul(gnn_session, test_items_emb.transpose(0, 1)) / self.temperature_parameter
        return scores
