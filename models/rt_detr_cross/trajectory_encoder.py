import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else _get_activation_fn(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # self.query_pos_head = MLP(encoder_layer.d_model, encoder_layer.d_model, encoder_layer.d_model, num_layers=2)
        self.query_pos_head = MLP(4, 2 * encoder_layer.d_model, encoder_layer.d_model, num_layers=2)

        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)

    def forward(self, instance):
        output = instance.history_output.transpose(0, 1)
        src_key_padding_mask = instance.mask
        history_embedding = instance.history_embedding.transpose(0, 1)

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=instance.query_pos_embed.transpose(0, 1),
                           history_embedding=history_embedding)

        # if self.norm is not None:
        #     output = self.norm(output)

        if len(output.shape) != 3:
            output = output.unsqueeze(dim=1)
        else:
            pass
        instance.history_embedding = output
        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #self.self_attn = LinearAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.linear_feat1 = nn.Linear(d_model, dim_feedforward)
        self.linear_feat2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_key_padding_mask, pos, history_embedding):
        q = v = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, v, value=history_embedding, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # src2 = self.self_attn(src, k, value=feat, key_padding_mask=None)[0]

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # 融合过往, 不太行过
        # query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(src))))
        # query_feat = history_embedding + self.dropout_feat2(query_feat2)
        # query_feat = self.norm_feat(query_feat)

        src = src.transpose(0, 1)
        return src


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_trajectory_encoder(args):
    depth_encoder_layer = TransformerEncoderLayer(
        args.hidden_dim, nhead=args.nheads, dim_feedforward=args.dim_feedforward, dropout=0.1)

    return TransformerEncoder(depth_encoder_layer, 1)
