import torch
from torch import nn

import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair

from graphmae.utils import create_activation


class DotGAT(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 nhead_out,
                 activation,
                 feat_drop,
                 attn_drop,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False
                 ):
        super(DotGAT, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            self.gat_layers.append(DotGatConv(
                in_dim, out_dim, nhead_out,
                feat_drop, attn_drop, last_residual, norm=last_norm, concat_out=concat_out))
        else:
            # input projection (no residual)
            self.gat_layers.append(DotGatConv(
                in_dim, num_hidden, nhead,
                feat_drop, attn_drop, residual, create_activation(activation), norm=norm, concat_out=concat_out))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(DotGatConv(
                    num_hidden * nhead, num_hidden, nhead,
                    feat_drop, attn_drop, residual, create_activation(activation), norm=norm, concat_out=concat_out))
            # output projection
            self.gat_layers.append(DotGatConv(
                num_hidden * nhead, out_dim, nhead_out,
                feat_drop, attn_drop, last_residual, activation=last_activation, norm=last_norm, concat_out=concat_out))
    
        self.head = nn.Identity()
    
    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h)
            hidden_list.append(h)
            # h = h.flatten(1)
        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)


class DotGatConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 residual,
                 activation=None,
                 norm=None,
                 concat_out=False,
                 allow_zero_in_degree=False):
        super(DotGatConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads
        self._concat_out = concat_out

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.activation = activation

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, self._out_feats*self._num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, self._out_feats*self._num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, self._out_feats*self._num_heads, bias=False)

        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        self.norm = norm
        if norm is not None:
            self.norm = norm(num_heads * out_feats)

    def forward(self, graph, feat, get_attention=False):
        r"""

        Description
        -----------
        Apply dot product version of self attention in GCN.

        Parameters
        ----------
        graph: DGLGraph or bi_partities graph
            The graph
        feat: torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}` is size
            of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """

        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise ValueError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        # check if feat is a tuple
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            print("!! tuple input in DotGAT !!")
        else:
            feat = self.feat_drop(feat)
            h_src = feat
            feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]

        # Assign features to nodes
        graph.srcdata.update({'ft': feat_src})
        graph.dstdata.update({'ft': feat_dst})

        # Step 1. dot product
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))

        # Step 2. edge softmax to compute attention scores
        graph.edata['sa'] = edge_softmax(graph, graph.edata['a'] / self._out_feats**0.5)
        graph.edata["sa"] = self.attn_drop(graph.edata["sa"])
        # Step 3. Broadcast softmax value to each edge, and aggregate dst node
        graph.update_all(fn.u_mul_e('ft', 'sa', 'attn'), fn.sum('attn', 'agg_u'))

        # output results to the destination nodes
        rst = graph.dstdata['agg_u']

        if self.res_fc is not None:
            # Use -1 rather than self._num_heads to handle broadcasting
            batch_size = feat.shape[0]
            resval = self.res_fc(h_dst).view(batch_size, -1, self._out_feats)
            rst = rst + resval

        if self._concat_out:
            rst = rst.flatten(1)
        else:
            rst = torch.mean(rst, dim=1)

        if self.norm is not None:
            rst = self.norm(rst)

        # activation
        if self.activation:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst
