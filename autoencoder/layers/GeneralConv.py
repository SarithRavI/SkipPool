import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import add_self_loops, degree

# GeneralConv implementation according to the `Spektral` library:
# https://graphneural.network/layers/convolution/#generalconv

class CustomGeneralConv(MessagePassing):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 batch_norm=True,
                 dropout=0.0,
                 aggregate="add",
                 activation="ReLU",
                 use_bias=True):
        

        super().__init__(aggr=aggregate)
        self.use_bias = use_bias
        self.use_bn = batch_norm
        self.lin = Linear(in_channels, out_channels, bias=False)
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.bias = Parameter(torch.empty(out_channels)) if use_bias else None
        torch.nn.init.zeros_(self.bias)
        self.bn = BatchNorm(out_channels) if batch_norm else None
        self.dropout = dropout
        self.activation = getattr(torch.nn,activation)()

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        if self.use_bias:
            x += self.bias
        if self.use_bn:
            x = self.bn(x)
        x = F.dropout(x,self.dropout)
        x = self.activation(x)

        out = self.propagate(edge_index, x=x)

        return out
