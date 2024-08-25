import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch_geometric.nn.norm import BatchNorm

class CustomMLP(torch.nn.Module):        
    def __init__(self, in_channels,
                 hidden_channels,
                 dropout=0.0,
                 batch_norm=True,
                 activation ='ReLU',
                 final_activation = None):
        super().__init__()
        self.dropout = dropout
        self.mlp = MLP(in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,
                    num_layers=2,
                    dropout=self.dropout,
                    act=getattr(torch.nn,activation)(),
                    norm='batch_norm')
        self.use_bn_last = batch_norm
        self.mlp_last_norm = BatchNorm(hidden_channels) if batch_norm else None
        self.fin_act = getattr(torch.nn,final_activation)() if final_activation else final_activation
    
    def forward(self,x):
        x = self.mlp(x)
        if self.use_bn_last:
            x = self.mlp_last_norm(x)
        x = F.dropout(input=x, p=self.dropout)
        if self.fin_act is not None:
            x = self.fin_act(x)
        return x 
