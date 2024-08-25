import torch
from torch_geometric.nn import MLP,GCNConv,GATv2Conv, GINConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gmp, global_max_pool as gmp, global_add_pool as gap
import torch.nn.functional as F
from layers import SKipPool as SKipPool

class Net(torch.nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.tot_epochs = args.epochs
        self.seed = args.seed
        self.fixStride = args.fixStride
        p_list = args.p_list
        if p_list is None:
            p_list=[1,1,1]

        self.conv1 = GCNConv(self.num_features, self.nhid)
        
        self.pool1 = SKipPool(self.nhid,last_ratio=self.pooling_ratio,
                              tot_epochs= self.tot_epochs,scorer=args.prePool,fixStride=self.fixStride,num_process=p_list[0],seed=self.seed)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        self.pool2 = SKipPool(self.nhid,last_ratio=self.pooling_ratio, 
                              tot_epochs= self.tot_epochs,scorer=args.prePool,fixStride=self.fixStride,num_process=p_list[1],seed=self.seed)
        
        self.conv3 = GCNConv(self.nhid, self.nhid)
                
        self.pool3 = SKipPool(self.nhid,last_ratio=self.pooling_ratio,
                              tot_epochs= self.tot_epochs,scorer=args.prePool,fixStride=self.fixStride,num_process=p_list[2],seed=self.seed)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

    def forward(self, data, epoch, isTest=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, epoch, None, batch, isTest)

        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, epoch, None, batch, isTest)
        
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, epoch,None, batch, isTest)
        
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x