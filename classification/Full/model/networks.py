import torch
from torch_geometric.nn import MLP,GCNConv,GATv2Conv, GINConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gmp, global_max_pool as gmp, global_add_pool as gap
import torch.nn.functional as F
from layers import SKipPool_Full
from layers_padded import SKipPool_Full as SkipPool_padded

class Net(torch.nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features or 1
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.tot_epochs = args.epochs
        self.seed = args.seed
        self.fixStride = args.fixStride
        self.edge_dim = self.args.edge_dim or 1
        p_list = args.p_list
        if p_list is None:
            p_list=[1,1]

        if args.conv == 'gcn':
            Conv = GCNConv
        elif args.conv == 'gin':
            Conv = GINConv
        elif args.conv == 'gatv2':
            Conv = GATv2Conv
        if args.conv == 'gcn':
            self.conv1 = Conv(self.num_features, self.nhid)
            self.conv2 = Conv(self.nhid, self.nhid*2)
            self.conv3 = Conv(self.nhid*2, self.nhid*4)
        elif args.conv == 'gatv2':
            self.conv1 = Conv(self.num_features, self.nhid,edge_dim=self.edge_dim)
            self.conv2 = Conv(self.nhid, self.nhid*2,edge_dim=self.edge_dim)
            self.conv3 = Conv(self.nhid*2, self.nhid*4,edge_dim=self.edge_dim)
        elif args.conv == 'gin':
            # for mlp struct see, 
            # https://github.com/flandolfi/k-mis-pool/blob/master/benchmark/models.py#L60
            mlp_0 = MLP([self.num_features, self.num_features, self.nhid], act=F.relu)
            self.conv1 = Conv(nn=mlp_0, train_eps=False)
            mlp_1 = MLP([self.nhid, self.nhid, self.nhid*2], act=F.relu)
            self.conv2 = Conv(nn=mlp_1, train_eps=False)
            mlp_2 = MLP([self.nhid*2, self.nhid*2, self.nhid*4], act=F.relu)
            self.conv3 = Conv(nn=mlp_2, train_eps=False)
        
        if args.addPadding:
            SKipPool = SkipPool_padded
        else:
            SKipPool = SKipPool_Full
        
        self.pool1 = SKipPool(self.nhid,last_ratio=self.pooling_ratio,
                              tot_epochs= self.tot_epochs,scorer=args.prePool,fixStride=self.fixStride,num_process=p_list[0],seed=self.seed)

        self.pool2 = SKipPool(self.nhid*2,last_ratio=self.pooling_ratio, 
                              tot_epochs= self.tot_epochs,scorer=args.prePool,fixStride=self.fixStride,num_process=p_list[1],seed=self.seed)
        
        # self.lin1 = torch.nn.Linear(self.nhid*8, self.nhid*2)
        # self.lin2 = torch.nn.Linear(self.nhid*2, self. num_classes)
        # https://github.com/flandolfi/k-mis-pool/blob/master/benchmark/models.py#L105
        self.mlp =  MLP([self.nhid*8, self.nhid, self.num_classes],
                           batch_norm=False, dropout=0.3)

    def forward(self, data, epoch, isTest=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, epoch, None, batch, isTest)
        
        x = F.relu(self.conv2(x, edge_index))
        
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, epoch, None, batch, isTest)

        x = F.relu(self.conv3(x, edge_index))
                
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.log_softmax(self.mlp(x), dim=-1)

        return x