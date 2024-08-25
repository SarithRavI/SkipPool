from torch_geometric.nn import Linear, GCNConv
# instead of importing topkpool or sagpool we implement them here
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import init, ReLU, Parameter, Dropout, Identity, GELU
from torch.nn import functional as F
from torch.linalg import vector_norm
import torch
from pooling.utils import unfold1d,PresetUnfold
import numpy as np
# from torch_geometric.utils import scatter,softmax
from torch_scatter import scatter, scatter_min,scatter_max
from torch_scatter.composite import scatter_softmax

class SKipPool(torch.nn.Module):
    def __init__(self,in_channels,init_ratio,last_ratio, tot_epochs,scorer:str='top-k',
                 non_linearity=torch.tanh,tauA=10,tauB=0.1, # 10,0.1
                 min_num_c =8, fixStride:int = None,seed:int=777,aggr_type='agg_scale'):
        
        super(SKipPool,self).__init__()
        self.in_channels = in_channels
        self.init_ratio = init_ratio
        self.last_ratio = last_ratio
        self.tot_epochs = tot_epochs
        self.tauA = tauA
        self.tauB = tauB
        if scorer == 'top-k':
            self.scorer = TopKPool(in_channels,init_ratio)
        elif scorer == "sag":
            self.scorer = SAGPool(in_channels,init_ratio)    
        self.non_linearity = non_linearity

        if init_ratio>=0.999:
            self.init_ratio = 1.0

        self.sim_transform= Linear(in_channels,in_channels)
        init.xavier_normal_(self.sim_transform.weight)
        self.att_params = Parameter(torch.randn(in_channels*2,1))

        self.relu = ReLU()

        self.inter_act = ReLU()
        
        self.min_num_c  = min_num_c
        # PROTEINS (4)  
        # DD (64) -> rep_0 (777) -> 78.34, rep_1 (8971) -> 76.324, rep_2 (9467) -> 76.4, rep_3 (32830) -> 78.603
        self.fs_ratio = 1/self.min_num_c
        self.eps = 1e-20
        self.fixStride = fixStride
        self.seed = seed

        self.aggr_type = aggr_type
        # self.fullStrideStrt = fixStride

    def getGKM_ei(self,x,edge_index,sigma=1):
        srcs,dests = edge_index
        distances = torch.norm(x[srcs.to(torch.long)]- x[dests.to(torch.long)], p=2, dim=-1)
        kmat = torch.exp(-distances/(2*sigma**2))
        return  kmat-(1e-5)
    
    def getGKM_with_SM_ei(self,sm_atts,dist_rowwise_inx):

        return scatter_softmax(sm_atts,dist_rowwise_inx)

    def getSM_ei(self,x,edge_index):
        srcs,dests = edge_index
        # x_ = self.sim_transform(x)
        x_ = x 
        atts = (torch.hstack((x_[srcs.to(torch.long)],x[dests.to(torch.long)])) @ self.att_params).view(-1)
        return atts

    def scatterSort(self,x,batch):
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
        batch_size = num_nodes.size(0)
        # max_num_nodes = max(3,int(num_nodes.max()))
        max_num_nodes = int(num_nodes.max())

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
                num_nodes.cumsum(dim=0)[:-1]], dim=0)
        inx_cumsum = cum_num_nodes.gather(-1,batch)
        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + ( batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ), -60000.0)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)
        srt_x, inx_ = dense_x.sort(stable=True,dim=-1, descending=True)
        
        return (srt_x.view(-1), 
            inx_.view(-1)+inx_cumsum,
            None)
        
        # if not calSortOps and not prune_inx:
        #     return (srt_x.view(-1), 
        #         inx_.view(-1)+inx_cumsum,
        #         None)
        # elif calSortOps and not prune_inx:
        #     return (srt_x.view(-1), 
        #             inx_.view(-1)+inx_cumsum,
        #             inx_)
        # elif prune_inx:
        #     return (srt_x.view(-1)[prune_inx.to(torch.int64)], 
        #             inx_.view(-1)[prune_inx.to(torch.int64)]+inx_cumsum,
        #             inx_)
        # return (srt_x.view(-1)[prune_inx.to(torch.int64)], 
        #         inx_.view(-1)[prune_inx.to(torch.int64)]+inx_cumsum,
        #         None)

    
    def forward(self, x, edge_index, epoch, edge_attr=None, batch=None,isHard=False):
        pre_device = 'cpu' # cuda:0
        n = x.size(0)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        _, batch_size_n = torch.unique(batch,return_counts=True)
        
        tau0 = self.tauA*((self.tauB/self.tauA)**(epoch/self.tot_epochs))
        
        if isinstance(self.scorer,TopKPool):
            x, edge_index, edge_attr, batch, scores, perm = self.scorer(x, edge_index,batch=batch)
        elif isinstance(self.scorer,SAGPool):
            x, edge_index, edge_attr, batch, scores, perm = self.scorer(x,edge_index,batch=batch)
        # print('here 3')
        
        scores_ranked = self.non_linearity(scores[perm]).view(-1, 1)

        zero_tensor = torch.tensor([0],device=pre_device)
        zero_tensor_c = torch.tensor([0],device='cuda:0')
        
        # get the total num of batches 
        # _, batch_size = torch.unique(batch,return_counts=True)
        batch_size = scatter(torch.ones(batch.size(0),device='cuda:0',dtype=torch.int32),batch,reduce='sum')
        batch_size = batch_size.to(device=pre_device)
        
        batch_cumsum = torch.cumsum(torch.cat((zero_tensor,batch_size))
                                    ,dim=0,dtype=torch.int64).to(device=pre_device)

        dests_all = []
        srcs_all = []
        # use for pruning instead of looping inside scatterSort
        new_prune_inx = []
        # use this to reduce the dist arr along cols
        dist_colwise_inx = []
        # use this for sorting the dist arr
        dist_rowwise_inx = []

        _dist_rowwise_inx_ = []

        new_rowwise_cumsum = torch.tensor([0],dtype=torch.int32,device=pre_device)
        new_colwise_cumsum = torch.tensor([0],dtype=torch.int32,device=pre_device)
        
        rows_inx = []
        # we can get this by considering the batch sizes

        fin_col_sizes = torch.ceil(batch_size*(self.last_ratio/self.init_ratio)).to(torch.int32) # ceil 
        
        # get inx where fin_col_sizes >= 2 
        skipping_g_inx = torch.where(fin_col_sizes >= 2)[0].tolist()

        # get inx where fin_col_sizes < 2 
        avoidSkip_g_inx = torch.where(fin_col_sizes < 2)[0].tolist()
        

        max_rSize = self.fixStride if self.fixStride else torch.max(torch.tensor([3],device=pre_device),torch.floor(batch_size*self.fs_ratio).max())

    
        unskip_pivots_in_x = batch_cumsum[avoidSkip_g_inx].cuda()
        unskip_x = x[unskip_pivots_in_x]
        
        FORCED_MASK_SIZE = 1 # check 6
        FORCED_MASK_INX = -3 #-1,-2 will be used for other purposes

        for inx in skipping_g_inx:
            col_size = batch_size[inx]
            # print(col_size)
            curr_row_bcs = new_rowwise_cumsum[-1]
            curr_col_bcs = new_colwise_cumsum[-1]
            curr_bsc = batch_cumsum[inx]

            col_arr = torch.arange(col_size,dtype=torch.int32,device=pre_device)
        
            # res & fs are sizes
            fs = self.fixStride if self.fixStride else max(3,int(col_size*self.fs_ratio))
            res = (col_size-fs)+1
            
            dests = unfold1d(col_arr,fs,1).to(torch.int32)
            dests_frow=dests[0]
            dests_all.append(dests.flatten()+ curr_bsc)
            
            # srcs = dests - torch.arange(fs)
            srcs = PresetUnfold.UNFOLDED_SRCS[:res,:fs].to(torch.int32)

            srcs_fcol = srcs[:,0].flatten()
            srcs_fcol_row_bcs = srcs_fcol+curr_row_bcs
            rows_inx.append(srcs_fcol_row_bcs)
            # new_batch_inx = torch.cat((new_batch_inx,srcs_fcol+curr_bsc))
            srcs_all.append(srcs.flatten()+curr_bsc)
            
            dist_rowwise_inx.append(srcs.flatten()+curr_row_bcs)

            # masked_srcs = srcs
            # masked_srcs[:,:FORCED_MASK_SIZE] = FORCED_MASK_INX-curr_row_bcs
            # _dist_rowwise_inx_.append(masked_srcs.flatten()+curr_row_bcs)

            _dist_rowwise_inx_.append(srcs.flatten()+curr_row_bcs)
            
            dests_firstR_tile = dests_frow.tile((res,))
            
            # prunes = (dests_firstR_tile.view(res,fs)+ (srcs_fcol_row_bcs.view(-1,1)*max_rSize))
            # new_prune_inx.append(prunes.flatten())
            
            # print(f'fs: {fs}, maxR: {max_rSize}')
            # print(inx, prunes,'\n')
            
            dist_colwise_inx.append(dests_firstR_tile + curr_col_bcs)

            new_rowwise_cumsum = torch.hstack((new_rowwise_cumsum,
                                                torch.tensor([res],device=pre_device)+curr_row_bcs))
            new_colwise_cumsum = torch.hstack((new_colwise_cumsum,
                                                torch.tensor([fs],device=pre_device)+curr_col_bcs))   
            # print(f'returning: {col_size}')     
        
        dests_all = torch.cat(dests_all).to(torch.int32).cuda()
        srcs_all = torch.cat(srcs_all).to(torch.int32).cuda()
        dist_colwise_inx = torch.cat(dist_colwise_inx).to(torch.int64).cuda()
        dist_rowwise_inx = torch.cat(dist_rowwise_inx).to(torch.int64).cuda()
        new_prune_inx = None    # torch.cat(new_prune_inx).to(torch.int32).cuda()
        rows_inx = torch.cat(rows_inx).to(torch.int64).cuda()

        _dist_rowwise_inx_ = torch.cat(_dist_rowwise_inx_).to(torch.int64).cuda()

        rowwise_count = scatter(torch.ones(dist_rowwise_inx.size(0),dtype=torch.int64,device='cuda:0'),
                                dist_rowwise_inx)
                
        rowwise_count_cumsum_ = torch.cat((zero_tensor_c,rowwise_count.cumsum(dim=-1)))
        rowwise_count_cumsum = rowwise_count_cumsum_[:-1]

        # very important
        one_places_bool = torch.full((dist_rowwise_inx.size(0),),False).cuda()
        one_places_bool[rowwise_count_cumsum] = True
                
        sm_atts = self.getSM_ei(x,(srcs_all,dests_all))
        
        # softmax sm_atts 
        sm_atts = scatter_softmax(sm_atts,dist_rowwise_inx)

        gk_dists = self.getGKM_ei(x,(srcs_all,dests_all))

        gk_dists_sort, sc_sort_inx, _ = self.scatterSort(gk_dists,dist_rowwise_inx)
        sm_atts_sort = sm_atts[sc_sort_inx]

        sim_arr = sm_atts_sort

        logits_ = scatter(sim_arr,dist_colwise_inx,reduce='sum')
        logits_all = logits_.gather(-1,dist_colwise_inx)
        
        logits_all = torch.where(one_places_bool,float('-inf'),logits_all)

        if self.seed:
            gumbels_all = (-torch.empty_like(logits_all, memory_format=torch.legacy_contiguous_format)
                        .exponential_(generator=torch.Generator(device='cuda:0').manual_seed(self.seed))
                        .log()) 
        else:
            gumbels_all = (-torch.empty_like(logits_all, memory_format=torch.legacy_contiguous_format).exponential_().log())

        logits_sample_sum = (gumbels_all+logits_all)/tau0
        # logits_sample_sum = logits_all/tau0

        gmbSoftmax_samples = scatter_softmax(logits_sample_sum,dist_rowwise_inx)
        

        logit_perrow = gk_dists_sort*gmbSoftmax_samples

        ths = scatter(logit_perrow,dist_rowwise_inx,reduce='sum').gather(-1,dist_rowwise_inx)#[dist_rowwise_inx]

        shifted_rows = self.inter_act(gk_dists - ths)  
        _dist_rowwise_inx_[rowwise_count_cumsum] = rows_inx[-1]+1

        with torch.no_grad():
            _,argmin = scatter_min(shifted_rows,_dist_rowwise_inx_) 
        argmin = argmin[:-1]
        argmin = argmin - rowwise_count_cumsum
       
        rows_argmin_added = rows_inx+argmin

        pivots = []
        pivots_in_x =[]

        new_rowwise_cumsum = new_rowwise_cumsum.to(torch.int64).cpu()
        fin_col_sizes = fin_col_sizes.cpu()
        rows_argmin_added = rows_argmin_added.cpu()

        for i,gi in enumerate(skipping_g_inx):
            rows_start, rows_end = new_rowwise_cumsum[i],new_rowwise_cumsum[i+1] 
            start_res = batch_cumsum[gi]-rows_start
            new_pivot = rows_start
            pivots_in_g =[]
            max_pivot_count =fin_col_sizes[gi]
            
            while len(pivots_in_g) < max_pivot_count and new_pivot < rows_end:
                pivots_in_g.append(new_pivot)
                new_pivot = rows_argmin_added[new_pivot]
            
            pivots_in_g= torch.tensor(pivots_in_g)
            pivots.append(pivots_in_g) 
            pivots_in_x.append(pivots_in_g+start_res)   
        
        pivots = torch.cat(pivots).cuda()
        pivots_in_x = torch.cat(pivots_in_x).cuda()
        
        att_w = shifted_rows
        # normalize att_w
        att_w_norm = scatter(att_w,dist_rowwise_inx,reduce='sum').gather(-1,dist_rowwise_inx)
        att_w = att_w / att_w_norm

        # === SRC ===
        num_pivots = pivots.size(0)
        s_att = att_w.view(att_w.size(0)//self.fixStride,self.fixStride)[pivots,:]
        s_dests = dests_all.view(dests_all.size(0)//self.fixStride,self.fixStride)[pivots,:] + torch.arange(start=0,end=num_pivots*x.size(0),step=x.size(0),
                                                                                                            dtype=dests_all.dtype,
                                                                                                            device=dests_all.device).view(-1,1)
        s = torch.zeros(num_pivots*x.size(0),dtype=s_att.dtype,device=s_att.device)
        s[s_dests.flatten()] = s_att.flatten()
        # S is in shape of K*N
        s = s.reshape(num_pivots,x.size(0))
        
        new_scores = scores_ranked[pivots_in_x]

        if self.aggr_type == 'aggr_scale':
            s = s*new_scores.view(-1,1) 
        elif self.aggr_type =='aggr':
            s = s
        # calculate skipped_x
        new_x = s @ x
        # N dim must be permuted to original order..
        s = s[:,perm.argsort(0)]
        # S is in shape of N*K
        s = s.T
        
        new_batch = batch[pivots_in_x]

        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, pivots_in_x, num_nodes=scores_ranked.size(0))
       
        return new_x, edge_index, edge_attr, new_batch, s

# only retrieves x,batch,perm
class TopKPool(torch.nn.Module):
    def __init__(self,in_channels,init_ratio,p=0.0):
        super(TopKPool,self).__init__()
        self.in_channels = in_channels
        self.init_ratio = init_ratio
        self.score_layer = Linear(in_channels, 1)
        self.drop = Dropout(p=p) if p > 0 else Identity()

    def forward(self, x,edge_index, edge_attr=None, batch=None):
        x = self.drop(x)
        if batch is None:
            batch = edge_index.new_zeros(x.size(0)) 
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        scores = self.score_layer(x).squeeze()/vector_norm(self.score_layer.weight)
        scores = scores.unsqueeze(-1) if scores.dim() == 0 else scores
        perm = topk(scores, self.init_ratio, batch)
        # print(f'scores min: {scores.min()}')
        # NOTE: if perm gives out of index error 
        # check if scores.min() < -60000.0
        # check topk_pool.py line 37
        x = x[perm]
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
                edge_index, edge_attr, perm, num_nodes=scores.size(0))

        return x, edge_index, edge_attr, batch, scores, perm


class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,init_ratio,Conv=GCNConv):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.init_ratio = init_ratio
        self.score_layer = Conv(in_channels,1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x   
        scores = self.score_layer(x,edge_index).squeeze()
        scores = scores.unsqueeze(-1) if scores.dim() == 0 else scores
        perm = topk(scores, self.init_ratio, batch)
        
        x = x[perm]
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=scores.size(0))

        return x, edge_index, edge_attr, batch, scores, perm
    
class SAGPool_(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh,p=0.0):
        super(SAGPool_,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        self.score_layer = Conv(in_channels,1)
        # self.score_layer = Linear(in_channels, 1)

        self.non_linearity = non_linearity
        self.drop = Dropout(p=p) if p > 0 else Identity()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.drop(x)
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        score = self.score_layer(x,edge_index).squeeze()
        # score = self.score_layer(x).squeeze()/vector_norm(self.score_layer.weight)

        score = score.unsqueeze(-1) if score.dim() == 0 else score
        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch, perm