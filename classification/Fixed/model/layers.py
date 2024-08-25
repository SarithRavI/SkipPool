import torch
from torch.multiprocessing import Pool
from torch.nn import ReLU, Parameter, Dropout, Identity
from torch.linalg import vector_norm
from torch_geometric.nn import Linear, GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_scatter import scatter, scatter_min
from torch_scatter.composite import scatter_softmax

from utils import unfold1d,PresetUnfold

def innerLoopStar(rows_start, rows_end,start_res,max_pivot_count,rows_argmin_added):
    new_pivot = rows_start
    pivots_in_g =[]
    
    while len(pivots_in_g) < max_pivot_count and new_pivot < rows_end:
        pivots_in_g.append(new_pivot)
        new_pivot = rows_argmin_added[new_pivot-rows_start]
    
    pivots_in_g= torch.tensor(pivots_in_g) 

    return pivots_in_g,pivots_in_g+start_res 

class SKipPool(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 last_ratio,
                 tot_epochs,
                 scorer:str='top-k',
                 non_linearity=torch.tanh,
                 tauA=10,
                 tauB=0.1,
                 fixStride:int = None,
                 num_process = 2,
                 seed:int=777,):
        
        super(SKipPool,self).__init__()
        self.in_channels = in_channels
        self.init_ratio = 0.99999
        self.last_ratio = last_ratio
        self.tot_epochs = tot_epochs
        self.tauA = tauA
        self.tauB = tauB
        if scorer == 'top-k':
            self.scorer = TopKPool(in_channels,self.init_ratio)
        elif scorer == "sag":
            self.scorer = SAGPool(in_channels,self.init_ratio)    
        self.non_linearity = non_linearity

        if self.init_ratio>=0.999:
            self.init_ratio = 1.0

        self.att_params = Parameter(torch.randn(in_channels*2,1))

        self.relu = ReLU()

        self.inter_act = ReLU()
        
        self.eps = 1e-20
        self.fixStride = fixStride
        self.seed = seed

        self.p = Pool(num_process)       

    def getDM_ei(self,x,edge_index,sigma=1):
        srcs,dests = edge_index
        distances = torch.norm(x[srcs.to(torch.long)]- x[dests.to(torch.long)], p=2, dim=-1)
        kmat = torch.exp(-distances/(2*sigma**2))
        return  kmat-self.eps
    
    def getSM_ei(self,x,edge_index):
        srcs,dests = edge_index
        x_ = x 
        atts = (torch.hstack((x_[srcs.to(torch.long)],x[dests.to(torch.long)])) @ self.att_params).view(-1)
        return atts

    def scatterSort(self,x,batch):
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
        batch_size = num_nodes.size(0)
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
        
    
    def forward(self, x, edge_index, epoch, edge_attr=None, batch=None,isHard=False):
        pre_device = 'cpu'

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
                
        tau0 = self.tauA*((self.tauB/self.tauA)**(epoch/self.tot_epochs))
        
        if isinstance(self.scorer,TopKPool):
            x, edge_index, edge_attr, batch, scores, perm = self.scorer(x, edge_index,batch=batch)
        elif isinstance(self.scorer,SAGPool):
            x, edge_index, edge_attr, batch, scores, perm = self.scorer(x,edge_index,batch=batch)
        
        scores_ranked = self.non_linearity(scores[perm]).view(-1, 1)
        zero_tensor = torch.tensor([0],device=pre_device)
        zero_tensor_c = torch.tensor([0],device='cuda:0')

        batch_size = scatter(torch.ones(batch.size(0),device='cuda:0',dtype=torch.int32),batch,reduce='sum')
        batch_size = batch_size.to(device=pre_device)
        
        batch_cumsum = torch.cumsum(torch.cat((zero_tensor,batch_size))
                                    ,dim=0,dtype=torch.int64).to(device=pre_device)

        dests_all = []
        srcs_all = []
        dist_colwise_inx = []
        dist_rowwise_inx = []
        _dist_rowwise_inx_ = []
        new_rowwise_cumsum = torch.tensor([0],dtype=torch.int32,device=pre_device)
        new_colwise_cumsum = torch.tensor([0],dtype=torch.int32,device=pre_device)
        
        rows_inx = []

        fin_col_sizes = torch.ceil(batch_size*(self.last_ratio/self.init_ratio)).to(torch.int32)
        
        # get inx where fin_col_sizes >= 2 
        skipping_g_inx = torch.where(fin_col_sizes >= 2)[0].tolist()
        # get inx where fin_col_sizes < 2 
        avoidSkip_g_inx = torch.where(fin_col_sizes < 2)[0].tolist()
        
        unskip_pivots_in_x = batch_cumsum[avoidSkip_g_inx].cuda()
        unskip_x = x[unskip_pivots_in_x]
        
        # if len(skipping_g_inx) > 0:
        fs = self.fixStride 
        for inx in skipping_g_inx:
            col_size = batch_size[inx]
            curr_row_bcs = new_rowwise_cumsum[-1]
            curr_col_bcs = new_colwise_cumsum[-1]
            curr_bsc = batch_cumsum[inx]

            col_arr = torch.arange(col_size,dtype=torch.int64,device=pre_device)
        
            res = (col_size-fs)+1
            
            dests = unfold1d(col_arr,fs,1)
            dests_frow=dests[0]
            dests_all.append(dests.flatten()+ curr_bsc)
            
            srcs = PresetUnfold.UNFOLDED_SRCS[:res,:fs]

            srcs_fcol = srcs[:,0].flatten()
            srcs_fcol_row_bcs = srcs_fcol+curr_row_bcs
            rows_inx.append(srcs_fcol_row_bcs)
            srcs_all.append(srcs.flatten()+curr_bsc)
            
            dist_rowwise_inx.append(srcs.flatten()+curr_row_bcs)

            _dist_rowwise_inx_.append(srcs.flatten()+curr_row_bcs)
            
            dests_firstR_tile = dests_frow.tile((res,))
            
            dist_colwise_inx.append(dests_firstR_tile + curr_col_bcs)

            new_rowwise_cumsum = torch.hstack((new_rowwise_cumsum,
                                                torch.tensor([res],device=pre_device)+curr_row_bcs))
            new_colwise_cumsum = torch.hstack((new_colwise_cumsum,
                                                torch.tensor([fs],device=pre_device)+curr_col_bcs))   
        
        dests_all = torch.cat(dests_all).cuda()
        srcs_all = torch.cat(srcs_all).cuda()
        dist_colwise_inx = torch.cat(dist_colwise_inx).cuda()
        dist_rowwise_inx = torch.cat(dist_rowwise_inx).cuda()
        rows_inx = torch.cat(rows_inx).to(torch.int64).cuda()
        _dist_rowwise_inx_ = dist_rowwise_inx.clone()

        rowwise_count = scatter(torch.ones(dist_rowwise_inx.size(0),dtype=torch.int64,device='cuda:0'),
                                dist_rowwise_inx)
        
        rowwise_count_cumsum_ = torch.cat((zero_tensor_c,rowwise_count.cumsum(dim=-1)))
        rowwise_count_cumsum = rowwise_count_cumsum_[:-1]

        sm_atts = self.getSM_ei(x,(srcs_all,dests_all))
        
        # softmax sm_atts 
        sm_atts = scatter_softmax(sm_atts,dist_rowwise_inx)

        gk_dists = self.getDM_ei(x,(srcs_all,dests_all))
        gk_dists_sort, sc_sort_inx, _ = self.scatterSort(gk_dists,dist_rowwise_inx)
        sm_atts_sort = sm_atts[sc_sort_inx]

        sim_arr = sm_atts_sort

        logits_ = scatter(sim_arr,dist_colwise_inx,reduce='sum')
        logits_all = logits_.gather(-1,dist_colwise_inx)
        
        # mask the first inx of every row
        logits_all[rowwise_count_cumsum] = torch.tensor(float('-inf'),dtype=torch.float32)

        gumbels_all = (-torch.empty_like(logits_all, memory_format=torch.legacy_contiguous_format)
                    .exponential_(generator=torch.Generator(device='cuda:0').manual_seed(self.seed))
                    .log()) 

        logits_sample_sum = (gumbels_all+logits_all)/tau0

        gmbSoftmax_samples = scatter_softmax(logits_sample_sum,dist_rowwise_inx)


        logit_perrow = gk_dists_sort*gmbSoftmax_samples

        ths = scatter(logit_perrow,dist_rowwise_inx,reduce='sum').gather(-1,dist_rowwise_inx)

        shifted_rows = self.inter_act(gk_dists - ths) 

        _dist_rowwise_inx_[rowwise_count_cumsum] = rows_inx[-1]+1

        # with torch.no_grad():
        _,argmin = scatter_min(shifted_rows,_dist_rowwise_inx_)
        argmin = argmin[:-1]
        argmin = argmin - rowwise_count_cumsum
                        
        rows_argmin_added = rows_inx+argmin

        # pivots in rowwise rep 
        pivots = []
        pivots_in_x =[]

        new_rowwise_cumsum = new_rowwise_cumsum.to(torch.int64).cpu()
        fin_col_sizes = fin_col_sizes.cpu()
        rows_argmin_added = rows_argmin_added.cpu()

        rows_start_all, rows_end_all = new_rowwise_cumsum[:-1], new_rowwise_cumsum[1:].tolist()
        start_res_all = batch_cumsum[skipping_g_inx].cpu()-rows_start_all
        max_pivot_counts = fin_col_sizes[skipping_g_inx]
        rows_argmin_added_chunks = []
        rows_start_all = rows_start_all.tolist()

        for l in  range(len(rows_start_all)):
            rows_argmin_added_chunks.append(rows_argmin_added[rows_start_all[l]:rows_end_all[l]].tolist())
        
        star_ls = list(zip(rows_start_all,
                    rows_end_all,
                    start_res_all.tolist(),
                    max_pivot_counts.tolist(),
                    rows_argmin_added_chunks))
        
        res = self.p.starmap(innerLoopStar,star_ls)

        for r in res:
            pivots.append(r[0])
            pivots_in_x.append(r[1])
        
        pivots = torch.cat(pivots).cuda()
        pivots_in_x = torch.cat(pivots_in_x).cuda()
        
        att_w = shifted_rows / (1 - ths + self.eps)

        att_w[rowwise_count_cumsum]=1.0

        att_w_norm = scatter(att_w,dist_rowwise_inx,reduce='sum').gather(-1,dist_rowwise_inx)
        att_w = att_w / att_w_norm

        scaled_node_fts = x[dests_all]*att_w.view(-1,1)
        
        aggr_node_fts = scatter(scaled_node_fts,dist_rowwise_inx,dim=0)

        skipped_x = aggr_node_fts[pivots,:]
        
        pivots_in_x,sort_inx = torch.sort(torch.cat((pivots_in_x,unskip_pivots_in_x)))
        new_x = torch.cat((skipped_x,unskip_x))[sort_inx]
        
        # if you uncomment the following then uncomment if in line 145
        # else:
        #     pivots_in_x = unskip_pivots_in_x
        #     new_x = unskip_x
        
        new_batch = batch[pivots_in_x]
        new_scores = scores_ranked[pivots_in_x]
            
        new_x = new_x*new_scores.view(-1,1)

        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, pivots_in_x, num_nodes=scores_ranked.size(0))
        
        return new_x, edge_index, edge_attr, new_batch, pivots_in_x

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