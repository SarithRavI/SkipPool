import argparse
import os.path as osp
import time
import os
import torch

import numpy as np
from layers.GeneralConv import CustomGeneralConv
from layers.CustomMLP import CustomMLP
from torch.nn import MSELoss
import torch.nn.functional as F

from pooling.layers_stable import SKipPool
from pooling.layers_full_stable import SKipPool_Full
from pooling.unpool import UnPool
from pooling.utils import PresetUnfold

from data import make_dataset

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--es_tol', type=float, default=1e-6)
parser.add_argument('--pooling_ratio', type=float, default=0.5)
parser.add_argument('--tot_patience', type=int, default=1000)
parser.add_argument('--tot_epochs', type=int, default=1000)
parser.add_argument('--runs', type=int, default=3)
parser.add_argument('--curr_run', type=int, default=0)
parser.add_argument('--breakRun', action="store_true",
                        help='whether to break the run after completion.')
parser.add_argument('--complete', action="store_true",
                        help='whether to use complete version instead of stable.')
parser.add_argument('--full', action="store_true",
                        help='whether to use full version of stable.')
parser.add_argument('--use_idw', action="store_true",
                        help='whether to use idw kernel.')
parser.add_argument('--isTb', action="store_true",
                        help='whether to plot using tb.')
parser.add_argument('--doKernelTest', action="store_true",
                        help='whether to plot using tb.')
parser.add_argument('--fixStride', type=int, default=3)


args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
    args.device= 'cuda'
else:
    device = torch.device('cpu')
    args.device= 'cpu'

train_data = make_dataset(name=args.dataset,device=device)
print(f'train data: {train_data}')
test_data = train_data

class GAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,pool,unpool, batch_norm= False, post_processing=True,seed=None):
        super().__init__()

        # encoder
        self.pre_mlp = CustomMLP(in_channels,hidden_channels,batch_norm=batch_norm)
        self.conv1 = CustomGeneralConv(hidden_channels,hidden_channels,batch_norm=batch_norm)
        self.skip = torch.cat   
        self.pool = pool(hidden_channels*2, 
                              init_ratio=0.9999,
                              last_ratio=args.pooling_ratio,
                              tot_epochs= args.tot_epochs,
                              scorer='sag',fixStride=args.fixStride,seed=seed)
        
        # decoder 
        self.unpool = unpool()
        self.post_processing = post_processing
        if post_processing:
            self.conv2 = CustomGeneralConv(hidden_channels*2,hidden_channels,batch_norm=batch_norm)
            self.post_mlp =CustomMLP(hidden_channels*3,in_channels,batch_norm=batch_norm,final_activation='Identity')

    def forward(self, x, edge_index,batch,epoch):        
        x = self.pre_mlp(x)
        x = self.skip([self.conv1(x,edge_index), x],dim=1)
        x_pool, a_pool, _, _, S = self.pool(x, edge_index, epoch, None, batch, False)
        if isinstance(S,list):
            s = S[0]
            max_diffs = S[1]
        else:
            s = S  

        x_unpool, _ = self.unpool(x_pool,a_pool,s)

        # for autoencoder test we use the original edge index as mentioned in the original paper..
        a_unpool = edge_index

        if self.post_processing:
            x_unpool = self.skip([self.conv2(x_unpool, a_unpool), x_unpool],dim=1)
            x_unpool = self.post_mlp(x_unpool)

        return x_unpool, a_unpool , x_pool, a_pool, S

in_channels = train_data.x.size(1)
hidden_channels = 256

def loss_fn(x,x_pred):
    loss = MSELoss()
    return loss(x_pred,x)

def to_numpy(x):
    if 'cuda' in str(x.device):
        x = x.cpu()
    return x.numpy()

@torch.no_grad()
def test(model, data,best_loss_epoch,trial_folder):
    model.eval()
    batch = torch.zeros(data.x.size(0),dtype = torch.int64, device=data.x.device)
    x_unpool, a_unpool, x_pool, a_pool, S = model(data.x, data.edge_index, batch, best_loss_epoch)
    if isinstance(S,list):
        s = S[0]
        max_diffs = S[1]
    else:
        s = S  

    loss = loss_fn(data.x, x_unpool)

    # print('Saving data for plotting..')
    # np.savez(
    #     os.path.join(trial_folder,"matrices.npz"),
    #     X=to_numpy(data.x),
    #     A=to_numpy(data.edge_index),
    #     X_pool=to_numpy(x_pool),
    #     A_pool=to_numpy(a_pool),
    #     X_pred=to_numpy(x_unpool),
    #     A_pred=to_numpy(a_unpool),
    #     S=to_numpy(s),
    #     loss=to_numpy(loss),
    #     training_times=[],
    # )

    return loss

run_seeds = [777]
for run in range(args.curr_run,len(run_seeds)):
    args.seed = run_seeds[run]
    epoch = 0 
    starting_epoch = 0
    es_tol = args.es_tol
    best_loss = 100000
    best_loss_epoch = 0 
    patience = 0 
    if args.full:
        FOLDER = osp.join(osp.dirname(osp.realpath(__file__)),'Full')
    else:
        FOLDER = osp.join(osp.dirname(osp.realpath(__file__)),'Fixed')
    
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    
    FOLDER = osp.join(FOLDER ,f'{args.dataset}')
    
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    
    trial_folder = os.path.join(FOLDER ,f'run_{run}')

    fin_pooled_gSize = 0

    if not os.path.exists(trial_folder):
        os.mkdir(trial_folder)

    checkpoints = os.listdir(trial_folder)
    checkpoint_path = list(filter(lambda i : 'latest-model-' in i , checkpoints))
    checkpoint_path = sorted(checkpoint_path,key=lambda n : [int(n[:-4].split("-")[2]),n])

    writer =  SummaryWriter(log_dir=f"{trial_folder}") if args.isTb else None

    # set the seed
    torch.manual_seed(args.seed)
    if args.device=='cuda':
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.full:
        pool = SKipPool_Full
    else:
        pool = SKipPool
    unpool = UnPool
    model = GAE(in_channels,hidden_channels,pool=pool,unpool=unpool,seed=args.seed) # ,seed=run_seeds[run]
    PresetUnfold(max_gSize=train_data.x.size(0),fixedStride=args.fixStride)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if len(checkpoint_path) > 0:
        checkpoint = torch.load(f"{trial_folder}/{checkpoint_path[-1]}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        patience = checkpoint['currentPatience']
        best_loss = checkpoint['best_loss']
        best_loss_epoch = checkpoint['best_loss_epoch']
    
    epoch += starting_epoch

    while patience != args.tot_patience:
        model.train()
        optimizer.zero_grad()
        batch = torch.zeros(train_data.x.size(0),dtype = torch.int64, device=train_data.x.device)
        x_pred, _, x_pool, _, S = model(train_data.x, train_data.edge_index, batch, epoch)
        loss = loss_fn(train_data.x, x_pred)
        loss.backward()
        optimizer.step()

        if isinstance(S,list):
            max_diffs = S[1]
        else:
            s = S  

        if writer:
            writer.add_scalar('mean',torch.mean(max_diffs), epoch)
            writer.add_scalar('std', torch.std(max_diffs), epoch)
            writer.flush()


        if epoch==0:
            print(f'original graph size: {train_data.x.size(0)}')
            print(f'pooled graph size: {x_pool.size(0)}')
        elif epoch % 500 == 0:
            print(f'pooled graph size at {epoch}: {x_pool.size(0)}')
        else:
            fin_pooled_gSize = x_pool.size(0)
        
        if loss + es_tol < best_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                }, f"{trial_folder}/latest.pth")
            best_loss = loss 
            best_loss_epoch = epoch
            patience = 0 
            print("min MSE at {} : {:.4e}".format(best_loss_epoch,best_loss))
        else:
            patience +=1 

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_loss':best_loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'currentPatience':patience,
                'best_loss_epoch':best_loss_epoch,
                }, f"{trial_folder}/latest-model-{epoch}.pth")

        if epoch >0:
            os.remove(f"{trial_folder}/latest-model-{epoch-1}.pth")        

        if patience == args.tot_patience:
            break

        epoch += 1 

    if fin_pooled_gSize > 0:
        print(f'final pooled graph size: {fin_pooled_gSize}')

    print("\nEvaluation...\n")

    # set the seed 
    torch.manual_seed(args.seed)
    if args.device=='cuda':
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # load the best model   
    model = GAE(in_channels,hidden_channels,pool=pool,unpool=unpool,seed=args.seed)
    model = model.to(device)

    checkpoint = torch.load('{}/latest.pth'.format(trial_folder))
    model.load_state_dict(checkpoint['model_state_dict'])
    best_loss_epoch = checkpoint['epoch']
    loss_out = test(model, test_data, best_loss_epoch,trial_folder)
    print("Final MSE: {:.4e}".format(loss_out.item()))

    fold_logs = open(f'{FOLDER}/res_log.txt', "a")
    fold_logs.write("run {}\tFinal MSE: {:.4e}\n".format(run, loss_out.item()))
    fold_logs.close()
    
    if args.breakRun:
        break
    print('sleeping...')
    time.sleep(300)
