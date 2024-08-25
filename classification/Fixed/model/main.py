import os
from tqdm import tqdm
from datetime import datetime
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.loader import ImbalancedSampler
from torch_geometric import utils
from networks import  Net
import torch.nn.functional as F
import argparse
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split,StratifiedKFold,KFold
import numpy as np
import time
from utils import PresetUnfold

def val_test(model,loader,epoch):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data,epoch=epoch)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)

def test(model,loader,last_epoch=None):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        if last_epoch is None:
            last_epoch=args.epochs
        out = model(data,epoch=last_epoch)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)

def train(args,FOLDER,checkpoint_path,model,optimizer,train_loader, val_loader):
    
    writerAvail = args.isTb

    writer =  SummaryWriter(log_dir=f"{FOLDER}") if writerAvail else None

    starting_epoch = 0
    min_loss = 1e10
    max_acc = 1e-10
    patience = 0
    min_loss_epoch = 0
    max_acc_epoch = 0

    # loading previous checkpoints
    if len(checkpoint_path) > 0:
        checkpoint = torch.load(f"{FOLDER}/{checkpoint_path[-1]}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']+1
        patience = checkpoint['currentPatience']
        min_loss = checkpoint['min_val_loss']
        max_acc = checkpoint['max_val_acc']
        min_loss_epoch = checkpoint['min_loss_epoch']
        max_acc_epoch = checkpoint['max_acc_epoch']

    for epoch in tqdm(range(starting_epoch,args.epochs)): #tqdm
        model.train()
        correct = 0.0
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            out = model(data, epoch=epoch)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += torch.sum(loss).item()

        train_acc = correct / len(train_loader.dataset)
        train_loss = train_loss / len(train_loader.dataset)
        val_acc,val_loss = val_test(model,val_loader,epoch)

        if writer:
            writer.add_scalar('Loss/train',train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)

            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

            writer.flush()

        # save checkpoint of current epoch
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'min_val_loss':min_loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'currentPatience':patience,
                'max_val_acc':max_acc,
                'min_loss_epoch':min_loss_epoch,
                'max_acc_epoch':max_acc_epoch
                }, f"{FOLDER}/latest-model-{epoch}.pth")

        if val_loss < min_loss:
            torch.save(model.state_dict(),'{}/latest.pth'.format(FOLDER))
            min_loss = val_loss
            patience = 0
            min_loss_epoch = epoch
        else:
            patience += 1

        if val_acc >= max_acc:
            torch.save(model.state_dict(),'{}/latest-acc.pth'.format(FOLDER))
            max_acc = val_acc
            max_acc_epoch = epoch
        
        if epoch >0:
            os.remove(f"{FOLDER}/latest-model-{epoch-1}.pth")

        if patience > args.patience:
            break 

    torch.save(model.state_dict(),'{}/latest-100.pth'.format(FOLDER))

    return min_loss_epoch  
    
def main(args,currentFold,skf_splits,dataset,logs_files):
    min_val_loss_accs = []

    logs = open(logs_files[0], "a")

    for f in range(currentFold,skf_splits):

        fold_path = os.path.join(folds_save_path,f'fold_{f}')

        print(f'Running fold {f}...')
        
        
        train_inx = np.load(f"{fold_path}/train_inx.npy",)
        val_inx = np.load(f"{fold_path}/val_inx.npy")
        test_inx = np.load(f"{fold_path}/test_inx.npy")

        training_set = dataset[torch.tensor(train_inx,dtype=torch.long)]
        validation_set = dataset[torch.tensor(val_inx,dtype=torch.long)]
        test_set = dataset[torch.tensor(test_inx,dtype=torch.long)]
        
        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
        test_loader = DataLoader(test_set,batch_size=1,shuffle=False)

        checkpoints = os.listdir(fold_path)
        checkpoint_path = list(filter(lambda i : 'latest-model-' in i , checkpoints))
        checkpoint_path = sorted(checkpoint_path,key=lambda n : [int(n[:-4].split("-")[2]),n])

        args.device = 'cpu'
        if args.seed:
            torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            if args.seed:
                torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            args.device = 'cuda:0'

        model = Net(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        min_loss_epoch = train(args,fold_path,checkpoint_path,model,optimizer,train_loader,val_loader)

        fold_logs = open(f'{logs_files[1]}/folds_res_log.txt', "a")

        if f == args.fold:
            fold_logs.write(f"batch_size: {args.batch_size}, nhid: {args.nhid}, lr: {args.lr}, weight_decay: {args.weight_decay}, dropout_ratio: {args.dropout_ratio}\n")

        fold_logs.write(f"====== FOLD {f} =======\n")

        model = Net(args).to(args.device)
        model.load_state_dict(torch.load('{}/latest.pth'.format(fold_path)))

        test_acc,test_loss = test(model,test_loader,min_loss_epoch)
        fold_logs.write("{}\n".format(test_acc))
        min_val_loss_accs.append(test_acc)

        fold_logs.close()

        if args.breakFold:
            break

    logs.write(f'====== LOG {0} ======\n')
    logs.write(f'{sum(min_val_loss_accs)/len(min_val_loss_accs)}\n')
    logs.close()

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--nhid', type=int, default=128, 
                        help='hidden size')
    parser.add_argument('--pooling_ratio', type=float, default=0.5,
                        help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.4,
                        help='dropout ratio')
    parser.add_argument('--dataset', type=str, default='DD',
                        help='DD/PROTEINS/ENZYMES/NCI1/NCI109/Mutagenicity')
    parser.add_argument('--epochs', type=int, default=1000 ,
                        help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for earlystopping')
    parser.add_argument('--pooling_layer_type', type=str, default='GCNConv')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for data loading')
    parser.add_argument('--isTb',action="store_true", 
                        help='is Tensorboard logging required')
    parser.add_argument('--dataDir', type=str, default='../../data', 
                        help='path of directory containing datasets.')
    parser.add_argument('--prePool', type=str, default='sag', 
                        help='type of pooling used before skipping.')
    parser.add_argument('--nFolds', type=int, default=10, 
                        help='number of folds for cross validation.')
    parser.add_argument('--fold', type=int, default=0, 
                        help='fold to start.')
    parser.add_argument('--splitsSaved', action="store_true", 
                        help='whether folds/splits are saved.')
    parser.add_argument('--breakFold',action="store_true", 
                        help='whether break from current rep.')
    parser.add_argument('--fixStride', type=int, default=3, 
                        help='fixed size of stride..')
    parser.add_argument('--p-list', type=list_of_ints, help='number of process for each pooling layer as a list.')
    parser.add_argument('--seed', type=int, default=777, 
                        help='seed.')


    args = parser.parse_args()
    REP = 0
    print(f'seed: {args.seed}')
    args.device = 'cpu'
    if torch.cuda.is_available():
        args.device = 'cuda:0'
            
    dataset = TUDataset(os.path.join(args.dataDir,args.dataset),name=args.dataset)
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features

    max_nodes = 0
    for d in dataset:
        max_nodes = max(d.num_nodes, max_nodes)

    PresetUnfold(max_gSize=max_nodes,fixedStride=args.fixStride)

    num_training = int(len(dataset)*0.8)

    kfold_exp_path = f'../{args.dataset}'  

    if not os.path.exists(kfold_exp_path):
        os.mkdir(kfold_exp_path)

    skf_splits = 10

    splitsSaved = args.splitsSaved

    currentFold = args.fold
    rep_res_log = os.path.join(kfold_exp_path,f"rep_res_log.txt")
    
    folds_save_path = os.path.join(kfold_exp_path,f"rep_{REP}")

    logs = [rep_res_log,folds_save_path]

    if not os.path.exists(folds_save_path):
        os.mkdir(folds_save_path)

    if splitsSaved is False:
        test_inx = []
        # StratifiedKFold
        skf = StratifiedKFold(n_splits=args.nFolds,shuffle=True,random_state=args.seed)
        for i, (train_inx,test_inx_i) in enumerate(skf.split(np.arange(len(dataset)), dataset.y.numpy())):
            test_inx.append(test_inx_i)
            
        val_inx = [test_inx[i-1] for i in range(len(test_inx))]

        for i in range(len(val_inx)):
            fold_path = os.path.join(folds_save_path,f'fold_{i}')
            os.makedirs(fold_path)
    
            train_mask = torch.ones(len(dataset), dtype=torch.int64)
            train_mask[test_inx[i]] = 0
            train_mask[val_inx[i]] = 0
            train_inx_i = train_mask.nonzero().view(-1).numpy().astype(val_inx[i].dtype)
            
            np.save(f"{fold_path}/train_inx.npy",train_inx_i)
            np.save(f"{fold_path}/val_inx.npy",val_inx[i])
            np.save(f"{fold_path}/test_inx.npy",test_inx[i])

    main(args,currentFold,skf_splits,dataset,logs)
