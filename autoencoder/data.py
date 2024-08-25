import os
import os.path as osp
import networkx as nx
import torch
import numpy as np
from pygsp import graphs
from torch_geometric.data import Data
from torch_geometric.utils import to_edge_index
from torch_geometric.datasets import Planetoid, ModelNet
import torch_geometric.transforms as T


MODELNET_CONFIG = {
    "Airplane": {
        "classname": "airplane",
        "split": "train",
        "sample": 150,
    },
    "Car": {
        "classname": "car",
        "split": "train",
        "sample": 78,
    },
    "Guitar": {
        "classname": "guitar",
        "split": "train",
        "sample": 37,
    },
    "Person": {
        "classname": "person",
        "split": "train",
        "sample": 82,
    },
}

MODELNET_CUMSUM = {'airplane': 0,
                    'car': 2391, 
                    'guitar': 4690, 
                    'person': 6212}



def make_dataset(name, device, **kwargs):
    if "seed" in kwargs:
        np.random.seed(kwargs.pop("seed"))
    if name in graphs.__all__:
        return make_cloud(name,device)
    if name in MODELNET_CONFIG:

        return make_modelnet(**MODELNET_CONFIG[name],device=device)
    if name in ['Cora', 'CiteSeer', 'PubMed']: 
        return make_citation(name,device)

def make_cloud(name,device):
    if name.lower() == "grid2d":
        G = graphs.Grid2d(N1=8, N2=8)
    elif name.lower() == "ring":
        G = graphs.Ring(N=64)
    elif name.lower() == "bunny":
        G = graphs.Bunny()
    elif name.lower() == "airfoil":
        G = graphs.Airfoil()
    elif name.lower() == "minnesota":
        G = graphs.Minnesota()
    elif name.lower() == "sensor":
        G = graphs.Sensor(N=64)
    elif name.lower() == "community":
        G = graphs.Community(N=64)
    elif name.lower() == "barabasialbert":
        G = graphs.BarabasiAlbert(N=64)
    elif name.lower() == "davidsensornet":
        G = graphs.DavidSensorNet(N=64)
    elif name.lower() == "erdosrenyi":
        G = graphs.ErdosRenyi(N=64)
    else:
        raise ValueError("Unknown dataset: {}".format(name))
    
    if not hasattr(G, "coords"):
        G.set_coordinates(kind="spring")
    x = G.coords.astype(np.float32)
    A = G.W
    A = A.toarray() 
    assert all(np.unique(A)==np.array([0.,1.]))
    if A.dtype.kind != "b":
        A = A.astype("i")
    A = to_tensor_device(A,device)
    A = A.to_sparse()
    data = Data(x=to_tensor_device(x,device),edge_index=to_edge_index(A)[0])
    return data


def make_modelnet(classname="airplane", split="train", sample=151,name='40',device='cpu'):
    path = osp.join(osp.dirname(osp.realpath(__file__)),'data', 'ModelNet')
    # path = osp.join('data', 'ModelNet')
    split = True if split=='train' else False
    dataset = ModelNet(path,name,train=split,pre_transform=T.FaceToEdge()) 
    sample_inx = MODELNET_CUMSUM[classname]+sample
    graph = dataset[sample_inx]
    x, a = to_numpy(graph.pos), to_numpy(graph.edge_index)
    x = normalize_point_cloud(x)
    return Data(x=to_tensor_device(x,device),edge_index=to_tensor_device(a,device))


def make_citation(name,device):
    path = osp.join(osp.dirname(osp.realpath(__file__)),'data', 'Planetoid')
    graph = Planetoid(path, name)[0]

    x, a = to_numpy(graph.x), to_numpy(graph.edge_index)

    gg = nx.Graph(a)
    lay = nx.spring_layout(gg)
    x = np.array([lay[i] for i in range(a.shape[0])])
    
    return Data(x=to_tensor_device(x,device),edge_index=to_tensor_device(a,device))

def to_numpy(x):
    if 'cuda' in str(x.device): 
        return x.cpu().numpy()
    return x.numpy()

def to_tensor_device(x,device='cpu'):
    if isinstance(x,torch.Tensor):
        return x.to(device=device)
    elif isinstance(x,np.ndarray):
        return torch.tensor(x,device=device)   

    
def normalize_point_cloud(x):
    offset = np.mean(x, -2, keepdims=True)
    scale = np.abs(x).max()
    x = (x - offset) / scale
    x /= np.linalg.norm(x, axis=0, keepdims=True)

    return x
