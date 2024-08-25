import torch
import torch.nn 
import torch.nn.functional as F

class UnPool(torch.nn.Module):
    def __init__(self):
        super(UnPool,self).__init__()

    def forward(self,x,a,s):
        if s.size(0)<s.size(1):
            # if s is in shape K*N it will be transposed
            s = s.T
        s_inv = torch.linalg.pinv(s)
        s_inv_t = s_inv.T 
        return s_inv_t @ x, a
    