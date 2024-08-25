import torch 

def unfold1d(input, kernel_size: int, stride: int):
    *shape, length = input.shape
    n_frames = (max(length, kernel_size) - kernel_size) // stride + 1
    tgt_length = (n_frames - 1) * stride + kernel_size
    input = input[..., :tgt_length].contiguous()
    strides = list(input.stride())
    strides = strides[:-1] + [stride, 1]
    return input.as_strided(shape + [n_frames, kernel_size], strides)


class PresetUnfold():
    UNFOLDED_ARR = None
    UNFOLDED_SRCS = None
    UNFOLDED_DESTS = None
    ARANGED_INX_FLIP= None
    
    def __init__(self,dataset_name=None,max_gSize=None, fixedStride=3):
        self.ds_dtype = torch.int32

        if dataset_name == 'PROTEINS':
            self.max_gSize = 620
        
        if dataset_name == 'DD':
            self.max_gSize = 5748

        if max_gSize is not None:
            self.max_gSize = max_gSize
        
        self.colSize = fixedStride
        
        self.setUnfoldedSrcs()
        self.setArangedInx()

    def setArangedInx(self):
        ARANGED_INX = torch.arange(self.max_gSize*self.colSize).reshape(self.max_gSize,
                                                                                  self.colSize)
        PresetUnfold.ARANGED_INX_FLIP = torch.flip(ARANGED_INX,(1,))
    def setUnfoldedArr(self):
        PresetUnfold.UNFOLDED_ARR = unfold1d(torch.arange(self.max_gSize),
                                self.colSize,1).to(self.ds_dtype).cuda()
    def setUnfoldedDests(self):
        col_arr = torch.arange(self.max_gSize)
        PresetUnfold.UNFOLDED_DESTS_FIRSTR = col_arr.repeat(self.max_gSize).reshape(self.max_gSize,self.max_gSize).cuda()

    def setUnfoldedSrcs(self):
        PresetUnfold.UNFOLDED_SRCS = torch.repeat_interleave(torch.arange(self.max_gSize),
                                                             self.colSize).reshape(self.max_gSize,
                                                                                   self.colSize) 



