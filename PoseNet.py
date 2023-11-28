import math
import torch
import torch.nn.functional as torch_F
from torch.optim.lr_scheduler import ExponentialLR
import rotation_conversions

# example configs
config = {
        'device': "cpu",#"cuda:0",
        'poseNet_freq': 5,
        'layers_feat': [None,256,256,256,256,256,256,256,256],
        'skip': [4],  
        'min_time': 0,
        'max_time': 100,
        'activ': 'relu',
        'cam_lr': 1e-3,
        'max_iter': 20000,
        'use_scheduler': False
        }


class PoseNet(torch.nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg['device']
        # init network
        self.transNet = TransNet(cfg)
        self.transNet.to(self.device)
        self.rotsNet = RotsNet(cfg)
        self.rotsNet.to(self.device)
        self.cam_lr = cfg['cam_lr']

        # init optimizer and scheudler
        self.optimizer_posenet = torch.optim.Adam([dict(params=self.transNet.parameters(),lr=self.cam_lr), dict(params=self.rotsNet.parameters(),lr=self.cam_lr*0.2) ])
        gamma = (1e-2)**(1./cfg['max_iter'])
        if self.cfg['use_scheduler']:
            self.scheduler = ExponentialLR(self.optimizer_posenet, gamma=gamma)

        # set the normalizer mapping
        self.min_time = cfg['min_time']
        self.max_time = cfg['max_time']


    def step(self):
        self.optimizer_posenet.step()
        if self.cfg['use_scheduler']:
            self.scheduler.step()
        self.optimizer_posenet.zero_grad()
 

    def forward(self, time):
        assert torch.all(time >= self.min_time) and torch.all(time <= self.max_time), 'time out of range'
        time = 2*(time - self.min_time) / (self.max_time - self.min_time) - 1
        trans_est = self.transNet.forward(self.cfg, time)
        rots_feat_est = self.rotsNet.forward(self.cfg, time)
        rotmat_est = rotation_conversions.quaternion_to_matrix(rots_feat_est)
        # make c2w
        c2w_est = torch.cat([rotmat_est, trans_est.unsqueeze(-1)],dim = -1 )

        return c2w_est

class TransNet(torch.nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.input_t_dim = cfg['poseNet_freq'] * 2 + 1
        self.define_network(cfg)
        self.device = cfg['device']
        self.cfg = cfg

    def define_network(self,cfg):
        self.mlp_transnet = torch.nn.ModuleList()
        layers_list = cfg['layers_feat'] 
        L = list(zip(layers_list[:-1],layers_list[1:]))  

        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = self.input_t_dim
            if li in cfg['skip'] : k_in += self.input_t_dim
            if li==len(L)-1: k_out = 3
            linear = torch.nn.Linear(k_in,k_out)
            self.initialize_weights(linear,out="small" if li==len(L)-1 else "all")
            self.mlp_transnet.append(linear)

    def initialize_weights(self,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="small":
            torch.nn.init.uniform_(linear.weight, b = 1e-6)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, cfg, index):
        index = torch.tensor(index).to(self.device)
        # todo encoding the index
        index = index.reshape(-1,1).to(torch.float32)
        points_enc = self.positional_encoding(index, L=cfg['poseNet_freq'] )
        points_enc = torch.cat([index,points_enc],dim=-1) # [B,...,6L+3]

        translation_feat = points_enc
        activ_f = getattr(torch_F,self.cfg['activ']) 

        for li,layer in enumerate(self.mlp_transnet):
            if li in cfg['skip']: translation_feat = torch.cat([translation_feat,points_enc],dim=-1)
            translation_feat = layer(translation_feat)
            if li==len(self.mlp_transnet)-1:
                translation_feat = torch_F.tanh(translation_feat) # note we assume bounds is [-1,1]
            else:
                translation_feat = activ_f(translation_feat) 
        return translation_feat

    def positional_encoding(self,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device = self.device)*math.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc


class RotsNet(torch.nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.input_t_dim = cfg['poseNet_freq'] * 2 + 1
        self.define_network(cfg)
        self.device = cfg['device']

    def define_network(self,cfg):
        layers_list = cfg['layers_feat'] 
        L = list(zip(layers_list[:-1],layers_list[1:]))  

        self.mlp_quad = torch.nn.ModuleList()
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = self.input_t_dim
            if li in cfg['skip']: k_in += self.input_t_dim
            if li==len(L)-1: k_out = 4 
            linear = torch.nn.Linear(k_in,k_out)

            self.initialize_weights(linear,out="small" if li==len(L)-1 else "all")
            self.mlp_quad.append(linear)

    def initialize_weights(self,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="small":
            torch.nn.init.uniform_(linear.weight, b = 1e-6)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, cfg, index):
        index = torch.tensor(index).to(self.device)
        index = index.reshape(-1,1).to(torch.float32)
        activ_f = getattr(torch_F,self.cfg['activ']) 

        points_enc = self.positional_encoding(index,L=cfg['poseNet_freq'] )
        points_enc = torch.cat([index,points_enc],dim=-1) # [B,...,6L+3]        
        rotation_feat = points_enc
        for li,layer in enumerate(self.mlp_quad):
            if li in cfg['skip']: rotation_feat = torch.cat([rotation_feat,points_enc],dim=-1)
            rotation_feat = layer(rotation_feat)
            if li==len(self.mlp_quad)-1:
                rotation_feat[:,1:] = torch_F.tanh(rotation_feat[:,1:])#torch_F.sigmoid(rotation_feat[:,1:])
                rotation_feat[:,0] = 1*(1 - torch_F.tanh(rotation_feat[:,0]))
            else:
                rotation_feat = activ_f(rotation_feat)

        norm_rots = torch.norm(rotation_feat,dim=-1)
        rotation_feat_norm = rotation_feat / (norm_rots[...,None] +1e-18)
        return rotation_feat_norm

    def positional_encoding(self,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device = self.device)*math.pi # [L] # ,device=cfg.device
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc
    

if __name__ == "__main__":
    """
    overfit exps
    """
    # data
    # random sample 100 different c2ws
    lie_instance = rotation_conversions.Lie()
    random_se3 = 0.15*torch.rand(config['max_time'], 6)
    random_SE3 = lie_instance.se3_to_SE3(random_se3) # ([100, 3, 4])
    
    # init network
    posenet = PoseNet(config)

    # train
    train_iters = 200
    times = torch.arange(100)
    for i in range(train_iters):
        est_c2ws = posenet.forward(times)
        # dummy loss 
        loss = torch.abs(random_SE3 - est_c2ws).mean()
        loss.backward()
        posenet.step()
        print(f"step{i},loss{loss}")



