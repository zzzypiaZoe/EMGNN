import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import init
import math
import numbers
import pandas as pd
import numpy as np

class dilated(nn.Module):
    def __init__(self, cin, cout, kernel_set, dilation_factor=2):
        super(dilated, self).__init__()
        cout = int(cout/len(kernel_set))
        self.temporal_mod = nn.ModuleList([
            nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor))
            for kern in kernel_set
        ])

    def forward(self,input):
        x = [temporal_mod(input)[..., -input.size(3):] for temporal_mod in self.temporal_mod]
        return torch.cat(x, dim=1)

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
    
    def forward(self,x,adj):
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj)
            out.append(h)
        ho = torch.cat(out,dim=1)
        return self.mlp(ho)

   
class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class temporal_mod(nn.Module):
    def __init__(self, res_ch: int, conv_ch: int, kernel_set, dilation_factor: int, dropout:float):
        super(temporal_mod, self).__init__()
        self.filter_conv = dilated(res_ch, conv_ch,kernel_set, dilation_factor)
        self.gate_conv = dilated(res_ch, conv_ch, kernel_set, dilation_factor)
        self.dropout = dropout

    def forward(self, x: Tensor):
        return F.dropout(torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x)), self.dropout, training=self.training)


class evolving_mod(nn.Module):
    def __init__(self, conv_ch: int, res_ch: int, gcn_depth: int,  static_embedding: int, 
                dynamic_embedding: int, dynamic_interval: int, dropout=0.3, propalpha=0.05):
        super(evolving_mod, self).__init__()
        self.linear_s2d = nn.Linear(static_embedding, dynamic_embedding)
        self.scale_spc_EGL = graph_mod(conv_ch, dynamic_embedding, dynamic_interval)
        self.dynamic_interval = dynamic_interval         

        self.gconv = mixprop(conv_ch, res_ch, gcn_depth, dropout, propalpha)


    def forward(self, x, static_node_ft):

        b, _, n, t = x.shape 
        dynamic_node_ft = self.linear_s2d(static_node_ft).unsqueeze(0).repeat( b, 1, 1)

        x_out = []       
        for i_t in range(0,t,self.dynamic_interval):     
            x_i =x[...,i_t:min(i_t+self.dynamic_interval,t)]

            input_state_i = torch.mean(x_i.transpose(1,2),dim=-1)
          
            dy_graph, dynamic_node_ft= self.scale_spc_EGL(input_state_i, dynamic_node_ft)

            x_out.append(self.gconv(x_i, dy_graph))

        return torch.cat(x_out, dim= -1) #x_out


class graph_mod(nn.Module):
    def __init__(self, input_size: int, dg_hidden_size: int, dynamic_interval: int):
        super(graph_mod, self).__init__()
        self.rz_gate = nn.Linear(input_size + dg_hidden_size, dg_hidden_size * 2)
        self.dg_hidden_size = dg_hidden_size
        self.h_candidate = nn.Linear(input_size + dg_hidden_size, dg_hidden_size)
        self.conv_layers = nn.ModuleList([normal_conv(dg_hidden_size) for _ in range(2)])
        self.dynamic_interval = dynamic_interval

    def forward(self, inputs: Tensor, states):
        r_z = torch.sigmoid(self.rz_gate(torch.cat([inputs, states], -1)))
        r, z = r_z.split(self.dg_hidden_size, -1)
        new_state = z * states + (1 - z) * torch.tanh(self.h_candidate(torch.cat([inputs, r * states], -1)))

        dy_sent = torch.unsqueeze(torch.relu(new_state), dim=-2).repeat(1, 1, states.size(1), 1)
        dy_revi = dy_sent.transpose(1, 2)
        y = torch.cat([dy_sent, dy_revi], dim=-1)
        support = self.conv_layers[0](y, (states.size(0), states.size(1), states.size(1)))
        mask = self.conv_layers[1](y, (states.size(0), states.size(1), states.size(1)))
        return support * torch.sigmoid(mask), new_state

class multiscale_ext(nn.Module):
    def __init__(self, res_ch: int, conv_ch: int, kernel_set, dilation_factor: int, gcn_depth: int, 
                static_embedding, dynamic_embedding, 
           skip_ch:int, t_len: int, nodes_num: int, layer_norm_affline, propalpha: float, dropout:float, dynamic_interval: int):
        super(multiscale_ext, self).__init__()

        self.t_conv = temporal_mod(res_ch, conv_ch, kernel_set, dilation_factor, dropout)
        self.skip_conv = nn.Conv2d(conv_ch, skip_ch, kernel_size=(1, t_len))
      
        self.s_conv = evolving_mod(conv_ch, res_ch, gcn_depth, static_embedding, dynamic_embedding, 
                                    dynamic_interval, dropout, propalpha)

        self.residual_conv = nn.Conv2d(conv_ch, res_ch, kernel_size=(1, 1))
        
        self.norm = LayerNorm((res_ch, nodes_num, t_len),elementwise_affine=layer_norm_affline)
       

    def forward(self, x: Tensor,  static_node_ft: Tensor):
        residual = x
        x = self.t_conv(x)       
        skip = self.skip_conv(x)
        x = self.s_conv(x, static_node_ft) + residual[:, :, :, -x.size(3):]
        return self.norm(x), skip

class Block(nn.ModuleList):
    def __init__(self, block_id: int, total_t_len: int, kernel_set, dilation_exp: int, n_layers: int, 
                 res_ch: int, conv_ch: int, gcn_depth: int, static_embedding, dynamic_embedding,  
                 skip_ch: int, nodes_num: int, layer_norm_affline, propalpha: float, dropout: float, dynamic_interval: int):
        super(Block, self).__init__()
        rf_block = int(1 + block_id * (kernel_set[-1] - 1) * (dilation_exp ** n_layers - 1) / (dilation_exp - 1)) if dilation_exp > 1 else block_id * n_layers * (kernel_set[-1] - 1) + 1
        dilation_factor = 1
        for i in range(n_layers):
            t_len = total_t_len - rf_block + (rf_block - 1) * dilation_factor
            self.append(multiscale_ext(res_ch, conv_ch, kernel_set, dilation_factor, gcn_depth, static_embedding, 
                                  dynamic_embedding, skip_ch, t_len, nodes_num, layer_norm_affline, propalpha, dropout, dynamic_interval))
            dilation_factor *= dilation_exp

    def forward(self, x: Tensor, static_node_ft: Tensor):
        skip = 0
        for l in self:
            x, skip_i = l(x, static_node_ft)
            skip = skip + skip_i
        return x, skip


class EMGNN(nn.Module):
    def __init__(self, dynamic_embedding: int, dynamic_interval: list, nodes_num: int, seq_length: int, pred_len : int,
                 in_dim: int, out_dim: int, n_blocks: int, n_layers: int, conv_ch: int, res_ch: int,
                 skip_ch: int, end_ch: int, kernel_set: list, dilation_exp: int, gcn_depth: int, device,
                 fc_dim: int, static_embedding=int, static_feat=None, dropout=float, propalpha=float, layer_norm_affline=True            
                 ):
        super(EMGNN, self).__init__()
       
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.nodes_num = nodes_num        
        self.device = device
        self.pred_len = pred_len
        self.static_embedding = static_embedding
        self.seq_length = seq_length
        self.receptive_field, self.total_t_len = self._calculate_receptive_field(n_blocks, n_layers, kernel_set, dilation_exp)    
      
        self.start_conv = nn.Conv2d(in_dim, res_ch, kernel_size=(1, 1))
        self.blocks = nn.ModuleList()
        for block_id in range(n_blocks):
            self.blocks.append(
                Block(block_id, self.total_t_len, kernel_set, dilation_exp, n_layers, res_ch, conv_ch, gcn_depth,
                 static_embedding, dynamic_embedding, skip_ch, nodes_num, layer_norm_affline, propalpha, dropout, dynamic_interval))

        self.skip0 = self._create_skip_conv(in_dim, skip_ch, self.total_t_len)
        self.skipE = self._create_skip_conv(res_ch, skip_ch, self.total_t_len - self.receptive_field + 1)
        
        in_channels = skip_ch
        final_channels = pred_len * out_dim

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, end_ch, kernel_size=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(end_ch, final_channels, kernel_size=(1, 1), bias=True)     
        )
        self.stfea_encode = multiscale_ext(static_embedding, fc_dim)
        self.static_feat = static_feat

    def _calculate_receptive_field(self, n_blocks, n_layers, kernel_set, dilation_exp):
        kernel_size = kernel_set[-1]
        if dilation_exp > 1:
            receptive_field = int(1 + n_blocks * (kernel_size - 1) * (dilation_exp ** n_layers - 1) / (dilation_exp - 1))
        else:
            receptive_field = n_blocks * n_layers * (kernel_size - 1) + 1
        total_t_len = max(receptive_field, self.seq_length)
        return receptive_field, total_t_len

    def _create_skip_conv(self, in_channels, out_channels, kernel_size):
        return nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), bias=True) 

    def forward(self, input):
        b, _, n, t = input.shape
        if self.seq_length<self.receptive_field:
            input = F.pad(input,(self.receptive_field-self.seq_length,0,0,0), mode='replicate')
        
        x = self.start_conv(input)
           
        static_node_ft = self.stfea_encode(self.static_feat)


        skip_list = [self.skip0(F.dropout(input, self.dropout, training=self.training))]
        for j in range(self.n_blocks):    
            x, skip_list= self.blocks[j](x, static_node_ft , skip_list)
                    
        skip_list.append(self.skipE(x)) 
        skip_list = torch.cat(skip_list, -1)
        skip_sum = torch.sum(skip_list, dim=3, keepdim=True)
        x = self.out(skip_sum)
        x = x.reshape(b, self.pred_len, -1, n).transpose(-1, -2)
        return x 





    