from base_classes import ODEblock
import torch
from utils import get_rw_adj, gcn_norm_fill_val
from block_fractional_euler import *
from torch import nn
class ConstantODEblock_FRAC_MULTI_ORDER(ODEblock):
  def __init__(self, odefunc, opt, data, device, t=torch.tensor([0, 1])):
    super(ConstantODEblock_FRAC_MULTI_ORDER, self).__init__(odefunc,  opt, data,   device, t)


    self.odefunc = odefunc(opt['hidden_dim'], opt['hidden_dim'], opt, data, device)

    if opt['data_norm'] == 'rw':
      edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                                                   fill_value=opt['self_loop_weight'],
                                                                   num_nodes=data.num_nodes,
                                                                   dtype=data.x.dtype)
    else:
      edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
                                           fill_value=opt['self_loop_weight'],
                                           num_nodes=data.num_nodes,
                                           dtype=data.x.dtype)
    self.odefunc.edge_index = edge_index.to(device)
    self.odefunc.edge_weight = edge_weight.to(device)
    self.device = device
    self.opt = opt

    self.alpha = self.opt['alpha_list']
    # create self.coefficient as a list of learned parameters with differnt initial values
    self.coefficient = nn.ParameterList([nn.Parameter(torch.tensor([1.0])) for _ in range(len(self.alpha))])
  def forward(self, x):
    t = self.t.type_as(x)
    func = self.odefunc
    state = x
    # for coff in self.coefficient:
    #   print(coff)
    if self.opt['method'] == "GLorder":
        z = GL_order_n(self.alpha, self.coefficient, func, state, tspan= torch.arange(0,self.opt['time'],self.opt['step_size']),device=self.device)
    else:
        raise ValueError("Method not implemented")

    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
