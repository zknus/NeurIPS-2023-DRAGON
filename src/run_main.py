# from ogb.nodeproppred import Evaluator
import argparse
import time
import os

import numpy as np
import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import torch.nn.functional as F

from GNN import GNN
# from GNN_early import GNNEarly
# from GNN_KNN import GNN_KNN
# from GNN_KNN_early import GNNKNNEarly
from data import get_dataset, set_train_val_test_split
# from graph_rewiring import apply_KNN, apply_beltrami, apply_edge_sampling
from best_params import best_params_dict
from heterophilic import get_fixed_splits
from utils import ROOT_DIR
# from CGNN import CGNN, get_sym_adj
# from CGNN import train as train_cgnn
import sys
import json
# from GNN_heter import GNNheter
# from GNN_he import GNNhe
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import is_undirected, to_undirected
import itertools
import random
from run_config import parser
import csv
def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


def add_labels(feat, labels, idx, num_classes, device):
  onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
  if idx.dtype == torch.bool:
    idx = torch.where(idx)[0]  # convert mask to linear index
  onehot[idx, labels.squeeze()[idx]] = 1

  return torch.cat([feat, onehot], dim=-1)


def get_label_masks(data, mask_rate=0.5):
  """
  when using labels as features need to split training nodes into training and prediction
  """
  if data.train_mask.dtype == torch.bool:
    idx = torch.where(data.train_mask)[0]
  else:
    idx = data.train_mask
  mask = torch.rand(idx.shape) < mask_rate
  train_label_idx = idx[mask]
  train_pred_idx = idx[~mask]
  return train_label_idx, train_pred_idx


def train(model, optimizer, data, pos_encoding=None):
  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask

  out = model(feat, pos_encoding)

  if model.opt['dataset'] == 'ogbn-arxiv':
    lf = torch.nn.functional.nll_loss
    loss_orig = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  elif model.opt['dataset'] in ['minesweeper', 'workers', 'questions']:
    # logits = F.log_softmax(out, dim=1)
    # logts = logits.argmax(dim=1)
    # print("out shape: ", out.shape)
    # print("data.y shape: ", data.y.shape)
    loss_orig = F.binary_cross_entropy_with_logits(out.squeeze()[data.train_mask], data.y[data.train_mask].float())
    # loss_orig = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
  else:
    lf = torch.nn.CrossEntropyLoss()
    loss_orig = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])

  loss = loss_orig
  model.fm.update(model.getNFE())
  model.resetNFE()
  loss.backward()
  optimizer.step()
  model.bm.update(model.getNFE())
  model.resetNFE()
  return loss.item()



@torch.no_grad()
def test(model, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
  model.eval()
  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
  logits, accs = model(feat, pos_encoding), []

  if opt['dataset'] in [ 'minesweeper', 'workers', 'questions']:
    # print("using ROC-AUC metric")
    logits = logits.clamp(min=-100, max=100)
    # logits = F.softmax(logits, dim=1)
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
      # pred = logits.max(1)[1]
      # acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
      mask_idx = torch.where(mask)[0]
      y_true = data.y[mask_idx].cpu().numpy()
      y_score = logits[mask_idx].cpu().numpy()
      # test if logits has nan
      if np.isnan(y_score).any():
        print("logits has nan")
        y_score = np.nan_to_num(y_score)

      acc = roc_auc_score(y_true=data.y[mask_idx].cpu().numpy(), y_score=y_score).item()
      # acc = roc_auc_score(y_true=data.y[mask_idx].cpu().numpy(),y_score=logits[:,1][mask_idx].cpu().numpy()).item()
      accs.append(acc)

  else:
    logits = F.log_softmax(logits, dim=1)
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
      pred = logits[mask].max(1)[1]
      acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
      accs.append(acc)
  return accs


def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)





def merge_cmd_args(cmd_opt, opt):

  if cmd_opt['beltrami']:
    opt['beltrami'] = True
  if cmd_opt['function'] is not None:
    opt['function'] = cmd_opt['function']
  if cmd_opt['block'] is not None:
    opt['block'] = cmd_opt['block']
  if cmd_opt['attention_type'] != 'scaled_dot':
    opt['attention_type'] = cmd_opt['attention_type']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['method'] is not None:
    opt['method'] = cmd_opt['method']
  if cmd_opt['step_size'] != 1:
    opt['step_size'] = cmd_opt['step_size']
  if cmd_opt['time'] is not None:
    opt['time'] = cmd_opt['time']
  if cmd_opt['epoch'] != 100:
    opt['epoch'] = cmd_opt['epoch']
  if not cmd_opt['not_lcc']:
    opt['not_lcc'] = False
  if cmd_opt['num_splits'] != 1:
    opt['num_splits'] = cmd_opt['num_splits']
  if cmd_opt['dropout'] is not None:
    opt['dropout'] = cmd_opt['dropout']
  if cmd_opt['hidden_dim'] is not None:
    opt['hidden_dim'] = cmd_opt['hidden_dim']
  if cmd_opt['decay'] is not None:
    opt['decay'] = cmd_opt['decay']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['edge_homo']  != 0:
    opt['edge_homo'] = cmd_opt['edge_homo']
  if cmd_opt['use_mlp'] is not None:
    opt['use_mlp'] = cmd_opt['use_mlp']

  if cmd_opt['lr'] is not None:
    opt['lr'] = cmd_opt['lr']
  if cmd_opt['input_dropout'] is not None:
    opt['input_dropout'] = cmd_opt['input_dropout']
  if cmd_opt['patience'] is not None:
    opt['patience'] = cmd_opt['patience']
  if cmd_opt['data_norm'] is not None:
    opt['data_norm'] = cmd_opt['data_norm']
  if cmd_opt['runtime'] is not None:
    opt['runtime'] = cmd_opt['runtime']

  if cmd_opt['alpha_list'] is not None:
    opt['alpha_list'] = cmd_opt['alpha_list']

  print("merge cmd args done")

def set_seed(seed=123):
  random.seed(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

def get_optimizer_group(optimizer_name, grouped_parameters, **kwargs):
  if optimizer_name == 'adam':
    optimizer = torch.optim.Adam(grouped_parameters, **kwargs)
  elif optimizer_name == 'sgd':
    optimizer = torch.optim.SGD(grouped_parameters, **kwargs)
  elif optimizer_name == 'rmsprop':
    optimizer = torch.optim.RMSprop(grouped_parameters, **kwargs)
  elif optimizer_name == 'adagrad':
    optimizer = torch.optim.Adagrad(grouped_parameters, **kwargs)
  elif optimizer_name == 'adamax':
    optimizer = torch.optim.Adamax(grouped_parameters, **kwargs)
  # Add more optimizers here as needed
  else:
    raise ValueError("Invalid optimizer name")
  return optimizer

def combined_optimizer(model, opt):
  parameters_coefficient = [p for name, p in model.named_parameters() if p.requires_grad and 'coefficient' in name]
  print("parameters_coefficient: ", parameters_coefficient)
  parameters_other = [p for name, p in model.named_parameters() if p.requires_grad and 'coefficient' not in name]

  grouped_parameters = [
    {'params': parameters_other, 'lr': opt['lr'], 'weight_decay': opt['decay']},
    {'params': parameters_coefficient, 'lr': opt['lr_weight'], 'weight_decay': 0.001}
  ]

  optimizer = get_optimizer_group(opt['optimizer'], grouped_parameters)
  return optimizer

def main(opt,split):


  set_seed(opt['seed'])
  dataset = get_dataset(opt, f'{ROOT_DIR}/data', opt['not_lcc'],split)
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  device = torch.device('cuda:' + str(opt['cuda']) if torch.cuda.is_available() else 'cpu')


  pos_encoding = None

  # if opt['dataset'] in ['minesweeper', 'workers', 'questions']:
  #   dataset.num_classes =1



  model = GNN(opt, dataset, device).to(device)
  #
  if not opt['planetoid_split'] and opt['dataset'] in ['Cora','Citeseer','Pubmed']:
    print("using random split: set_train_val_test_split")
    dataset.data = set_train_val_test_split(opt['seed'], dataset.data, num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)

  data = dataset.data.to(device)

  data.edge_index = to_undirected(data.edge_index)
  # print("is undirected: ", is_undirected(data.edge_index, data.edge_attr))
  # data = dataset[0].to(device)
  print("num of train samples: ", len(torch.nonzero(data.train_mask,as_tuple=True)[0]))
  print("num of val samples: ", len(torch.nonzero(data.val_mask,as_tuple=True)[0]))
  print("num of test samples: ", len(torch.nonzero(data.test_mask,as_tuple=True)[0]))

  parameters = [p for p in model.parameters() if p.requires_grad]
  print_model_params(model)
  optimizer = combined_optimizer(model, opt)
  best_time = best_epoch = train_acc = val_acc = test_acc = 0


  this_test = test
  counter = 0
  for epoch in range(1, opt['epoch']):
    start_time = time.time()

    loss = train(model, optimizer, data, pos_encoding)
    tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)



    best_time = opt['time']
    if tmp_val_acc > val_acc:
      best_epoch = epoch
      train_acc = tmp_train_acc
      val_acc = tmp_val_acc
      test_acc = tmp_test_acc
      best_time = opt['time']
      counter = 0
    else:
      counter = counter + 1
      if counter == opt['patience']:
        break


    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best time: {:.4f}'

    print(log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, tmp_train_acc, tmp_val_acc, tmp_test_acc, best_time))
  print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d} and best time {:03f}'.format(val_acc, test_acc,
                                                                                                     best_epoch,
                                                                                                     best_time))
  alpha_coff = [p.detach().cpu().numpy().tolist() for p in model.odeblock.coefficient]


  return train_acc, val_acc, test_acc,opt,alpha_coff


if __name__ == '__main__':


  args = parser.parse_args()

  cmd_opt = vars(args)

  try:
    best_opt = best_params_dict[cmd_opt['dataset']]
    opt = {**cmd_opt, **best_opt}
    merge_cmd_args(cmd_opt, opt)
    print("combined with best params")
    # print(opt)
  except KeyError:
    opt = cmd_opt
    print("using cmd args")
    print(opt)






  timestr = time.strftime("%Y%m%d-%H%M%S")

  # create a folder to store the log
  if not os.path.exists("log_order"):
    os.makedirs("log_order")
  filename = "log_order/" + str(args.dataset)+ '_' + str(args.method) + '_'+ str(args.function)+ '_' + str(args.block)+ '_'+ timestr + ".txt"








  command_args = " ".join(sys.argv)
  best_test_acc = 0
  with open(filename, 'a') as f:
    json.dump(command_args, f)
    f.write("\n")

  # n_splits = 5
  best = []
  train_acc_list = []
  val_acc_list = []
  t_total = time.time()
  n_splits = opt['runtime']
  seed_init = opt['seed']
  for split in range(opt['runtime']):
    opt['seed'] = seed_init + split
    train_acc,val_acc, test_acc, opt_final,alpha_coff = main(opt,split)
    best.append(test_acc)


    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    with open(filename, 'a') as f:
      json.dump(test_acc, f)
      f.write("\n")
  print('Mean test accuracy: ', np.mean(np.array(best) * 100), 'std: ', np.std(np.array(best) * 100))
  print("test acc: ", best)
  test_result = np.mean(np.array(best) * 100)
  test_std = np.std(np.array(best) * 100)

  with open(filename, 'a') as f:
    f.write("*" * 50 + "\n")
    f.write("test_acc_mean: " + str(test_result) + "\n")
    f.write("test_acc_std: " + str(test_std) + "\n")
    f.write("\n")
    f.write("alpha_list: " + str(opt_final['alpha_list']) + "\n")
    f.write("alpha_coffe: " + str(alpha_coff) + "\n")

  with open(filename, 'a') as f:
    json.dump(best_opt, f, indent=2)






