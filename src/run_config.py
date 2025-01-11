import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--use_cora_defaults', action='store_true',
                  help='Whether to run with best params for cora. Overrides the choice of dataset')
parser.add_argument('--cuda', default=1, type=int)
# data args
parser.add_argument('--dataset', type=str, default='twitch-gamer',
                  help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv,chameleon, squirrel,'
                       'wiki-cooc, roman-empire, amazon-ratings, minesweeper, workers, questions',)
parser.add_argument('--data_norm', type=str, default='gcn',
                  help='rw for random walk, gcn for symmetric gcn norm')
parser.add_argument('--self_loop_weight', default=1,type=float, help='Weight of self-loops.')
parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
parser.add_argument('--geom_gcn_splits', default=True, dest='geom_gcn_splits', action='store_true',
                  help='use the 10 fixed splits from '
                       'https://arxiv.org/abs/2002.05287')
parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                  help='the number of splits to repeat the results on')
parser.add_argument('--label_rate', type=float, default=0.5,
                  help='% of training labels to use when --use_labels is set.')
parser.add_argument('--planetoid_split', action='store_true',
                  help='use planetoid splits for Cora/Citeseer/Pubmed')

parser.add_argument('--random_splits',action='store_true',help='fixed_splits')

parser.add_argument('--edge_homo', type=float, default=0.0, help="edge_homo")


# GNN args
parser.add_argument('--hidden_dim',default=64, type=int,  help='Hidden dimension.')
parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                  help='Add a fully connected layer to the decoder.')
parser.add_argument('--input_dropout', type=float,default=0.2,  help='Input dropout rate.')
parser.add_argument('--dropout', type=float,default=0.4, help='Dropout rate.')
parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--decay', type=float,default=0.0001,  help='Weight decay for optimization')
parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                  help='apply sigmoid before multiplying by alpha')
parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
parser.add_argument('--block', default='constant_graph',type=str,  help='constant, mixed, attention, hard_attention')
parser.add_argument('--function',default='laplacian', type=str, help='laplacian, transformer, dorsey, GAT')
parser.add_argument('--use_mlp', type=bool,
                  help='Add a fully connected layer to the encoder.')
parser.add_argument('--add_source', dest='add_source', action='store_true',
                  help='If try get rid of alpha param and the beta*x0 source term')
parser.add_argument('--cgnn', dest='cgnn', action='store_true', help='Run the baseline CGNN model from ICML20')

parser.add_argument('--patience', type=int, default=100, help='Number of training patience per iteration.')

# ODE args
parser.add_argument('--time',default=3,  type=float, help='End time of ODE integrator.')
parser.add_argument('--augment', action='store_true',
                  help='double the length of the feature vector by appending zeros to stabilist ODE learning')
parser.add_argument('--method',default='ceuler',  type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
parser.add_argument('--step_size', type=float, default=1,
                  help='fixed step size when using fixed step solvers e.g. rk4')
parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                  help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                  help='use the adjoint ODE method to reduce memory footprint')
parser.add_argument('--adjoint_step_size', type=float, default=1,
                  help='fixed step size when using fixed step adjoint solvers e.g. rk4')
parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                  help="multiplier for adjoint_atol and adjoint_rtol")
parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
parser.add_argument("--max_nfe", type=int, default=100000000000,
                  help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
parser.add_argument("--no_early", action="store_true",
                  help="Whether or not to use early stopping of the ODE integrator when testing.")
parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
parser.add_argument("--max_test_steps", type=int, default=100,
                  help="Maximum number steps for the dopri5Early test integrator. "
                       "used if getting OOM errors at test time")

# Attention args
parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                  help='slope of the negative part of the leaky relu used in attention')
parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
parser.add_argument('--attention_dim', type=int, default=64,
                  help='the size to project x to before calculating att scores')
parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                  help='apply a feature transformation xW to the ODE')
parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                  help="multiply attention scores by edge weights before softmax")
parser.add_argument('--attention_type', type=str, default="scaled_dot",
                  help="scaled_dot,cosine_sim,pearson, exp_kernel")
parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

# regularisation args
parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

# rewiring args
parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                  help="obove this edge weight, keep edges when using threshold")
parser.add_argument('--gdc_avg_degree', type=int, default=64,
                  help="if gdc_threshold is not given can be calculated by specifying avg degree")
parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
parser.add_argument('--att_samp_pct', type=float, default=1,
                  help="float in [0,1). The percentage of edges to retain based on attention scores")
parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                  help='incorporate the feature grad in attention based edge dropout')
parser.add_argument("--exact", action="store_true",
                  help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")
parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
parser.add_argument('--new_edges', type=str, help="random, random_walk, k_hop")
parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
parser.add_argument('--threshold_type', type=str, default="topk_adj", help="topk_adj, addD_rvR")
parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")
parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
parser.add_argument('--rewire_KNN_epoch', type=int, default=5, help="frequency of epochs to rewire")
parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
parser.add_argument('--KNN_online', action='store_true', help='perform rewiring online')
parser.add_argument('--KNN_online_reps', type=int, default=4, help="how many online KNN its")
parser.add_argument('--KNN_space', type=str, default="pos_distance", help="Z,P,QKZ,QKp")
# beltrami args
parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
parser.add_argument('--fa_layer', action='store_true', help='add a bottleneck paper style layer with more edges')
parser.add_argument('--pos_enc_type', type=str, default="DW64",
                  help='positional encoder either GDC, DW64, DW128, DW256')
parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
parser.add_argument('--edge_sampling_add_type', type=str, default="importance",
                  help="random, ,anchored, importance, degree")
parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
parser.add_argument('--edge_sampling_space', type=str, default="attention",
                  help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
parser.add_argument('--symmetric_attention', action='store_true',
                  help='maks the attention symmetric for rewring in QK space')

parser.add_argument('--fa_layer_edge_sampling_rmv', type=float, default=0.8, help="percentage of edges to remove")
parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")

parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")
parser.add_argument('--alpha_ode', type=float, default=0.5, help='alpha_ode')
parser.add_argument('--runtime', type=int, default=10, help="runtime")
parser.add_argument('--seed', type=int, default=123, help="seed")


# for large non-homophilic datasets
parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
parser.add_argument('--train_prop', type=float, default=.5,
                    help='training label proportion')
parser.add_argument('--valid_prop', type=float, default=.25,
                    help='validation label proportion')
parser.add_argument('--sub_dataset', type=str, default='')
parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
parser.add_argument('--rocauc', action='store_true',
                        help='set the eval function to rocauc')
parser.add_argument('--sampling', action='store_true', help='use neighbor sampling')
parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')

parser.add_argument('--train_batch', type=str, default='graphsaint-rw', help='cluster,graphsaint-rw')
parser.add_argument('--no_mini_batch_test', action='store_true', help='whether to test on mini batches as well')
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--num_parts', type=int, default=100, help='number of partitions for partition batching')
parser.add_argument('--cluster_batch_size', type=int, default=1, help='number of clusters to use per cluster-gcn step')
parser.add_argument('--saint_num_steps', type=int, default=5, help='number of steps for graphsaint')
parser.add_argument('--test_num_parts', type=int, default=10, help='number of partitions for testing')

# gread args
parser.add_argument('--reaction_term', type=str, default='bspm', help='bspm, fisher, allen-cahn')
parser.add_argument('--beta_diag', type=eval, default=True)
# parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
parser.add_argument('--source_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) source')
# parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')


parser.add_argument('--num_terms', type=int, default=1, help="num_terms")
parser.add_argument('--exp_s', type=float, default=2.0,help='exp_s label proportion')
parser.add_argument('--heterophily', action='store_true', help='heterophily')
parser.add_argument('--distance', type=int,default=4, help='distance')
parser.add_argument('--poly_order', type=int,default=4, help='poly_order')
parser.add_argument('--learn_weight', type=bool,default=False,help='learn_weight')
parser.add_argument('--end_terms', type=int, default=1, help="end_terms")
parser.add_argument('--memory_k', type=int, default=0, help="memory_k")


#particle argument
parser.add_argument('--init_alpha', type=float, default=0.0, help="init value of coefficient of diffusion term")
parser.add_argument('--init_delta', type=float, default=-10.0, help="init value of coefficient of allen cahn term")
parser.add_argument('--beta', type=float, default=0.0, help="control attract or replusive force")
parser.add_argument('--channel_mixing', type=bool, default=False, help="control attract or replusive force")
parser.add_argument('--barrier', type=float, default=1.0, help="control attract or replusive force")

#flag argument
parser.add_argument('--step_size_adv', type=float, default=0.001, help="step_size_adv")
parser.add_argument('--attack_epoch', type=int, default=3, help="attack_epoch")
parser.add_argument('--attack', type=str, default=None, help="attack")
parser.add_argument('--pre_epoch', type=int, default=0, help="attack_epoch")

# multi term argument lambda_l1
parser.add_argument('--lambda_l1', type=float, default=0.01,help='lambda_l1 for l1 regularization')
parser.add_argument('--lr_weight', type=float, default=0.005,help='lr_weight')

parser.add_argument('--alpha_list',nargs='+', type=float, default=[0.5,1.0],help='alpha_list')

#difformer

parser.add_argument('--num_heads', type=int, default=4, help="num_heads")