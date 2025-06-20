import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import sys
import random
import argparse
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import (
    to_networkx
)
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import train_test_split_edges, to_undirected
import copy
import torch
import argparse
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid 
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from torch.utils.data import DataLoader
import wandb

from graphgps.utility.utils import mvari_str2csv
from baselines.gnn_utils import get_root_dir, get_logger, get_config_dir, Logger, init_seed, save_emb
from utils import PermIterator
from gnn_ogb_heart import init_seed
from automorphism import run_wl_test_and_group_nodes, compute_automorphism_metrics
from syn_real.gnn_utils  import evaluate_hits, evaluate_auc, evaluate_mrr
from syn_real.gnn_utils import (
    get_root_dir, 
    get_logger, 
    get_config_dir, 
    Logger, 
    init_seed
)
from functools import partial
from model import predictor_dict, convdict, GCN, DropEdge
from typing import Iterable
server = 'Horeka'
# python real_syn_automorphic.py --dataset Citeseer --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1 --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5 --batch_size 1024 
# python real_syn_automorphic.py --dataset Cora --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1 --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5 --batch_size 1024 
# python real_syn_automorphic.py --dataset ogbl-ddi --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1 --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5 --batch_size 1024 
# Cora
# intra ratio has no effect on Cora
# inter ratio has a big effect on Cora
# inter_ratios = [0.1] # Try also: 0.1–0.9
# intra_ratios = [0.5]    # Fixed intra ratio
# total_edges_list = [0.2, 1, 4, 7, 12, 18, 20, 28]*250 # Will be scaled × 10^3 

# Citeseer
# inter_ratios = [0.1] # Try also: 0.1–0.9
# intra_ratios = [0.5]    # Fixed intra ratio
# total_edges_list = [0.2, 1, 2, 3, 4, 5, 7, 8, 10, 14] * 1000

# DDI
# inter_ratios = [0.5]  # Try also: 0.1–0.9
# intra_ratios = [0.5]    
# total_edges_list =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 1

dir_path = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())
DATASET_PATH = '/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/baselines/dataset'

PT_LIST = [f"plots/Citeseer/processed_graph_inter0.5_intra0.5_edges1000_auto0.7200_norm1_0.7676.pt"]
    

def save_new_results(loggers, dataset, file_name='test_results_0.25_0.5.csv'):
    new_data = []
    
    for key in loggers.keys():
        if key == 'AUC':
            if len(loggers[key].results[0]) > 0:
                print(key)
                best_valid, best_valid_mean, mean_list, var_list, test_res = loggers[key].print_statistics()

                # Prepare row data
                new_data.append([dataset, key, best_valid, best_valid_mean, mean_list, var_list, test_res])
        
        # Merge and save the new results with the old ones
    load_and_merge_data(new_data, dataset, file_name)


def load_and_merge_data(new_data, dataset, num_node, file_name='test_results.csv'):
    try:
        # Try to read the existing CSV file
        old_data = pd.read_csv(file_name)
        
        # Merge the new data (convert new_data to a DataFrame)
        new_data_df = pd.DataFrame(new_data, columns=['dataset', 'Metric', 'Best Valid', 'Best Valid Mean', 'Mean List', 'Variance List', 'Test Result'])
        
        # Concatenate only the necessary columns
        merged_data = new_data_df
    
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame for the new data
        new_data_df = pd.DataFrame(new_data, columns=['dataset', 'Metric', 'Best Valid', 'Best Valid Mean', 'Mean List', 'Variance List', 'Test Result'])
        merged_data = new_data_df
    
    # Save the merged data back to the CSV file
    file_exists = os.path.exists(file_name)
    merged_data.to_csv(
        file_name,
        mode='a',                     # Append mode
        index=False,                  # Don't write row numbers
        header=not file_exists        # Write header only if file doesn't exist
    )
    # merged_data.to_csv(file_name, index=False)
    print(f'Merged data saved to {file_name}')


def remove_random_edges(graph_data, inter_ratio=0.5, intra_ratio=0.5, total_edges=1000):
    """
    Removes random edges from within and between two graph copies in a controlled way.

    Args:
        graph_data (Data): The graph structure (PyG format).
        inter_ratio (float): Fraction of edges to remove **between** the two graph copies.
        intra_ratio (float): Fraction of edges to remove **within** each graph copy.
        total_edges (int): Total number of edges to remove.

    Returns:
        Data: Graph with specified edges removed.
        torch.Tensor: Tensor of removed edges.
    """
    edge_index = graph_data.edge_index.cpu()
    num_nodes = graph_data.num_nodes // 2

    # Separate edges into intra and inter
    intra_mask = ((edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)) | \
                 ((edge_index[0] >= num_nodes) & (edge_index[1] >= num_nodes))
    inter_mask = ~intra_mask

    intra_edges = edge_index[:, intra_mask]
    inter_edges = edge_index[:, inter_mask]

    # Determine number of edges to remove
    num_inter_remove = int(total_edges * inter_ratio)
    num_intra_remove = total_edges - num_inter_remove

    # Sample edges to remove
    inter_remove_idx = np.random.choice(inter_edges.shape[1], min(num_inter_remove, inter_edges.shape[1]), replace=False)
    intra_remove_idx = np.random.choice(intra_edges.shape[1], min(num_intra_remove, intra_edges.shape[1]), replace=False)

    # Mask to keep edges
    keep_edges = torch.ones(edge_index.shape[1], dtype=torch.bool)

    # Build mapping from full edge index back to edge type masks
    intra_full_indices = torch.where(intra_mask)[0]
    inter_full_indices = torch.where(inter_mask)[0]

    keep_edges[intra_full_indices[intra_remove_idx]] = False
    keep_edges[inter_full_indices[inter_remove_idx]] = False

    # Updated edge index
    updated_edge_index = edge_index[:, keep_edges]
    removed_edges = edge_index[:, ~keep_edges]

    return Data(edge_index=updated_edge_index, num_nodes=graph_data.num_nodes, x=graph_data.x), removed_edges


def perturb_disjoint(graph_data, args, inter_ratio, intra_ratio, total_edges):
    """
    Run the experiment with the given parameters.
    
    Parameters:
        graph_data (torch_geometric.data.Data): The input graph data.
        args (argparse.Namespace): Arguments containing dataset name.
        inter_ratio (float): Fraction of edges to add between the two graph copies.
        intra_ratio (float): Fraction of edges to add within each graph copy.
        total_edges (int): Total number of random edges to add.
    """
    # Add random edges to the graph
    new_edges = 0
    if inter_ratio != 0 and intra_ratio != 0 and total_edges != 0:
        # updated_graph_data, new_edges = add_random_edges(graph_data, inter_ratio=inter_ratio, intra_ratio=intra_ratio, total_edges=total_edges)
        updated_graph_data, new_edges = remove_random_edges(graph_data, inter_ratio=inter_ratio, intra_ratio=intra_ratio, total_edges=total_edges)
        print(new_edges)
    else:
        updated_graph_data = graph_data
    # Convert to NetworkX graph for visualization
    G = to_networkx(updated_graph_data, to_undirected=True)
    num_nodes = updated_graph_data.num_nodes
    # print degree distribution 
    node_groups, node_labels = run_wl_test_and_group_nodes(updated_graph_data.edge_index, num_nodes=num_nodes, num_iterations=30)
    
    metrics_after, num_nodes, group_sizes = compute_automorphism_metrics(node_groups, num_nodes)
    metrics_after.update({'head': f'{args.dataset}_inter{inter_ratio}_intra{intra_ratio}_edges{total_edges}'})
    csv_path = f'plots/{args.dataset}/_Node_Merging.csv'
    file_exists = os.path.isfile(csv_path)
    df = pd.DataFrame([metrics_after])
    df.to_csv(csv_path, mode='a', index=False, header=not file_exists)
    print(df)
    
    # plot_group_size_distribution(group_sizes, args, f'plots/{args.dataset}/group_size_log1p{args.dataset}_inter{inter_ratio}_intra{intra_ratio}_edges{total_edges}.png')
    # plot_histogram_group_size_log_scale(group_sizes, metrics_after, args, f'plots/{args.dataset}/hist_group_size_log_{args.dataset}_inter{inter_ratio}_intra{intra_ratio}_edges{total_edges}.png')
    # plot_graph_visualization(updated_graph_data, node_labels, args,  f'plots/{args.dataset}/wl_test_{args.dataset}_vis_inter{inter_ratio}_intra{intra_ratio}_edges{total_edges}.png')
    print(f"Finished with inter_ratio={inter_ratio}, intra_ratio={intra_ratio}, total_edges={total_edges}")
    return updated_graph_data, metrics_after# , node_groups, node_labels, new_edges



def load_real_world_graph(dataset_name="Cora"):
    """
    Load a real-world graph dataset (e.g., Cora) from PyTorch Geometric.
    Args:
        dataset_name (str): The dataset name (default: "Cora").
    Returns:
        Data: PyTorch Geometric Data object.
    """
    if dataset_name in ['Cora', 'Citeseer', 'PubMed']:
        dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
        data = dataset[0]  
        # data.x = torch.eye(data.num_nodes, dtype=torch.float)
        # data.x = torch.rand(data.num_nodes, data.num_nodes)
        data.x = torch.diag(torch.arange(data.num_nodes).float())
    return data



def create_disjoint_graph(data):
    """
    Creates two disjoint copies of a real-world graph (e.g., Cora).
    Args:
        data (Data): PyG Data object representing the original graph.
    Returns:
        Data: PyG Data object representing the new merged graph.
    """
    num_nodes = data.num_nodes
    G = to_networkx(data, to_undirected=True)
    G2 = nx.relabel_nodes(G, lambda x: x + num_nodes)
    merged_graph = nx.compose(G, G2)
    merged_edge_index = torch.tensor(list(merged_graph.edges)).mT

    if hasattr(data, "x") and data.x is not None:
        merged_x = torch.cat([data.x, data.x], dim=0)
    merged_data = Data(edge_index=merged_edge_index, num_nodes=2 * num_nodes)
    if hasattr(data, "x") and data.x is not None:
        merged_data.x = merged_x  
    print(merged_data.x)
    return merged_data



def add_random_edges(graph_data, inter_ratio=0.5, intra_ratio=0.5, total_edges=1000):
    """
    Adds random edges between and within two graph copies in a controlled way.

    Args:
        graph_data (Data): The graph structure (PyG format).
        inter_ratio (float): Fraction of edges to add **between** the two graph copies.
        intra_ratio (float): Fraction of edges to add **within** each graph copy.
        total_edges (int): Total number of random edges to add.

    Returns:
        Data: Graph with additional edges.
    """
    num_nodes = graph_data.num_nodes // 2 
    inter_edges = int(total_edges * inter_ratio)
    intra_edges = total_edges - inter_edges  
    inter_edges_list = [
        (random.randint(0, num_nodes - 1), random.randint(num_nodes, 2 * num_nodes - 1))
        for _ in range(inter_edges)
    ]
    intra_edges_list = []
    for _ in range(intra_edges):
        copy = random.choice([0, 1]) 
        base_offset = num_nodes * copy 
        u, v = random.sample(range(base_offset, base_offset + num_nodes), 2)
        intra_edges_list.append((u, v))
        
    new_edges = torch.tensor(inter_edges_list + intra_edges_list, dtype=torch.long).T
    updated_edge_index = torch.cat([graph_data.edge_index, new_edges], dim=1)
    return Data(edge_index=updated_edge_index, num_nodes=graph_data.num_nodes, x=graph_data.x)


def parse_args():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--metric', type=str, default='AUC')
    parser.add_argument('--seed', type=int, default=9999)
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--inter_ratio', type=float, required=False, help='Inter ratio', default=0.5)
    parser.add_argument('--intra_ratio', type=float, required=False, help='Intra ratio', default=0.5)
    parser.add_argument('--total_edges', type=int, required=False, help='Total edges', default=1000)
    parser.add_argument('--use_valedges_as_input', action='store_true', help="whether to add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--epochs', type=int, default=50, help="number of epochs")
    parser.add_argument('--runs', type=int, default=3, help="number of repeated runs")
    parser.add_argument('--dataset', type=str, default='Cora')

    parser.add_argument('--testbs', type=int, default=8192, help="batch size for test")
    parser.add_argument('--maskinput', action="store_true", help="whether to use target link removal")
    parser.add_argument('--batch_size', type=int, default=256)
    
    parser.add_argument('--mplayers', type=int, default=3, help="number of message passing layers")
    parser.add_argument('--nnlayers', type=int, default=3, help="number of mlp layers")
    parser.add_argument('--hiddim', type=int, default=512, help="hidden dimension")
    parser.add_argument('--ln', action="store_true", help="whether to use layernorm in MPNN")
    parser.add_argument('--lnnn', action="store_true", help="whether to use layernorm in mlp")
    parser.add_argument('--res', action="store_true", help="whether to use residual connection")
    parser.add_argument('--jk', action="store_true", help="whether to use JumpingKnowledge connection")
    parser.add_argument('--gnndp', type=float, default=0, help="dropout ratio of gnn")
    parser.add_argument('--xdp', type=float, default=0.1, help="dropout ratio of gnn")
    parser.add_argument('--tdp', type=float, default=0, help="dropout ratio of gnn")
    parser.add_argument('--gnnedp', type=float, default=0, help="edge dropout ratio of gnn")
    parser.add_argument('--predp', type=float, default=0, help="dropout ratio of predictor")
    parser.add_argument('--preedp', type=float, default=0.1, help="edge dropout ratio of predictor")
    parser.add_argument('--gnnlr', type=float, default=0.001, help="learning rate of gnn")
    parser.add_argument('--prelr', type=float, default=0.001, help="learning rate of predictor")
    # detailed hyperparameters
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--increasealpha", action="store_true")
    parser.add_argument('--kill_cnt', dest='kill_cnt', default=5, type=int, help='early stopping')
    parser.add_argument('--splitsize', type=int, default=-1, help="split some operations inner the model. Only speed and GPU memory consumption are affected.")

    # parameters used to calibrate the edge existence probability in NCNC
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument("--learnpt", action="store_true")

    # For scalability, NCNC samples neighbors to complete common neighbor. 
    parser.add_argument('--trndeg', type=int, default=-1, help="maximum number of sampled neighbors during the training process. -1 means no sample")
    parser.add_argument('--tstdeg', type=int, default=-1, help="maximum number of sampled neighbors during the test process")
    # NCN can sample common neighbors for scalability. Generally not used. 
    parser.add_argument('--cndeg', type=int, default=-1)
    
    # predictor used, such as NCN, NCNC
    parser.add_argument('--predictor', choices=predictor_dict.keys(), default='cn1')
    parser.add_argument("--depth", type=int, default=1, help="number of completion steps in NCNC")
    # gnn used, such as gin, gcn.
    parser.add_argument('--model', choices=convdict.keys(), default='gcn')

    parser.add_argument('--save_gemb', action="store_true", help="whether to save node representations produced by GNN")
    parser.add_argument('--load', type=str, help="where to load node representations produced by GNN")
    parser.add_argument("--loadmod", action="store_true", help="whether to load trained models")
    parser.add_argument("--savemod", action="store_true", help="whether to save trained models")
    
    parser.add_argument("--savex", action="store_true", help="whether to save trained node embeddings")
    parser.add_argument("--loadx", action="store_true", help="whether to load trained node embeddings")

    
    # not used in experiments
    parser.add_argument('--cnprob', type=float, default=0)
    args = parser.parse_args()
    # print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    # print('use_val_edge:', args.use_valedges_as_input)
    # print('use_hard_negative: ',args.use_hard_negative)
    # print(args)
    return args
    
    
def randomsplit(data, val_ratio: float = 0.05, test_ratio: float = 0.15):
    def removerepeated(ei):
        ei = to_undirected(ei)
        ei = ei[:, ei[0]<ei[1]]
        return ei
    data.num_nodes = data.x.shape[0]
    
    data = train_test_split_edges(data, test_ratio, test_ratio)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    
    num_val = int(data.val_pos_edge_index.shape[1] * val_ratio/test_ratio)
    data.val_pos_edge_index = data.val_pos_edge_index[:, torch.randperm(data.val_pos_edge_index.shape[1])]
    split_edge['train']['edge'] = removerepeated(
        torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
    split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
    split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
    split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
    split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()
    return split_edge

    
    
def data2dict(data, splits, dataset) -> dict:
    #TODO test with all ogbl-datasets, start with collab
    if dataset in ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'ogbl-ddi']:
        datadict = {}
        datadict.update({'adj': data.adj_t})
        datadict.update({'train_pos': splits['train']['edge']})
        # datadict.update({'train_neg': splits['train']['edge_neg']})
        datadict.update({'valid_pos': splits['valid']['edge']})
        datadict.update({'valid_neg': splits['valid']['edge_neg']})
        datadict.update({'test_pos': splits['test']['edge']})
        datadict.update({'test_neg': splits['test']['edge_neg']})   
        datadict.update({'train_val': torch.cat([splits['valid']['edge'], splits['train']['edge']])})
        datadict.update({'x': data.x}) 
    else:
        raise ValueError('dataset not supported')
    return datadict



def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [1, 5, 10, 20, 50]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)
    # result_hit = {}
    for K in k_list:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])
    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])
    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)
    
    # result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_val_pred.repeat(pos_train_pred.size(0), 1))
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1))
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1))
    for k in result_mrr_val.keys():
        result[k] = (0, result_mrr_val[k], result_mrr_test[k])

    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])
    # print(result.keys())
    return result


def get_graph_statistics(G, graph_name="Graph"):
    """Calculate and return statistics of a NetworkX graph."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    
    # Compute degree statistics
    degrees = [deg for node, deg in G.degree()]
    avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0
    min_degree = min(degrees) if degrees else None
    max_degree = max(degrees) if degrees else None
    
    stats = {
        "Graph Name": graph_name,
        "Number of Nodes": num_nodes,
        "Number of Edges": num_edges,
        "Density": density,
        "Average Degree": avg_degree,
        "Min Degree": min_degree,
        "Max Degree": max_degree,
    }
    return stats


def train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None,
          device='cpu'):
    
    if alpha is not None:
        predictor.setalpha(alpha)
    
    model.train()
    predictor.train()

    pos_train_edge =  split_edge['train']['edge'].t()
    negedge =  split_edge['valid']['edge_neg']

    total_loss = []
    x = data['x'].to(device)
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    for perm in PermIterator(
            adjmask.device, adjmask.shape[0], batch_size
    ):
        optimizer.zero_grad()
        num_nodes = data['x'].shape[0]
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            
            adj = SparseTensor.from_edge_index(tei.t(), sparse_sizes=(num_nodes, num_nodes)).to(device)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data['adj'].to(device)
            
        h = model(x, adj)
        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h,
                                                adj,
                                                edge,
                                                cndropprobs=cnprobs)

        pos_losss = -F.logsigmoid(pos_outs).mean()
        negedge = torch.randint(0, x.size(0), (pos_train_edge.size()))
        edge = negedge[:, perm]
        neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_losss + pos_losss
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    
    return total_loss


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size,
         use_valedges_as_input, device):
    model.eval()
    predictor.eval()

    pos_train_edge = split_edge['train']['edge'].to(data['x'].device)
    pos_valid_edge = split_edge['valid']['edge'].to(data['x'].device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data['x'].device)
    pos_test_edge = split_edge['test']['edge'].to(data['x'].device)
    neg_test_edge = split_edge['test']['edge_neg'].to(data['x'].device)
    
    x = data['x'].to(device)
    adj = data['adj'].to(device)
    # row = torch.from_numpy(adj.tocoo().row).long()
    # col = torch.from_numpy(adj.tocoo().col).long()
    # value = torch.from_numpy(adj.tocoo().data).float()
    # adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(data.num_nodes, data.num_nodes))
    h = model(x, adj)

    pos_train_pred = torch.cat([
        predictor(h, adj, pos_train_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], batch_size, False)
    ], dim=0)

    pos_valid_pred = torch.cat([
        predictor(h, adj, pos_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False)
    ],  dim=0)
    neg_valid_pred = torch.cat([
        predictor(h, adj, neg_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], batch_size, False)
    ],  dim=0)
    if use_valedges_as_input:
        adj = data.full_adj_t
        h = model(data.x, adj)

    pos_test_pred = torch.cat([
        predictor(h, adj, pos_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False)
    ],  dim=0)

    neg_test_pred = torch.cat([
        predictor(h, adj, neg_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                 batch_size, False)
    ],  dim=0)

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    results = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    return results



def run_training_pipeline(data, metrics, inter, intra, total_edges, args):
    
    data = copy.deepcopy(data)
    G = to_networkx(data)
    stats = get_graph_statistics(G, graph_name=args.dataset)
    print(stats)
    data.adj_t = SparseTensor.from_edge_index(
        data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)
    ).to_symmetric().coalesce()
    split_edge = randomsplit(data)
    print("Dataset split:")
    for key1 in split_edge:
        for key2 in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])
    data.edge_index = to_undirected(split_edge["train"]["edge"].t())
    data = data2dict(data, split_edge, args.dataset)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@5': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs),
    }

    # import itertools
    # hyperparams = {
    #     'batch_size': [2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12],
    #     'lr': [0.01, 0.001, 0.0001],
    # }
    if args.dataset == 'Cora': 
        args.batch_size = 1024
        args.lr = 0.01
    elif args.dataset == 'Citeseer':
        args.batch_size = 1024
        args.lr = 0.001
    elif args.dataset == 'ogbl-ddi':
        args.batch_size = 2**5
        args.lr = 0.00001

    args.name_tag = (
        f'{args.dataset}_'
        f'Orbits_{metrics["Number of Unique Groups (C_auto)"]:.2f}_'
        f'ArScore_{metrics["automorphism_score"]:.2f}'
        f'{args.model}_'
        f'{args.predictor}_'
        f'inter{inter:.2f}_'
        f'intra{intra:.2f}_'
        f'total{total_edges:.0f}_'
        f'drop_{args.preedp}_'
        f'use_wl_{args.use_xlin}'
    )

    # for batch_size, lr in itertools.product(hyperparams['batch_size'], hyperparams['lr']):
    #     args.batch_size = batch_size
    #     args.lr = lr
    eval_metric = args.metric
    predfn = predictor_dict[args.predictor]
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)

    predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                        args.predp, args.preedp, args.lnnn).to(device)
    evaluator = Evaluator(name='ogbl-ppa')
    for run in range(0, args.runs):
        init_seed(run)
        num_features = data['x'].shape[1]
        model = GCN(num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, -1,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, 
                    taildropout=args.tdp, noinputlin=args.loadx).to(device)
        
        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)

        import itertools
        hyperparams = {
            'batch_size': [256],
            'lr': [0.001],
            'testbs': [256] # [2048, 4096, 8192, 16384, 32768]
        }
        best_score = 0
        kill_cnt = 0
        for batch_size, lr, testbs in itertools.product(hyperparams['batch_size'], hyperparams['lr'], hyperparams['testbs']):
            args.batch_size = batch_size
            args.gnnlr = lr
            args.prelr = lr
            args.testbs = testbs 

            name_tag = f"{args.name_tag}_{args.model}_{args.testbs}_{args.predictor}_{args.gnnedp}_{args.predp}_{args.predp}_{args.tdp}_{args.xdp}_{args.hiddim}_{args.mplayers}_{args.nnlayers}_{args.use_xlin}"
            wandb.init(project=f"{args.dataset}_tab2", name=name_tag, config=vars(args))
        
            optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
            {'params': predictor.parameters(), 'lr': args.prelr}])
            
            for epoch in range(1, 1 + args.epochs):
                alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
                # t1 = time.time()
                loss = train(model, predictor, data, split_edge, optimizer,
                            args.batch_size, args.maskinput, [], alpha, device)
                wandb.log({'train_loss': loss.item()}, step = epoch)
                
                if epoch % 1 == 0:
                    results = test(model, predictor, data, split_edge, evaluator,
                                args.testbs, args.use_valedges_as_input, device)
                    for key, result in results.items():
                        try:
                            loggers[key].add_result(run, result)
                            wandb.log({f"Metrics/{key}": result[-1]}, step=epoch)
                        except:
                            pass
                            
                    if epoch % 1 == 0:
                        for key, result in results.items():
                            if key == 'AUC':
                                train_hits, valid_hits, test_hits = result
                                log_print.info(
                                    f'Run: {run + 1:02d}, '
                                    f'Epoch: {epoch:02d}, '
                                    f'Loss: {loss:.4f}, '
                                    f'Train: {100 * train_hits:.2f}%, '
                                    f'Valid: {100 * valid_hits:.2f}%, '
                                    f'Test: {100 * test_hits:.2f}%')

                    r = torch.tensor(loggers[eval_metric].results[run])
                    best_valid_current = round(r[:, 1].max().item(), 4)
                    best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

                    print(eval_metric)
                    log_print.info(f'best valid: {100*best_valid_current:.2f}%, '
                                    f'best test: {100*best_test:.2f}%')
                    
                    if len(loggers['AUC'].results[run]) > 0:
                        r = torch.tensor(loggers['AUC'].results[run])
                        best_valid_auc = round(r[:, 1].max().item(), 4)
                        best_test_auc = round(r[r[:, 1].argmax(), 2].item(), 4)
                        
                        print('AUC')
                        log_print.info(f'best valid: {100*best_valid_auc:.2f}%, '
                                    f'best test: {100*best_test_auc:.2f}%')
                
                    print('---')
                    
                    if best_valid_current > best_score:
                        best_score = best_valid_current
                        kill_cnt = 0
                    else:
                        kill_cnt += 1
                        
                        if kill_cnt > args.kill_cnt: 
                            print("Early Stopping!!")
                            break

            wandb.finish()
            
            for key in loggers.keys():
                if len(loggers[key].results[0]) > 0:
                    if key in ['AUC', 'MRR', 'Hits@1']:
                        try: 
                            best_valid, best_valid_mean, mean_list, var_list, test_res = loggers[key].print_statistics()
                            print(best_valid)
                            print(best_valid_mean)
                            print(mean_list)
                            print(var_list)
                            print(test_res)
                        except:
                            pass

    save_new_results(loggers, name_tag, file_name=f'{args.dataset}_{args.model}_{args.predictor}test_results_0.25_0.5.csv')
    result_all_run = {} 
    save_dict = {}
    for key in loggers.keys():
        if key in ['Hits@1', 'AUC', 'AP', 'MRR']:
            best_metric, best_valid_mean, mean_list, var_list, test_res = loggers[key].print_statistics()
            if key == eval_metric:
                best_metric_valid_str = best_metric
            if key == 'AUC':
                best_auc_valid_str = best_metric
            result_all_run[key] = [mean_list, var_list]
            save_dict[key] = test_res
            print(save_dict)
    print(best_metric_valid_str + ' ' + best_auc_valid_str)
    print(args.name_tag)
    mvari_str2csv(args.name_tag, save_dict, f'syn_random_{args.dataset}_{args.predictor}tuned.csv')


def main():
    args = parse_args()
    init_seed(args.seed)

    if os.path.exists(f'plots/{args.dataset}') == False:
        os.makedirs(f'plots/{args.dataset}')

    csv_path = f'plots/{args.dataset}/_Node_Merging.csv'
    file_exists = os.path.isfile(csv_path)
    original_data = load_real_world_graph(args.dataset)
    perturb_disjoint(original_data, args, 0, 0, 0)
    
    disjoint_graph = create_disjoint_graph(original_data)
    disjoint_graph, metrics = perturb_disjoint(disjoint_graph, args, 0, 0, 0)
    run_training_pipeline(disjoint_graph, metrics, 0, 0, 0, args)
    
    if args.dataset == 'Cora':
        # Cora
        inter_ratios = [0.1]   
        intra_ratios =  [0.5]
        total_edges_list = [0.2, 1, 4, 7, 12, 18, 20, 28]
        multi_factor = 250

    elif args.dataset == 'Citeseer':
        # Citeseer
        inter_ratios = [0.1] 
        intra_ratios = [0.5] 
        total_edges_list = [14] #[0.2, 1, 2, 3, 4, 5, 7, 8, 10, 14]
        multi_factor = 100
        
    elif args.dataset == 'ogbl-ddi':
        # DDI
        inter_ratios = [0.5] 
        intra_ratios = [0.5]    
        total_edges_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # np.round(np.arange(0, 40, 2), 2).tolist()# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
        multi_factor = 1
        
    for inter in inter_ratios:
        for intra in intra_ratios:
            for edge_factor in total_edges_list:
                total_edges = int(edge_factor * multi_factor)
                data, metrics = perturb_disjoint(disjoint_graph, args, inter, intra, total_edges)
                run_training_pipeline(data, metrics, inter, intra, total_edges, args)

if __name__ == "__main__":
    main()
