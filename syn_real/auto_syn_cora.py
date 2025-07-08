import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
    to_networkx,
    from_networkx
)

from baselines.gnn_utils import (GCN, 
                                 GAT, 
                                 SAGE, 
                                 GIN, 
                                 MF, 
                                 DGCNN, 
                                 GCN_seal, 
                                 SAGE_seal, 
                                 DecoupleSEAL, 
                                 mlp_score, 
                                 dot_product, 
                                 ChebGCN, 
                                 MixHopGCN)
import pandas as pd
from torch_geometric.utils import (train_test_split_edges, 
								   to_undirected)
import copy
from torch_geometric.datasets import Planetoid 
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from torch.utils.data import DataLoader
import wandb
from syn_real.gnn_utils  import (evaluate_hits, 
								 evaluate_auc, 
								 evaluate_mrr)

from baselines.gnn_utils import (get_root_dir, 
                                 get_logger, 
                                 get_config_dir, 
                                 Logger, 
                                 init_seed, 
                                 save_emb)

from graphgps.utility.utils import mvari_str2csv
from syn_real.gnn_ogb_heart import init_seed
from syn_real.automorphism import (run_wl_test_and_group_nodes, 
                                   count_automorphic_edges, 
                                   compute_automorphism_metrics)
from syn_real.auto_operation import (create_disjoint_graph, 
									 add_random_edges)
from syn_real.syn_datagen import (get_k_hop_subgraph_from_dataset, 
                                add_node_connected_to_node,
                                analyze_automorphisms)

# python real_syn_automorphic.py --data_name Citeseer --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1 --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5 --batch_size 1024 
# python real_syn_automorphic.py --data_name Cora --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1 --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5 --batch_size 1024 
# python real_syn_automorphic.py --data_name ogbl-ddi --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1 --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5 --batch_size 1024 
# Cora
# intra ratio has no effect on Cora
# inter ratio has a big effect on Cora
# inter_ratios = [0.1] # Try also: 0.1–0.9
# intra_ratios = [0.5]    # Fixed intra ratio
# total_edges_list = [0.2, 1, 4, 7, 12, 18, 28]*250 # Will be scaled × 10^3



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
    if inter_ratio != 0 and intra_ratio != 0 and total_edges != 0:
        updated_graph_data = add_random_edges(graph_data, 
                                              inter_ratio=inter_ratio, 
                                              intra_ratio=intra_ratio, 
                                              total_edges=total_edges)
    else:
        updated_graph_data = graph_data
    # Convert to NetworkX graph for visualization
    G = to_networkx(updated_graph_data, to_undirected=True)
    num_nodes = updated_graph_data.num_nodes
    # print degree distribution 
    node_groups, node_labels, new_labels = run_wl_test_and_group_nodes(updated_graph_data.edge_index, num_nodes=num_nodes, num_iterations=30)
    intra_orbit_edges, inter_orbit_edges = count_automorphic_edges(G, node_labels)
    metrics_after, num_nodes, group_sizes = compute_automorphism_metrics(node_groups, num_nodes)
    # metrics_after.update({'head': f'{args.data_name}_inter{inter_ratio}_intra{intra_ratio}_edges{total_edges}'})
    # csv_path = f'plots/{args.data_name}/_Node_Merging.csv'
    # file_exists = os.path.isfile(csv_path)
    df = pd.DataFrame([metrics_after])
    # df.to_csv(csv_path, mode='a', index=False, header=not file_exists)
    print(df)
    
    # plot_group_size_distribution(group_sizes, args, f'plots/{args.data_name}/group_size_log1p{args.data_name}_inter{inter_ratio}_intra{intra_ratio}_edges{total_edges}.png')
    # plot_histogram_group_size_log_scale(group_sizes, metrics_after, args, f'plots/{args.data_name}/hist_group_size_log_{args.data_name}_inter{inter_ratio}_intra{intra_ratio}_edges{total_edges}.png')
    # plot_graph_visualization(updated_graph_data, node_labels, args,  f'plots/{args.data_name}/wl_test_{args.data_name}_vis_inter{inter_ratio}_intra{intra_ratio}_edges{total_edges}.png')
    print(f"Finished with inter_ratio={inter_ratio}, intra_ratio={intra_ratio}, total_edges={total_edges}")
    return updated_graph_data, metrics_after 

    
    
# --- 1️⃣ Load Real-World Graph (Cora) ---
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
    elif dataset_name.startswith('ogbl'):
        pass
    return data


def plot_group_size_distribution(group_sizes, args, file_name):
    """ 
    Plots the group size distribution with log-log scaling.
    
    Parameters:
        group_sizes (list): Sizes of automorphism groups.
        args (argparse.Namespace): Arguments containing dataset name.
    """
    # Not readable
    # plt.figure()
    # plt.plot(group_sizes)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel("Group Index (log scale)")
    # plt.ylabel("Group Size (log scale)")
    # plt.title("Group Size Distribution (Log-Log Scale)")
    # plt.savefig(f'plots/{args.data_name}/group_size_{args.data_name}.png')
    # plt.close()

    plt.figure()
    plt.plot(np.log1p(group_sizes))
    plt.xlabel("Group Index (log scale)")
    plt.ylabel("Group Size (log scale)")
    plt.title("Group Size Distribution (Log-Log Scale)")
    plt.savefig(file_name)
    plt.close()
    

def plot_histogram_group_size(group_sizes, metrics_before, args):
    """ 
    Plots a histogram of group sizes.
    
    Parameters:
        group_sizes (list): Sizes of automorphism groups.
        metrics_before (dict): Dictionary containing WL test metrics.
        args (argparse.Namespace): Arguments containing dataset name.
    """
    plot_dir = f'plots/{args.data_name}'
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    counts, bins, _ = plt.hist(group_sizes, bins=20, edgecolor='black', alpha=0.75, density=True)
    counts = counts * 100 * np.diff(bins)
    plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', alpha=0.75)
    plt.xlabel("Group Size")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Group Sizes {metrics_before['A_r_norm_1']}")
    save_path = f'{plot_dir}/hist_group_size_{args.data_name}.png'
    plt.savefig(save_path)
    plt.close()
    # print(f"Saved to {save_path}")
    # print(f"Automorphism fraction before adding random edges: {metrics_before}")



def plot_graph_visualization(graph_data, node_labels, args, save_path):
    """ 
    Plots a general visualization of the graph using WL-based node coloring.
    
    Parameters:
        graph_data (torch_geometric.data.Data): The input graph data.
        node_labels (list or array): Node labels for coloring.
        args (argparse.Namespace): Arguments containing dataset name.
    """
    plt.figure(figsize=(6, 6))
    G = to_networkx(graph_data, to_undirected=True)
    nx.draw(G, node_size=10, font_size=8, cmap='Set1', node_color=node_labels, edge_color="gray")
    plt.title("Graph Visualization with WL-based Node Coloring")
    plt.savefig(save_path)
    plt.close()


def plot_histogram_group_size_log_scale(group_sizes, metrics_before, args, save_path):
    """ 
    Plots a histogram of group sizes with log scale on both axes.
    
    Parameters:
        group_sizes (list): Sizes of automorphism groups.
        metrics_before (dict): Dictionary containing WL test metrics.
        args (argparse.Namespace): Arguments containing dataset name.
    """

    plt.figure(figsize=(6, 4))
    counts, bins, _ = plt.hist(group_sizes, bins=20, edgecolor='black', alpha=0.75, density=True)
    counts = counts * 100 * np.diff(bins)
    plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', alpha=0.75)
    plt.yscale('log') 
    plt.xlabel("Group Size (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"Histogram of Group Sizes {metrics_before['A_r_norm_1']}")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved to {save_path}")
    print(f"Automorphism fraction before adding random edges: {metrics_before}")
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default="Cora")
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    parser.add_argument('--pt_path', default=f"plots/Citeseer/processed_graph_inter0.5_intra0.5_edges1000_auto0.7200_norm1_0.7676.pt",
                        type=str)
    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--gnnout_hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--eval_metric', type=str, default='AUC')
    
    ### train setting
    parser.add_argument('--batch_size', type=int, default=2**8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=20,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)

    # 
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--remove_edge_aggre', action='store_true', default=False)
    parser.add_argument('--name_tag', type=str, default='')
    parser.add_argument('--gin_mlp_layer', type=int, default=2)
    parser.add_argument('--gat_head', type=int, default=1)
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')
    parser.add_argument('--test_batch_size', type=int, default=1024 * 64) 
    parser.add_argument('--use_hard_negative', default=False, action='store_true')
    parser.add_argument('--wandb_log', default=True, action='store_true')
    parser.add_argument('--metric', type=str, default='AUC')
    parser.add_argument('--inter_ratio', type=float, required=False, help='Inter ratio', default=0.5)
    parser.add_argument('--intra_ratio', type=float, required=False, help='Intra ratio', default=0.5)
    parser.add_argument('--total_edges', type=int, required=False, help='Total edges', default=1000)
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

    
    
def data2dict(data, splits, data_name) -> dict:
    #TODO test with all ogbl-datasets, start with collab
    if data_name in ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'ogbl-ddi']:
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
        raise ValueError('data_name not supported')
    return datadict


def train(model, score_func, train_pos, x, optimizer, batch_size):
    model.train()
    score_func.train()

    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0

    for perm in DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        num_nodes = x.size(0)
        ######################### remove loss edges from the aggregation
        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
        train_edge_mask = train_pos[mask].transpose(1,0)

        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)

        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        x = x.to(train_pos.device)
        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
        h = model(x, adj)
        
        edge = train_pos[perm].t()
        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        # Just do some trivial random sampling.
        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples



@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size):
    # input_data  = input_data.transpose(1, 0)
    # with torch.no_grad():
    preds = []
    for perm  in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()
        preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
    pred_all = torch.cat(preds, dim=0)
    return pred_all



@torch.no_grad()
def test(model, score_func, data, x, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    score_func.eval()
    # adj_t = adj_t.transpose(1,0)
    h = model(x, data['adj'].to(x.device))
    # print(h[0][:10])
    x = h
    pos_train_pred = test_edge(score_func, data['train_val'], h, batch_size)
    neg_valid_pred = test_edge(score_func, data['valid_neg'], h, batch_size)
    pos_valid_pred = test_edge(score_func, data['valid_pos'], h, batch_size)
    pos_test_pred = test_edge(score_func, data['test_pos'], h, batch_size)
    neg_test_pred = test_edge(score_func, data['test_neg'], h, batch_size)
    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)
    # print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x.cpu()]
    return result, score_emb



def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [1, 10, 20, 50, 100]
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


@torch.no_grad()
def test(model, score_func, data, x, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    score_func.eval()
    # adj_t = adj_t.transpose(1,0)
    h = model(x, data['adj'].to(x.device))
    # print(h[0][:10])
    x = h
    pos_train_pred = test_edge(score_func, data['train_val'], h, batch_size)
    neg_valid_pred = test_edge(score_func, data['valid_neg'], h, batch_size)
    pos_valid_pred = test_edge(score_func, data['valid_pos'], h, batch_size)
    pos_test_pred = test_edge(score_func, data['test_pos'], h, batch_size)
    neg_test_pred = test_edge(score_func, data['test_neg'], h, batch_size)
    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)
    # print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x.cpu()]
    # print(result.keys())
    return result, score_emb



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


def run_training_pipeline(data, metrics, par1, par2, par3, args):
    
    data = copy.deepcopy(data)
    G = to_networkx(data)
    stats = get_graph_statistics(G, graph_name=args.data_name)
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
    data = data2dict(data, split_edge, args.data_name)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    x = data['x'].to(device)
    if args.cat_n2v_feat:
        print('cat n2v embedding!!')
        n2v_emb = torch.load(os.path.join(get_root_dir(), 'dataset', args.data_name + '-n2v-embedding.pt'))
        x = torch.cat((x, n2v_emb), dim=-1)
    train_pos = data['train_pos'].to(x.device)
    node_num = x.size(0)
    input_channel = x.size(1)
    
    if args.gnn_model == 'MixHopGCN':
        args.num_layers = 1
        
    model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.dropout, 
                    mlp_layer=args.gin_mlp_layer, head=args.gat_head, 
                    node_num=node_num, cat_node_feat_mf=args.cat_node_feat_mf,  
                    data_name=args.data_name).to(device)
    
    if args.gnn_model == 'MixHopGCN':
        args.hidden_channels = 3 * args.hidden_channels
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)
    
    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    loggers = {
        key: Logger(args.runs) for key in [
            'Hits@1', 'Hits@10', 'Hits@20', 'Hits@50', 'Hits@100', 
            'MRR', 'AUC', 'AP', 'mrr_hit1', 'mrr_hit3', 'mrr_hit10', 
            'mrr_hit20', 'mrr_hit50', 'mrr_hit100'
        ]
    }

    # import itertools
    # hyperparams = {
    #     'batch_size': [2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12],
    #     'lr': [0.01, 0.001, 0.0001],
    # }
    if args.data_name == 'Cora': 
        args.batch_size = 1024
        args.lr = 0.001
    elif args.data_name == 'Citeseer':
        args.batch_size = 1024
        args.lr = 0.001
    elif args.data_name == 'ogbl-ddi':
        args.batch_size = 2**5
        args.lr = 0.00001

    args.name_tag = args.data_name+ f"_ArScore_{metrics:.2f}_{args.gnn_model}_{args.score_model}_inter{par1:.2f}_intra{par2:.2f}_total{par3:.0f}"
    
    # for batch_size, lr in itertools.product(hyperparams['batch_size'], hyperparams['lr']):
    #     args.batch_size = batch_size
    #     args.lr = lr
    
    for run in range(args.runs):
        if args.wandb_log:
            wandb.init(
                project=f"{args.data_name}_4",
                name=f"{args.name_tag}_{par1}_{par2}_{par3}"#{args.name_tag}_{args.gnn_model}_{args.score_model}_{args.runs}"
            )
            wandb.config.update(args)
        print(f'#################################          Run {run}          #################################')
        seed = args.seed if args.runs == 1 else run
        print('seed:', seed)
        init_seed(seed)
        save_path = os.path.join(
            args.output_dir,
            f'lr{args.lr}_drop{args.dropout}_l2{args.l2}_numlayer{args.num_layers}_'
            f'numPredlay{args.num_layers_predictor}_numGinMlplayer{args.gin_mlp_layer}_'
            f'dim{args.hidden_channels}_best_run_{seed}'
        )
        model.reset_parameters()
        score_func.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(score_func.parameters()),
            lr=args.lr,
            weight_decay=args.l2
        )
        best_valid, best_test, kill_cnt, step = 0, 0, 0, 0
        for epoch in range(1, args.epochs + 1):
            loss = train(model, score_func, train_pos, x, optimizer, args.batch_size)
            if epoch % args.eval_steps == 0:
                results_rank, score_emb = test(
                    model, score_func, data, x,
                    evaluator_hit, evaluator_mrr, args.batch_size
                )

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)
                    if loss > 20: 
                        continue
                    wandb.log({'train_loss': loss}, step=epoch) if args.wandb_log else None
                    wandb.log({f"Metrics/{key}": result[-1]}, step=epoch) if args.wandb_log else None
                    step += 1
                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()
                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if args.save:
                        save_emb(score_emb, save_path)
                else:
                    kill_cnt += 1
                    if kill_cnt > args.kill_cnt:
                        print("Early Stopping!!")
                        break
        wandb.finish() if args.wandb_log else None
                
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
    mvari_str2csv(args.name_tag, save_dict, f'results/syn_{args.data_name}_{args.gnn_model}tuned.csv')

def remove_random_edges(G: nx.Graph, num_edges: int, protected_edges: set):
    """
    Randomly removes num_edges from G excluding those in protected_edges.
    """
    all_edges = set(G.edges)
    removable_edges = list(all_edges - protected_edges)

    if len(removable_edges) < num_edges:
        raise ValueError("Not enough removable edges to delete.")

    to_remove = random.sample(removable_edges, num_edges)
    G.remove_edges_from(to_remove)
    return G

def attach_star_graph(G_orig: nx.Graph, N: int, ig: int):
    G_combined = G_orig.copy()
    offset = max(G_combined.nodes) + 1 if len(G_combined.nodes) > 0 else 0

    G_star = nx.star_graph(N - 1)
    mapping = {i: i + offset for i in G_star.nodes}
    G_star = nx.relabel_nodes(G_star, mapping)
    center_node_new = mapping[0]

    # Add star to combined graph
    G_combined.add_nodes_from(G_star.nodes(data=True))
    G_combined.add_edges_from(G_star.edges(data=True))

    # Add connecting edge to original graph
    G_combined.add_edge(center_node_new, ig)

    # Track star nodes (including connecting edge)
    star_nodes = set(G_star.nodes)
    star_edges = set(G_star.edges)
    star_edges.add((center_node_new, ig))  # include connection edge

    # Convert to PyG
    data = from_networkx(G_combined)
    data.x = torch.ones((data.num_nodes, 1433))
    return G_combined, data, star_edges



def add_random_edges(G: nx.Graph, num_edges: int, protected_edges: set = set(), protected_nodes: set = set()):
    """
    Randomly adds num_edges to G, avoiding protected_edges and optionally protected_nodes.
    
    Args:
        G (nx.Graph): The graph to modify.
        num_edges (int): Number of edges to add.
        protected_edges (set): Edges to avoid adding (e.g., existing or forbidden).
        protected_nodes (set): Nodes to avoid using in added edges.
        
    Returns:
        G (nx.Graph): Graph with added edges.
    """
    import itertools

    existing_edges = set(G.edges)
    all_nodes = list(G.nodes)
    
    # Generate all possible node pairs (i < j) not in protected sets
    candidate_edges = {
        (u, v)
        for u, v in itertools.combinations(all_nodes, 2)
        if (u, v) not in existing_edges
        and (v, u) not in existing_edges
        and (u, v) not in protected_edges
        and (v, u) not in protected_edges
        and u not in protected_nodes
        and v not in protected_nodes
    }

    if len(candidate_edges) < num_edges:
        raise ValueError("Not enough candidate edges to add.")

    new_edges = random.sample(list(candidate_edges), num_edges)
    G.add_edges_from(new_edges)
    return G


def main():
    args = parse_args()
    init_seed(args.seed)

    if os.path.exists(f'plots/{args.data_name}') == False:
        os.makedirs(f'plots/{args.data_name}')

    csv_path = f'plots/{args.data_name}/_Node_Merging.csv'
    file_exists = os.path.isfile(csv_path)

    data = get_k_hop_subgraph_from_dataset(dataset_name="Cora", num_hops=5, node_idx=0, visualize=True)
    print("Before:")
    print("Num nodes:", data.num_nodes)
    print("Num edges:", data.edge_index.size(1))

    G = to_networkx(data, to_undirected=True)
    edge_dis = analyze_automorphisms(data, G)
    run_training_pipeline(data, edge_dis, 0, 0, 0, args)

    
    G_data = to_networkx(data)
    N = 10
    M = 20
    for i in range(10):
        ig = random.choice(list(G_data.nodes))
        G_data, pyg_data, star_edges = attach_star_graph(G_data, N, ig)
        # Randomly remove N - 1 edges excluding star structure

        G_data = remove_random_edges(G_data, N - 1, protected_edges=star_edges)
        # Add M random edges, avoiding star structure and original node
        G_data = add_random_edges(G_data, num_edges=M, protected_edges=star_edges, protected_nodes=set([ig]))

        pyg_data = from_networkx(G_data)
        edge_dis = analyze_automorphisms(pyg_data, G_data)
        pyg_data.x = torch.ones((pyg_data.num_nodes, 1433))
        run_training_pipeline(pyg_data, edge_dis, i, 0, 0, args)

if __name__ == "__main__":
    main()
