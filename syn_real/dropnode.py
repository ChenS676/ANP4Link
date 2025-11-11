import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import copy
import random
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
    train_test_split_edges,
    to_undirected,
)
from torch_geometric.nn import GCNConv

from ogb.linkproppred import Evaluator, PygLinkPropPredDataset  # noqa: F401 (PygLinkPropPredDataset imported but not used)
import wandb

from syn_real.gnn_utils import (
    evaluate_hits,
    evaluate_auc,
    evaluate_mrr,
)
from baselines.gnn_utils import (
    get_root_dir,
    get_logger,
    get_config_dir,
    Logger,
    init_seed,
    save_emb,
)
from syn_real.auto_operation import (
    create_disjoint_graph,
    add_random_edges,
)
from graphgps.utility.utils import mvari_str2csv
from syn_real.gnn_ogb_heart import init_seed as init_seed_ogb  # noqa: F401 (kept for compatibility)
from syn_real.automorphism import (
    run_wl_test_and_group_nodes,
    count_automorphic_edges,
    compute_automorphism_metrics,
)


# -------------------------
# Model: GCN (second design)
# -------------------------
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, mlp_layer=None, head=None, node_num=None,
                 cat_node_feat_mf=False, data_name=None):
        super(GCN, self).__init__()

        self.hiddenlayers = nn.ModuleList()

        if data_name == 'ogbl-citation2':
            if num_layers == 1:
                self.hiddenlayers.append(GCNConv(in_channels, out_channels, normalize=False))
            else:
                self.hiddenlayers.append(GCNConv(in_channels, hidden_channels, normalize=False))
                for _ in range(num_layers - 2):
                    self.hiddenlayers.append(GCNConv(hidden_channels, hidden_channels, normalize=False))
                self.hiddenlayers.append(GCNConv(hidden_channels, out_channels, normalize=False))
        else:
            if num_layers == 1:
                self.hiddenlayers.append(GCNConv(in_channels, out_channels))
            else:
                self.hiddenlayers.append(GCNConv(in_channels, hidden_channels))
                for _ in range(num_layers - 2):
                    self.hiddenlayers.append(GCNConv(hidden_channels, hidden_channels))
                self.hiddenlayers.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.invest = 1  # print once

    def reset_parameters(self):
        for conv in self.hiddenlayers:
            conv.reset_parameters()

    @staticmethod
    def _doconcat(denseout: Optional[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """Concatenate features across levels (dense connectivity)."""
        return x if denseout is None else torch.cat([denseout, x], dim=1)

    def forward(self, input, adj):
        """
        Second-design forward with per-layer dropout and ReLU on hidden layers:
            denseout = None
            for layer in hiddenlayers:
                denseout = concat(denseout, x)
                x = layer(x, adj)
                if not last layer: x = relu(x)
                x = dropout(x)
        Returns final `x` (shape = out_channels).
        """
        x = input
        denseout = None

        if self.invest == 1:
            print('layers in gcn: ', len(self.hiddenlayers))
            self.invest = 0

        for i, gc in enumerate(self.hiddenlayers):
            denseout = self._doconcat(denseout, x)
            x = gc(x, adj)
            if i < len(self.hiddenlayers) - 1:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# -------------------------
# Score function: MLP
# -------------------------
class mlp_score(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(mlp_score, self).__init__()
        self.lins = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


# -----------------------------------
# Star attachment (for perturbations)
# -----------------------------------
def attach_star_graph_with_features(G_orig: nx.Graph, data: Data, N: int, ig: int):
    """
    Adds a star graph with N nodes to G_orig, connects center to ig, and fills features
    of new nodes using the feature of node ig.
    """
    G_combined = G_orig.copy()
    offset = max(G_combined.nodes) + 1 if len(G_combined.nodes) > 0 else 0

    # Create and relabel star graph (center=0)
    G_star = nx.star_graph(N - 1)
    mapping = {i: i + offset for i in G_star.nodes}
    G_star = nx.relabel_nodes(G_star, mapping)
    center_node_new = mapping[0]

    # Add star to G_combined and connect to ig
    G_combined.add_nodes_from(G_star.nodes(data=True))
    G_combined.add_edges_from(G_star.edges(data=True))
    G_combined.add_edge(center_node_new, ig)

    # Track added edges
    star_edges = set(G_star.edges)
    star_edges.add((center_node_new, ig))

    # Extend feature matrix
    original_x = data.x
    ig_feature = original_x[ig]  # [F]
    num_new_nodes = N  # center + leaves

    new_feats = ig_feature.unsqueeze(0).repeat(num_new_nodes, 1)  # [N, F]
    new_x = torch.cat([original_x, new_feats], dim=0)             # [V+N, F]

    # Convert back to PyG
    new_data = from_networkx(G_combined)
    new_data.x = new_x

    return G_combined, new_data, star_edges


# -----------------------------------
# Perturbation wrapper
# -----------------------------------
def perturb_disjoint(graph_data, args, inter_ratio, intra_ratio, total_edges):
    """
    Optionally add random edges, run WL tests & attach a star gadget.
    """
    if inter_ratio != 0 and intra_ratio != 0 and total_edges != 0:
        updated_graph_data = add_random_edges(
            graph_data,
            inter_ratio=inter_ratio,
            intra_ratio=intra_ratio,
            total_edges=total_edges
        )
    else:
        updated_graph_data = graph_data

    G = to_networkx(updated_graph_data, to_undirected=True)
    num_nodes = updated_graph_data.num_nodes

    node_groups, node_labels, new_labels = run_wl_test_and_group_nodes(
        updated_graph_data.edge_index, num_nodes=num_nodes, num_iterations=30
    )
    intra_orbit_edges, inter_orbit_edges = count_automorphic_edges(G, node_labels)

    ig = random.choice(list(G.nodes))
    N = 20
    _, updated_graph_data, star_edges = attach_star_graph_with_features(G, updated_graph_data, N, ig)  # noqa: F841 (star_edges unused)
    metrics_after, num_nodes, group_sizes = compute_automorphism_metrics(node_groups, num_nodes)
    df = pd.DataFrame([metrics_after])
    print(df)

    print(f"Finished with inter_ratio={inter_ratio}, intra_ratio={intra_ratio}, total_edges={total_edges}")
    return updated_graph_data, metrics_after, intra_orbit_edges, inter_orbit_edges


# -----------------------------------
# Dataset loader
# -----------------------------------
def load_real_world_graph(dataset_name="Cora"):
    """
    Load a real-world graph dataset (e.g., Cora) from PyTorch Geometric.
    """
    if dataset_name in ['Cora', 'Citeseer', 'PubMed']:
        dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
        data = dataset[0]
    elif dataset_name.startswith('ogbl'):
        # Placeholder: extend for OGB datasets as needed
        raise NotImplementedError("ogbl datasets loading not implemented here.")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return data


# -----------------------------------
# Plot helpers
# -----------------------------------
def plot_group_size_distribution(group_sizes, args, file_name):
    plt.figure()
    plt.plot(np.log1p(group_sizes))
    plt.xlabel("Group Index (log scale)")
    plt.ylabel("Group Size (log scale)")
    plt.title("Group Size Distribution (Log-Log Scale)")
    plt.savefig(file_name)
    plt.close()


def plot_histogram_group_size(group_sizes, metrics_before, args):
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


def plot_graph_visualization(graph_data, node_labels, args, save_path):
    plt.figure(figsize=(6, 6))
    G = to_networkx(graph_data, to_undirected=True)
    nx.draw(G, node_size=10, font_size=8, cmap='Set1', node_color=node_labels, edge_color="gray")
    plt.title("Graph Visualization with WL-based Node Coloring")
    plt.savefig(save_path)
    plt.close()


def plot_histogram_group_size_log_scale(group_sizes, metrics_before, args, save_path):
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


# -----------------------------------
# Args
# -----------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default="Cora")
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    parser.add_argument('--pt_path', default=f"plots/Citeseer/processed_graph_inter0.5_intra0.5_edges1000_auto0.7200_norm1_0.7676.pt",
                        type=str)

    # gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--gnnout_hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--eval_metric', type=str, default='AUC')

    # train setting
    parser.add_argument('--batch_size', type=int, default=2**8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--kill_cnt', default=20, type=int, help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)

    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--remove_edge_aggre', action='store_true', default=False)
    parser.add_argument('--name_tag', type=str, default='')
    parser.add_argument('--gin_mlp_layer', type=int, default=2)
    parser.add_argument('--gat_head', type=int, default=1)
    parser.add_argument('--cat_node_feat_mf', action='store_true', default=False)
    parser.add_argument('--cat_n2v_feat', action='store_true', default=False)
    parser.add_argument('--test_batch_size', type=int, default=1024 * 64)
    parser.add_argument('--use_hard_negative', action='store_true', default=False)
    parser.add_argument('--wandb_log', action='store_true', default=True)
    parser.add_argument('--metric', type=str, default='AUC')
    parser.add_argument('--inter_ratio', type=float, default=0.5)
    parser.add_argument('--intra_ratio', type=float, default=0.5)
    parser.add_argument('--total_edges', type=int, default=1000)
    return parser.parse_args()


# -----------------------------------
# Splits
# -----------------------------------
def randomsplit(data, val_ratio: float = 0.05, test_ratio: float = 0.15):
    def removerepeated(ei):
        ei = to_undirected(ei)
        ei = ei[:, ei[0] < ei[1]]
        return ei

    data.num_nodes = data.x.shape[0]
    data = train_test_split_edges(data, test_ratio, test_ratio)

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    num_val = int(data.val_pos_edge_index.shape[1] * val_ratio / test_ratio)
    data.val_pos_edge_index = data.val_pos_edge_index[:, torch.randperm(data.val_pos_edge_index.shape[1])]

    split_edge['train']['edge'] = removerepeated(
        torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)
    ).t()
    split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
    split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
    split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
    split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()
    return split_edge


def data2dict(data, splits, data_name) -> dict:
    if data_name in ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'ogbl-ddi']:
        datadict = {
            'adj': data.adj_t,
            'train_pos': splits['train']['edge'],
            'valid_pos': splits['valid']['edge'],
            'valid_neg': splits['valid']['edge_neg'],
            'test_pos': splits['test']['edge'],
            'test_neg': splits['test']['edge_neg'],
            'train_val': torch.cat([splits['valid']['edge'], splits['train']['edge']]),
            'x': data.x,
        }
    else:
        raise ValueError('data_name not supported')
    return datadict


# -----------------------------------
# Train / Test
# -----------------------------------
def train(model, score_func, train_pos, x, optimizer, batch_size):
    model.train()
    score_func.train()

    total_loss = total_examples = 0
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        num_nodes = x.size(0)

        # remove loss edges from aggregation
        mask = torch.ones(train_pos.size(0), dtype=torch.bool, device=train_pos.device)
        mask[perm] = 0
        train_edge_mask = train_pos[mask].transpose(1, 0)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1, 0]]), dim=1)

        edge_weight_mask = torch.ones(train_edge_mask.size(1), dtype=torch.float, device=train_pos.device)
        x = x.to(train_pos.device)
        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)

        h = model(x, adj)

        # positive
        edge = train_pos[perm].t()
        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # negative (random)
        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long, device=h.device)
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
    preds = []
    for perm in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()
        preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
    pred_all = torch.cat(preds, dim=0)
    return pred_all


def get_metric_score(evaluator_hit, evaluator_mrr,
                     pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    result = {}
    k_list = [1, 10, 20, 50, 100]

    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    for K in k_list:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'],
                               result_hit_val[f'Hits@{K}'],
                               result_hit_test[f'Hits@{K}'])

    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([
        torch.ones(pos_train_pred.size(0), dtype=int),
        torch.zeros(neg_val_pred.size(0), dtype=int)
    ])
    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([
        torch.ones(pos_val_pred.size(0), dtype=int),
        torch.zeros(neg_val_pred.size(0), dtype=int)
    ])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([
        torch.ones(pos_test_pred.size(0), dtype=int),
        torch.zeros(neg_test_pred.size(0), dtype=int)
    ])

    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1))
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1))
    for k in result_mrr_val.keys():
        result[k] = (0, result_mrr_val[k], result_mrr_test[k])

    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])
    return result


@torch.no_grad()
def test(model, score_func, data, x, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    score_func.eval()

    h = model(x, data['adj'].to(x.device))
    x = h

    pos_train_pred = test_edge(score_func, data['train_val'], h, batch_size)
    neg_valid_pred = test_edge(score_func, data['valid_neg'], h, batch_size)
    pos_valid_pred = test_edge(score_func, data['valid_pos'], h, batch_size)
    pos_test_pred = test_edge(score_func, data['test_pos'], h, batch_size)
    neg_test_pred = test_edge(score_func, data['test_neg'], h, batch_size)

    # flatten
    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred = torch.flatten(neg_valid_pred)
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)
    neg_test_pred = torch.flatten(neg_test_pred)

    result = get_metric_score(evaluator_hit, evaluator_mrr,
                              pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    score_emb = [pos_valid_pred.cpu(), neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x.cpu()]
    return result, score_emb


# -----------------------------------
# Utils
# -----------------------------------
def get_graph_statistics(G, graph_name="Graph"):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    degrees = [deg for _, deg in G.degree()]
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


def run_training_pipeline(data, metrics, inter, intra, total_edges, args):
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

    model = eval(args.gnn_model)(
        input_channel, args.hidden_channels, args.hidden_channels,
        args.num_layers, args.dropout,
        mlp_layer=args.gin_mlp_layer, head=args.gat_head,
        node_num=node_num, cat_node_feat_mf=args.cat_node_feat_mf,
        data_name=args.data_name
    ).to(device)

    if args.gnn_model == 'MixHopGCN':
        args.hidden_channels = 3 * args.hidden_channels

    score_func = eval(args.score_model)(
        args.hidden_channels, args.hidden_channels,
        1, args.num_layers_predictor, args.dropout
    ).to(device)

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

    # quick per-dataset lr/batch choices
    if args.data_name == 'Cora':
        args.batch_size = 1024
        args.lr = 0.01
    elif args.data_name == 'Citeseer':
        args.batch_size = 1024
        args.lr = 0.001
    elif args.data_name == 'ogbl-ddi':
        args.batch_size = 2 ** 5
        args.lr = 1e-5

    args.name_tag = (
        f'{args.data_name}_'
        f'Non_Edge_{metrics:.2f}_'
        f'ArScore_None'
        f'{args.gnn_model}_'
        f'{args.score_model}_'
        f'inter{inter:.2f}_'
        f'intra{intra:.2f}_'
        f'total{total_edges:.0f}_'
    )

    for run in range(args.runs):
        if args.wandb_log:
            wandb.init(
                project=f"{args.data_name}_3",
                name=f"{args.name_tag}_{args.batch_size}{args.lr}"
            )
            wandb.config.update(vars(args))
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
        best_valid, kill_cnt, step = 0, 0, 0

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
                    if args.wandb_log:
                        wandb.log({'train_loss': loss}, step=epoch)
                        wandb.log({f"Metrics/{key}": result[-1]}, step=epoch)
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

        if args.wandb_log:
            wandb.finish()

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


# -----------------------------------
# Main
# -----------------------------------
def main():
    args = parse_args()
    init_seed(args.seed)

    os.makedirs(f'plots/{args.data_name}', exist_ok=True)

    original_data = load_real_world_graph(args.data_name)
    perturb_disjoint(original_data, args, 0, 0, 0)

    disjoint_graph = create_disjoint_graph(original_data)
    disjoint_graph, metrics, intra_orbit_edges, inter_orbit_edges = perturb_disjoint(disjoint_graph, args, 0, 0, 0)
    run_training_pipeline(disjoint_graph, intra_orbit_edges + inter_orbit_edges, 0, 0, 0, args)

    if args.data_name == 'Cora':
        inter_ratios = [0.1]
        intra_ratios = [0.5]
        total_edges_list = [0.2, 1, 4, 7, 12, 18, 20, 28]
        multi_factor = 250
    elif args.data_name == 'Citeseer':
        inter_ratios = [0.1]
        intra_ratios = [0.5]
        total_edges_list = [0.2, 1, 2, 3, 4, 5, 7, 8, 10, 14]
        multi_factor = 1000
    elif args.data_name == 'ogbl-ddi':
        inter_ratios = [0.5]
        intra_ratios = [0.5]
        total_edges_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        multi_factor = 1
    else:
        inter_ratios = [0.5]
        intra_ratios = [0.5]
        total_edges_list = [1]
        multi_factor = 1

    for inter in inter_ratios:
        for intra in intra_ratios:
            for edge_factor in total_edges_list:
                total_edges = int(edge_factor * multi_factor)
                data, metrics, intra_orbit_edges, inter_orbit_edges = perturb_disjoint(
                    disjoint_graph, args, inter, intra, total_edges
                )
                G = to_networkx(data, to_undirected=True)  # noqa: F841 (kept for possible debugging/plots)
                run_training_pipeline(data, intra_orbit_edges + inter_orbit_edges, inter, intra, total_edges, args)


if __name__ == "__main__":
    main()
