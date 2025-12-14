import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
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
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from torch.utils.data import DataLoader
import wandb
import pandas as pd
import copy

# ---- your utilities (unchanged) ----
from syn_real.gnn_utils  import (evaluate_hits, evaluate_auc, evaluate_mrr)
from baselines.gnn_utils import (get_root_dir, get_logger, get_config_dir, Logger, init_seed, save_emb)
from syn_real.auto_operation import (create_disjoint_graph, add_random_edges)
from graphgps.utility.utils import mvari_str2csv
from syn_real.gnn_ogb_heart import init_seed as init_seed_dup
from syn_real.automorphism import (run_wl_test_and_group_nodes, count_automorphic_edges, compute_automorphism_metrics)

dir_path = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())

# -------------------- Models (yours) --------------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, mlp_layer=None, head=None, node_num=None,  cat_node_feat_mf=False, data_name=None):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        if data_name == 'ogbl-citation2':
            if num_layers == 1:
                self.convs.append(GCNConv(in_channels, out_channels, normalize=False))
            elif num_layers > 1:
                self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
                for _ in range(num_layers - 2):
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))
                self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
        else:
            if num_layers == 1:
                self.convs.append(GCNConv(in_channels, out_channels))
            elif num_layers > 1:
                self.convs.append(GCNConv(in_channels, hidden_channels))
                for _ in range(num_layers - 2):
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        if self.invest == 1:
            print('layers in gcn: ', len(self.convs))
            self.invest = 0
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class mlp_score(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(mlp_score, self).__init__()
        self.lins = torch.nn.ModuleList()
        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
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

# -------------------- Star attach helpers (yours) --------------------
def attach_star_graph_with_features(G_orig: nx.Graph, data: Data, N: int, ig: int):
    G_combined = G_orig.copy()
    offset = max(G_combined.nodes) + 1 if len(G_combined.nodes) > 0 else 0
    G_star = nx.star_graph(N - 1)
    mapping = {i: i + offset for i in G_star.nodes}
    G_star = nx.relabel_nodes(G_star, mapping)
    center_node_new = mapping[0]
    G_combined.add_nodes_from(G_star.nodes(data=True))
    G_combined.add_edges_from(G_star.edges(data=True))
    G_combined.add_edge(center_node_new, ig)
    star_edges = set(G_star.edges)
    star_edges.add((center_node_new, ig))
    original_x = data.x
    ig_feature = original_x[ig]
    num_new_nodes = N
    new_feats = ig_feature.unsqueeze(0).repeat(num_new_nodes, 1)
    new_x = torch.cat([original_x, new_feats], dim=0)
    new_data = from_networkx(G_combined)
    new_data.x = new_x
    return G_combined, new_data, star_edges

def perturb_disjoint(graph_data, args, inter_ratio, intra_ratio, total_edges):
    if inter_ratio != 0 and intra_ratio != 0 and total_edges != 0:
        updated_graph_data = add_random_edges(graph_data, inter_ratio=inter_ratio, intra_ratio=intra_ratio, total_edges=total_edges)
    else:
        updated_graph_data = graph_data
    G = to_networkx(updated_graph_data, to_undirected=True)
    num_nodes = updated_graph_data.num_nodes
    node_groups, node_labels, new_labels = run_wl_test_and_group_nodes(updated_graph_data.edge_index, num_nodes=num_nodes, num_iterations=30)
    intra_orbit_edges, inter_orbit_edges = count_automorphic_edges(G, node_labels)
    ig = random.choice(list(G.nodes))
    N = 20
    G_data, updated_graph_data, star_edges = attach_star_graph_with_features(G, updated_graph_data, N, ig)
    metrics_after, num_nodes, group_sizes = compute_automorphism_metrics(node_groups, num_nodes)
    df = pd.DataFrame([metrics_after])
    print(df)
    print(f"Finished with inter_ratio={inter_ratio}, intra_ratio={intra_ratio}, total_edges={total_edges}")
    return updated_graph_data, metrics_after, intra_orbit_edges, inter_orbit_edges

# -------------------- Dataset helpers (yours) --------------------
def load_real_world_graph(dataset_name="Cora"):
    if dataset_name in ['Cora', 'Citeseer', 'PubMed']:
        dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
        data = dataset[0]
    elif dataset_name.startswith('ogbl'):
        raise NotImplementedError
    return data

def parse_args():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default="Citeseer")
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    parser.add_argument('--pt_path', type=str, default=f"plots/Citeseer/processed_graph_inter0.5_intra0.5_edges1000_auto0.7200_norm1_0.7676.pt")
    ## gnn
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--gnnout_hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--eval_metric', type=str, default='AUC')
    ## train
    parser.add_argument('--batch_size', type=int, default=2**8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--kill_cnt', type=int, default=20, help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2', type=float, default=0.0)
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
    # ---- AdaEdge-LP flags (NEW) ----
    parser.add_argument('--use_adaedge', type=int, default=0, help='0=off, 1=on')
    parser.add_argument('--ae_every', type=int, default=1, help='apply every k eval steps')
    parser.add_argument('--ae_num_add', type=int, default=200, help='max edges to add each adjust')
    parser.add_argument('--ae_num_rmv', type=int, default=200, help='max edges to remove each adjust')
    parser.add_argument('--ae_conf_add', type=float, default=0.95, help='add if score >= this')
    parser.add_argument('--ae_conf_rmv', type=float, default=0.05, help='remove if score <= this')
    parser.add_argument('--ae_topk_per_node', type=int, default=20, help='candidate non-edges per node (shortlist by dot-product)')
    parser.add_argument('--ae_order', type=str, default='add_first', choices=['add_first','remove_first'])
    args = parser.parse_args()
    return args

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
    split_edge['train']['edge'] = removerepeated(torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
    split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
    split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
    split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
    split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()
    return split_edge

def data2dict(data, splits, data_name) -> dict:
    if data_name in ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'ogbl-ddi']:
        datadict = {}
        datadict.update({'adj': data.adj_t})
        datadict.update({'train_pos': splits['train']['edge']})
        datadict.update({'valid_pos': splits['valid']['edge']})
        datadict.update({'valid_neg': splits['valid']['edge_neg']})
        datadict.update({'test_pos': splits['test']['edge']})
        datadict.update({'test_neg': splits['test']['edge_neg']})
        datadict.update({'train_val': torch.cat([splits['valid']['edge'], splits['train']['edge']])})
        datadict.update({'x': data.x})
    else:
        raise ValueError('data_name not supported')
    return datadict

# -------------------- Training/Eval (yours, unchanged) --------------------
def train(model, score_func, train_pos, x, optimizer, batch_size):
    model.train()
    score_func.train()
    total_loss = total_examples = 0
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        num_nodes = x.size(0)
        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
        train_edge_mask = train_pos[mask].transpose(1,0)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]), dim=1)
        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        x = x.to(train_pos.device)
        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
        h = model(x, adj)
        edge = train_pos[perm].t()
        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
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

@torch.no_grad()
def test(model, score_func, data, x, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    score_func.eval()
    h = model(x, data['adj'].to(x.device))
    x = h
    pos_train_pred = test_edge(score_func, data['train_val'], h, batch_size)
    neg_valid_pred = test_edge(score_func, data['valid_neg'], h, batch_size)
    pos_valid_pred = test_edge(score_func, data['valid_pos'], h, batch_size)
    pos_test_pred  = test_edge(score_func, data['test_pos'],  h, batch_size)
    neg_test_pred  = test_edge(score_func, data['test_neg'],  h, batch_size)
    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred), torch.flatten(pos_valid_pred)
    pos_test_pred,  neg_test_pred  = torch.flatten(pos_test_pred),  torch.flatten(neg_test_pred)
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    score_emb = [pos_valid_pred.cpu(), neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x.cpu()]
    return result, score_emb

def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    result = {}
    k_list = [1, 10, 20, 50, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val   = evaluate_hits(evaluator_hit, pos_val_pred,   neg_val_pred, k_list)
    result_hit_test  = evaluate_hits(evaluator_hit, pos_test_pred,  neg_test_pred, k_list)
    for K in k_list:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])
    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), torch.zeros(neg_val_pred.size(0), dtype=int)])
    val_pred   = torch.cat([pos_val_pred, neg_val_pred])
    val_true   = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int),   torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred  = torch.cat([pos_test_pred, neg_test_pred])
    test_true  = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int),  torch.zeros(neg_test_pred.size(0), dtype=int)])
    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val   = evaluate_auc(val_pred,   val_true)
    result_auc_test  = evaluate_auc(test_pred,  test_true)
    result_mrr_val   = evaluate_mrr(evaluator_mrr, pos_val_pred,  neg_val_pred.repeat(pos_val_pred.size(0), 1))
    result_mrr_test  = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1))
    for k in result_mrr_val.keys():
        result[k] = (0, result_mrr_val[k], result_mrr_test[k])
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP']  = (result_auc_train['AP'],  result_auc_val['AP'],  result_auc_test['AP'])
    return result

# -------------------- AdaEdge-LP (NEW) --------------------
@torch.no_grad()
def _sparse_to_edge_index(adj: SparseTensor) -> torch.Tensor:
    row, col, _ = adj.coo()
    ei = torch.stack([row, col], dim=0)
    # keep undirected unique i<j
    mask = ei[0] < ei[1]
    return ei[:, mask]

@torch.no_grad()
def _edge_index_to_sparse(ei: torch.Tensor, num_nodes: int, device) -> SparseTensor:
    if ei.numel() == 0:
        return SparseTensor.sparse_diag(torch.ones(1, device=device))[:num_nodes, :num_nodes].coalesce()  # empty
    w = torch.ones(ei.size(1), dtype=torch.float32, device=device)
    adj = SparseTensor(row=ei[0], col=ei[1], value=w, sparse_sizes=(num_nodes, num_nodes))
    # make undirected symmetric
    adj = adj.to_symmetric().coalesce()
    return adj

@torch.no_grad()
def _edge_scores(score_func, h, ei: torch.Tensor, bs: int = 1<<18):
    # batched scoring for a set of edges (2, E)
    scores = []
    E = ei.size(1)
    for s in range(0, E, bs):
        t = min(s + bs, E)
        u = ei[0, s:t]
        v = ei[1, s:t]
        scores.append(score_func(h[u], h[v]).flatten())
    return torch.cat(scores, dim=0)

@torch.no_grad()
def _candidate_non_edges_by_dot(h: torch.Tensor, adj_set: set, topk_per_node: int):
    """
    Shortlist non-edges using raw dot product similarity per node.
    Returns unique undirected pairs (i<j) as a (2, M) tensor.
    """
    n = h.size(0)
    device = h.device
    # normalize lightly to avoid extreme scale (optional)
    # h_norm = F.normalize(h, p=2, dim=1)
    h_norm = h
    all_pairs = []
    # chunk to control memory if needed
    for i in range(n):
        # quick dot with everyone
        s = (h_norm[i] * h_norm).sum(dim=1)
        s[i] = -1e9  # remove self
        # mask existing neighbors by setting very low score
        # neighbors known from adj_set: check membership (i,j) with i<j
        # Build a mask lazily:
        # pick top 4*topk to compensate later removal
        k = min(topk_per_node * 4, n-1)
        topv, idx = torch.topk(s, k=k)
        kept = []
        for j in idx.tolist():
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in adj_set:
                kept.append(j)
            if len(kept) >= topk_per_node:
                break
        for j in kept:
            a, b = (i, j) if i < j else (j, i)
            all_pairs.append((a, b))
    if not all_pairs:
        return torch.empty(2, 0, dtype=torch.long, device=device)
    # unique
    pairs = torch.tensor(all_pairs, dtype=torch.long, device=device)
    pairs = torch.unique(pairs, dim=0)
    return pairs.t()  # (2, M)

@torch.no_grad()
def adaedge_lp_adjust(model, score_func, x, adj: SparseTensor, data_dict: dict,
                      num_add: int = 200, num_rmv: int = 200,
                      conf_add: float = 0.95, conf_rmv: float = 0.05,
                      topk_per_node: int = 20, order: str = 'add_first'):
    """
    Link-prediction AdaEdge:
      - Add up to `num_add` non-edges with link score >= conf_add
      - Remove up to `num_rmv` train edges with score <= conf_rmv
    Only modifies the **training adjacency** (data_dict['adj']).
    """
    device = x.device
    model.eval()
    score_func.eval()

    # current embeddings on train-adj
    h = model(x, adj.to(device))

    # existing train edges (undirected)
    ei_train = _sparse_to_edge_index(adj)
    num_nodes = x.size(0)

    # Build fast membership set for existing edges
    adj_set = set([(int(i), int(j)) for i, j in ei_train.t().tolist()])

    # PROTECT: do not delete validation/test positives; we don't have them in adj,
    # but we also avoid ADDing edges that are already in any pos split (to be safe).
    if 'valid_pos' in data_dict and 'test_pos' in data_dict:
        protected_pos = torch.cat([data_dict['valid_pos'], data_dict['test_pos']], dim=0).to(device)
        protected_set = set([(int(min(a,b)), int(max(a,b))) for a,b in protected_pos.tolist()])
    else:
        protected_set = set()

    # ---- compute removal set (low-confidence existing train edges) ----
    rmv_idx = torch.empty(0, dtype=torch.long, device=device)
    if ei_train.numel() > 0 and num_rmv > 0:
        scores_train = _edge_scores(score_func, h, ei_train)  # prob of edge existing
        # sort ascending (lowest confidence first)
        val, idx = torch.sort(scores_train)
        # respect conf threshold
        mask_thr = val <= conf_rmv
        idx = idx[mask_thr]
        if idx.numel() > num_rmv:
            idx = idx[:num_rmv]
        rmv_idx = idx

    # ---- compute addition set (high-confidence non-edges) ----
    add_pairs = torch.empty(2, 0, dtype=torch.long, device=device)
    if num_add > 0:
        cand_pairs = _candidate_non_edges_by_dot(h, adj_set, topk_per_node)  # (2, M)
        if cand_pairs.numel() > 0:
            # filter out pairs that are protected positives (sanity)
            if len(protected_set) > 0:
                keep = []
                for a,b in cand_pairs.t().tolist():
                    if (a,b) not in protected_set:
                        keep.append([a,b])
                if keep:
                    cand_pairs = torch.tensor(keep, dtype=torch.long, device=device).t()
                else:
                    cand_pairs = torch.empty(2, 0, dtype=torch.long, device=device)

        if cand_pairs.numel() > 0:
            cand_scores = _edge_scores(score_func, h, cand_pairs)
            # keep >= conf_add
            keep_mask = cand_scores >= conf_add
            cand_pairs = cand_pairs[:, keep_mask]
            cand_scores = cand_scores[keep_mask]
            if cand_pairs.numel() > 0:
                # sort by score desc and take top num_add
                val, ord_idx = torch.sort(cand_scores, descending=True)
                take = min(num_add, ord_idx.numel())
                ord_idx = ord_idx[:take]
                add_pairs = cand_pairs[:, ord_idx]

    # ---- apply order ----
    ei_new = ei_train.clone()
    if order == 'add_first':
        if add_pairs.numel() > 0:
            ei_new = torch.cat([ei_new, add_pairs], dim=1)
            ei_new = to_undirected(ei_new)
            # unique i<j
            mask = ei_new[0] < ei_new[1]
            ei_new = ei_new[:, mask]
        if rmv_idx.numel() > 0:
            keep_mask = torch.ones(ei_new.size(1), dtype=torch.bool, device=device)
            # rmv_idx refers to indices of ei_train; need to map to ei_new.
            # Safer: rebuild removal on current ei_new by intersecting with original removed edges.
            if ei_train.numel() > 0:
                rmv_edges = ei_train[:, rmv_idx]
                # mark matches in ei_new
                to_remove = set([(int(i), int(j)) for i,j in rmv_edges.t().tolist()])
                keep_list = []
                for k in range(ei_new.size(1)):
                    a,b = int(ei_new[0,k]), int(ei_new[1,k])
                    if (a,b) not in to_remove:
                        keep_list.append(True)
                    else:
                        keep_list.append(False)
                keep_mask = torch.tensor(keep_list, dtype=torch.bool, device=device)
            ei_new = ei_new[:, keep_mask]
    else:  # remove_first
        if rmv_idx.numel() > 0:
            keep_mask = torch.ones(ei_new.size(1), dtype=torch.bool, device=device)
            rmv_edges = ei_train[:, rmv_idx]
            to_remove = set([(int(i), int(j)) for i,j in rmv_edges.t().tolist()])
            keep_list = []
            for k in range(ei_new.size(1)):
                a,b = int(ei_new[0,k]), int(ei_new[1,k])
                keep_list.append((a,b) not in to_remove)
            keep_mask = torch.tensor(keep_list, dtype=torch.bool, device=device)
            ei_new = ei_new[:, keep_mask]
        if add_pairs.numel() > 0:
            ei_new = torch.cat([ei_new, add_pairs], dim=1)
            ei_new = to_undirected(ei_new)
            mask = ei_new[0] < ei_new[1]
            ei_new = ei_new[:, mask]

    # ---- finalize SparseTensor ----
    new_adj = _edge_index_to_sparse(ei_new, num_nodes=num_nodes, device=device)

    stats = {
        'ae_added': int(add_pairs.size(1)),
        'ae_removed': int(rmv_idx.numel()),
        'train_edges_before': int(ei_train.size(1)),
        'train_edges_after': int(ei_new.size(1)),
    }
    return new_adj, stats

# -------------------- Utility plots (unchanged) --------------------
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

def get_graph_statistics(G, graph_name="Graph"):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    degrees = [deg for _, deg in G.degree()]
    avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0
    stats = {
        "Graph Name": graph_name,
        "Number of Nodes": num_nodes,
        "Number of Edges": num_edges,
        "Density": density,
        "Average Degree": avg_degree,
        "Min Degree": min(degrees) if degrees else None,
        "Max Degree": max(degrees) if degrees else None,
    }
    return stats

# -------------------- Pipeline (your loop; only added AdaEdge call) --------------------
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

    # model + scorer
    model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.dropout,
                    mlp_layer=args.gin_mlp_layer, head=args.gat_head,
                    node_num=node_num, cat_node_feat_mf=args.cat_node_feat_mf,
                    data_name=args.data_name).to(device)
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels, 1,
                    args.num_layers_predictor, args.dropout).to(device)

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

    if args.data_name == 'Cora':
        args.batch_size = 1024; args.lr = 0.01
    elif args.data_name == 'Citeseer':
        args.batch_size = 1024; args.lr = 0.001
    elif args.data_name == 'ogbl-ddi':
        args.batch_size = 2**5; args.lr = 0.00001

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
            wandb.init(project=f"{args.data_name}_3",
                name=f"{args.name_tag}_{args.batch_size}{args.lr}")
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
            lr=args.lr, weight_decay=args.l2
        )
        best_valid, best_test, kill_cnt, step = 0, 0, 0, 0

        # Keep the training adjacency inside data['adj']
        data['adj'] = data['adj'].to(device)

        for epoch in range(1, args.epochs + 1):
            loss = train(model, score_func, train_pos, x, optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results_rank, score_emb = test(
                    model, score_func, data, x,
                    evaluator_hit, evaluator_mrr, args.batch_size
                )
                # ------- (NEW) AdaEdge-LP adjust after evaluation -------
                if args.use_adaedge and ((epoch // args.eval_steps) % args.ae_every == 0):
                    new_adj, ae_stats = adaedge_lp_adjust(
                        model, score_func, x, data['adj'],
                        data_dict=data,
                        num_add=args.ae_num_add, num_rmv=args.ae_num_rmv,
                        conf_add=args.ae_conf_add, conf_rmv=args.ae_conf_rmv,
                        topk_per_node=args.ae_topk_per_node, order=args.ae_order
                    )
                    data['adj'] = new_adj  # update train adjacency
                    if args.wandb_log:
                        wandb.log({f"AdaEdge/added": ae_stats['ae_added'],
                                   f"AdaEdge/removed": ae_stats['ae_removed'],
                                   f"AdaEdge/train_edges_before": ae_stats['train_edges_before'],
                                   f"AdaEdge/train_edges_after":  ae_stats['train_edges_after']}, step=epoch)
                    print("[AdaEdge] ", ae_stats)
                # -------------------------------------------------------

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

# -------------------- Main (yours) --------------------
def main():
    args = parse_args()
    init_seed(args.seed)

    if os.path.exists(f'plots/{args.data_name}') == False:
        os.makedirs(f'plots/{args.data_name}')

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
        total_edges_list = [1,2,3,4,5,6,7,8,9,10]
        multi_factor = 1

    for inter in inter_ratios:
        for intra in intra_ratios:
            for edge_factor in total_edges_list:
                total_edges = int(edge_factor * multi_factor)
                data, metrics, intra_orbit_edges, inter_orbit_edges = perturb_disjoint(disjoint_graph, args, inter, intra, total_edges)
                G = to_networkx(data, to_undirected=True)
                run_training_pipeline(data, intra_orbit_edges + inter_orbit_edges, inter, intra, total_edges, args)

if __name__ == "__main__":
    main()
