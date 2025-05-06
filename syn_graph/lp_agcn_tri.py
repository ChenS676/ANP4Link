# adopted from benchmarking/exist_setting_ogb: Run models on ogbl-collab, ogbl-ppa, and ogbl-citation2 under the existing setting.
# python gnn_ogb_heart.py  --use_valedges_as_input  --dataset ogbl-collab  --gnn_model GCN --hidden_channels 256 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3 --epochs 9999 --kill_cnt 100  --batch_size 65536 
# OBGL-PPA,DDI, CITATION2, VESSEL, COLLAB
# basic idea is to replace diffusion operator in mpnn and say whether it works better in ogbl-collab and citation2
# and then expand to synthetic graph
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

# from logger import Logger
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import to_networkx, to_undirected
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from baselines.gnn_utils  import evaluate_hits, evaluate_auc, evaluate_mrr
from torch_geometric.utils import negative_sampling
from tqdm import tqdm 
from graphgps.utility.utils import mvari_str2csv, random_sampling_ogb
import torch
from torch_geometric.data import Data
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import networkx as nx
import random
from syn_random import (init_regular_tilling, 
                        init_pyg_regtil, 
                        RegularTilling, 
                        local_edge_rewiring,
                        nx2Data_split)
from baselines.gnn_utils import (get_root_dir, 
                                 get_logger, 
                                 get_config_dir, 
                                 Logger, 
                                 init_seed, 
                                 save_emb)
from baselines.gnn_utils import GCN
from model import GCN, CNLinkPredictor
from graph_generation import generate_graph, GraphType
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from model import predictor_dict, convdict, GCN, DropEdge
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from utils import PermIterator
import time
from ogbdataset import loaddataset
from typing import Iterable
import wandb
server = 'Horeka'

dir_path = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())


def save_new_results(loggers, dataset, num_node, file_name='test_results_0.25_0.5.csv'):
    new_data = []
    
    for key in loggers.keys():
        if key == 'AUC':
            if len(loggers[key].results[0]) > 0:
                print(key)
                best_valid, best_valid_mean, mean_list, var_list, test_res = loggers[key].print_statistics()

                # Prepare row data
                new_data.append([dataset, num_node, key, best_valid, best_valid_mean, mean_list, var_list, test_res])
        
        # Merge and save the new results with the old ones
    load_and_merge_data(new_data, dataset, num_node, file_name)


def load_and_merge_data(new_data, dataset, num_node, file_name='test_results.csv'):
    try:
        # Try to read the existing CSV file
        old_data = pd.read_csv(file_name)
        
        # Merge the new data (convert new_data to a DataFrame)
        new_data_df = pd.DataFrame(new_data, columns=['dataset', 'num_node', 'Metric', 'Best Valid', 'Best Valid Mean', 'Mean List', 'Variance List', 'Test Result'])
        
        # Concatenate only the necessary columns
        merged_data = new_data_df
    
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame for the new data
        new_data_df = pd.DataFrame(new_data, columns=['dataset', 'num_node', 'Metric', 'Best Valid', 'Best Valid Mean', 'Mean List', 'Variance List', 'Test Result'])
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


def get_graph_statistics(G, graph_name="Graph", perturbation="None"):
    """Calculate and return statistics of a NetworkX graph."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    
    # Compute degree statistics
    degrees = [deg for node, deg in G.degree()]
    avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0
    min_degree = min(degrees) if degrees else None
    max_degree = max(degrees) if degrees else None

    # Create a dictionary with the statistics
    graph_key = f"{graph_name}_Perturbation_{perturbation}"
    
    stats = {
        "Graph Name": graph_key,
        "Number of Nodes": num_nodes,
        "Number of Edges": num_edges,
        "Density": density,
        "Average Degree": avg_degree,
        "Min Degree": min_degree,
        "Max Degree": max_degree,
    }

    return stats

def save_graph_statistics(stats, filename="graph_statistics.csv"):
    """Load existing data, merge new data, and save back to CSV."""
    # Convert new stats to DataFrame
    new_df = pd.DataFrame.from_dict(stats, orient='index')

    # Check if file exists, and load old data if present
    if os.path.exists(filename):
        old_df = pd.read_csv(filename, index_col=0)
        # Merge old and new data, ensuring uniqueness
        updated_df = pd.concat([old_df, new_df]).drop_duplicates()
    else:
        updated_df = new_df

    # Save the merged DataFrame back to CSV
    updated_df.to_csv(filename)
    print(f"Updated graph statistics saved to {filename}")



# Example usage function
def rewiring():
    perturb_ratio = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
    N = 20
    node_size =150
    font_size = 100
    g_type = RegularTilling.TRIANGULAR
    G, _, _, pos = init_regular_tilling(N, g_type, seed=None)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    nx.draw(G, pos, node_size=node_size, font_size=font_size, node_color="gray", edge_color="gray")
    # plt.title(f"Original Triangular {G.number_of_edges()}")
    
    for pr in perturb_ratio:
        G_rewired, rewired_list = local_edge_rewiring(G, num_rewirings=int(pr * G.number_of_edges()), seed=None) # num_rewirting = int(pr * G.number_of_edges())

        node_colors = ["gray"] * len(G_rewired.nodes())
        highlight_nodes = rewired_list

        # Set selected nodes to green
        for i, node in enumerate(G_rewired.nodes()):
            if node in highlight_nodes:
                node_colors[i] = "green"

        # Draw the rewired graph
        
        plt.subplot(1, 2, 2)
        nx.draw(G_rewired, pos, node_size=node_size, font_size=font_size, node_color=node_colors, edge_color="gray")
        # plt.title(f"Rewired Triangular {G_rewired.number_of_edges()}")
        plt.savefig(f'rewired_{pr}.png')
        data_rewired, split_rewired, G_rewired, pos = nx2Data_split(G_rewired, pos, True, 0.25, 0.5)
    
    return data_rewired, split_rewired, G_rewired, pos



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None):
    
    if alpha is not None:
        predictor.setalpha(alpha)
    
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['pos_edge_label_index']
    negedge = split_edge['valid']['neg_edge_label_index']

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    for perm in PermIterator(
            adjmask.device, adjmask.shape[0], batch_size
    ):
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
            row = torch.from_numpy(adj.tocoo().row).long()
            col = torch.from_numpy(adj.tocoo().col).long()
            value = torch.from_numpy(adj.tocoo().data).float()

            adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(data.num_nodes, data.num_nodes))
            
        h = model(data.x, adj)
        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h,
                                                adj,
                                                edge,
                                                cndropprobs=cnprobs)

        pos_losss = -F.logsigmoid(pos_outs).mean()
        negedge = torch.randint(0, data.x.size(0), (pos_train_edge.size()))
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
         use_valedges_as_input):
    model.eval()
    predictor.eval()

    pos_train_edge = split_edge['train'].pos_edge_label_index.to(data.x.device)
    pos_valid_edge = split_edge['valid'].pos_edge_label_index.to(data.x.device)
    neg_valid_edge = split_edge['valid'].neg_edge_label_index.to(data.x.device)
    pos_test_edge = split_edge['test'].pos_edge_label_index.to(data.x.device)
    neg_test_edge = split_edge['test'].neg_edge_label_index.to(data.x.device)

    adj = data.adj_t
    row = torch.from_numpy(adj.tocoo().row).long()
    col = torch.from_numpy(adj.tocoo().col).long()
    value = torch.from_numpy(adj.tocoo().data).float()

    adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(data.num_nodes, data.num_nodes))
    h = model(data.x, adj)

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

    # results = {}
    # for K in [20, 50, 100]:
    #     evaluator.K = K

    #     train_hits = evaluator.eval({
    #         'y_pred_pos': pos_train_pred,
    #         'y_pred_neg': neg_valid_pred,
    #     })[f'hits@{K}']

    #     valid_hits = evaluator.eval({
    #         'y_pred_pos': pos_valid_pred,
    #         'y_pred_neg': neg_valid_pred,
    #     })[f'hits@{K}']
    #     test_hits = evaluator.eval({
    #         'y_pred_pos': pos_test_pred,
    #         'y_pred_neg': neg_test_pred,
    #     })[f'hits@{K}']

    #     results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    results = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    return results


def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [1, 5, 50]
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
    
    mrr_train =  evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred)
    mrr_valid =  evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred)
    mrr_test =  evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred)
    
    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])
    result['MRR'] =  (mrr_train, mrr_valid, mrr_test)
    return result

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_valedges_as_input', action='store_true', help="whether to add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
    parser.add_argument('--runs', type=int, default=3, help="number of repeated runs")
    parser.add_argument('--dataset', type=str, default='RegularTilling.TRIANGULAR')
    parser.add_argument('--N', type=str, help='number of the node in synthetic graph')
    parser.add_argument('--pr', type=float, default=0.1, help='percentage of perturbation of edges')
    
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
    parser.add_argument('--xdp', type=float, default=0, help="dropout ratio of gnn")
    parser.add_argument('--tdp', type=float, default=0, help="dropout ratio of gnn")
    parser.add_argument('--gnnedp', type=float, default=0.0, help="edge dropout ratio of gnn")
    parser.add_argument('--predp', type=float, default=0.0, help="dropout ratio of predictor")
    parser.add_argument('--preedp', type=float, default=0.0, help="edge dropout ratio of predictor")
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
    parser.add_argument('--predictor', choices=predictor_dict.keys(), default='cn0')
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
    return args


def main():
    args = parseargs()
    print(args, flush=True)

    hpstr = str(args).replace(" ", "").replace("Namespace(", "").replace(
        ")", "").replace("True", "1").replace("False", "0").replace("=", "").replace("epochs", "").replace("runs", "").replace("save_gemb", "")
    writer = SummaryWriter(f"./rec/{args.model}_{args.predictor}")
    writer.add_text("hyperparams", hpstr)

    evaluator = Evaluator(name=f'ogbl-ppa')

    device = 'cpu'# torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.dataset =='RegularTilling.TRIANGULAR':
        eval_metric = 'AUC'
        N = 4000

    G, _, _, pos = init_regular_tilling(N, eval(args.dataset), seed=None)
    
    graph_stats = get_graph_statistics(G, graph_name=args.dataset, perturbation=args.pr)
    save_graph_statistics(graph_stats)

    for key, value in graph_stats.items():
        print(f"{key}: {value}")
    
    G_rewired, rewired_list = local_edge_rewiring(G, num_rewirings=int(args.pr * G.number_of_edges()), seed=None)
    rewired_stats = get_graph_statistics(G_rewired, graph_name=args.dataset, perturbation=args.pr)
    save_graph_statistics(rewired_stats)
    data, split_edge, G, pos = nx2Data_split(G_rewired, pos, True, 0.25, 0.5)

    data = data.to(device)
    predfn = predictor_dict[args.predictor]
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@5': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        # 'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs),
        # 'mrr_hit20':  Logger(args.runs),
        # 'mrr_hit50':  Logger(args.runs),
        # 'mrr_hit100':  Logger(args.runs),
    }

    for run in range(0, args.runs):

        set_seed(run)
        
        # build model
        data.max_x = -1
        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, 
                    taildropout=args.tdp, noinputlin=args.loadx).to(device)
        
        if args.loadx:
            with torch.no_grad():
                model.xemb[0].weight.copy_(torch.load(f"gemb/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"))
            model.xemb[0].weight.requires_grad_(False)
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
            name_tag = f"{args.dataset}_{args.pr}_{args.model}_{args.testbs}_{args.predictor}_{args.gnnedp}_{args.predp}_{args.predp}_{args.tdp}_{args.xdp}_{args.hiddim}_{args.mplayers}_{args.nnlayers}_{args.use_xlin}"
            wandb.init(project=f"{args.dataset}_grand_{server}_{args.runs}", name=name_tag, config=vars(args))
        
            optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
            {'params': predictor.parameters(), 'lr': args.prelr}])
            
            for epoch in range(1, 1 + args.epochs):
                alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
                # t1 = time.time()
                loss = train(model, predictor, data, split_edge, optimizer,
                            args.batch_size, args.maskinput, [], alpha)
                wandb.log({'train_loss': loss.item()}, step = epoch)
                
                if epoch % 1 == 0:
                    results = test(model, predictor, data, split_edge, evaluator,
                                args.testbs, args.use_valedges_as_input)
                    for key, result in results.items():
                        loggers[key].add_result(run, result)
                        wandb.log({f"Metrics/{key}": result[-1]}, step=epoch)
                    
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
                    if key == 'AUC':
                        try: 
                            best_valid, best_valid_mean, mean_list, var_list, test_res = loggers[key].print_statistics()
                            print(best_valid)
                            print(best_valid_mean)
                            print(mean_list)
                            print(var_list)
                            print(test_res)
                        except:
                            pass

    save_new_results(loggers, name_tag, N, file_name=f'{args.dataset}_{args.model}_{args.predictor}test_results_0.25_0.5.csv')


if __name__ == "__main__":
    main()