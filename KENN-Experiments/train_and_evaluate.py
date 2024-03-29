# train kenn-sub-Experiments here
# this should later on be done in another file but to keep the overview I have it in a separate file now
# Remark: only transductive training at the moment, only one base NN (= MLP)
import argparse
import os.path
from time import time
import gc

import numpy as np
import torch
import torch.backends.mps
import torch.nn.functional as F
import torch_geometric
from torch.utils.tensorboard.writer import SummaryWriter

import wandb
from app_stats import RunStats, ExperimentStats
from generate_knowledge import generate_knowledge
from model import get_model
from ogb.nodeproppred import Evaluator
from preprocess_data import load_and_preprocess
from training_batch import train, test


def callback_early_stopping(valid_accuracies, epoch, args):
    """
    Takes as argument the list with all the validation accuracies.
    If patience=k, checks if the mean of the last k accuracies is higher than the mean of the
    previous k accuracies (i.e. we check that we are not overfitting). If not, stops learning.
    @param valid_accuracies - list(float) , validation accuracy per epoch
    @param args: arguments in input
    @param epoch: current epoch
    @return bool - if training stops or not

    """
    step = len(valid_accuracies)
    patience = args.es_patience // args.eval_steps
    # no early stopping for 2 * patience epochs
    if epoch < 2 * args.es_patience:
        return False

    # Mean loss for last patience epochs and second-last patience epochs
    mean_previous = np.mean(valid_accuracies[step - 2 * patience:step - patience])
    mean_recent = np.mean(valid_accuracies[step - patience:step])
    delta = mean_recent - mean_previous
    if delta <= args.es_min_delta:
        print("CB_ES Validation Accuracy didn't increase in the last %d epochs" % args.es_patience)
        print("CB_ES delta:", delta)
        print(f"callback_early_stopping signal received at epoch {epoch}")
        print("Terminating training")
        return True
    else:
        return False


def run_experiment(args):
    torch_geometric.seed_everything(args.seed)
    print(f"backend available {torch.backends.mps.is_available()}")
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Cuda available? {torch.cuda.is_available()}, Number of devices: {torch.cuda.device_count()}')

    if os.path.exists('knowledge_base'):
        os.remove('knowledge_base')
        print('knowledge base deleted')
    if os.path.exists('data_stats'):
        os.remove('data_stats')
        print('data stats deleted')

    print(f'Start {args.mode} Training')
    xp_stats = ExperimentStats()

    test_accuracies = []
    valid_accuracies = []
    train_accuracies = []

    for run in range(args.runs):

        data, train_loader, all_loader = load_and_preprocess(args)
        generate_knowledge(data.num_classes, args)

        print(f"Run: {run} of {args.runs}")

        if not args.full_batch:
            print(f"Number of Training Batches with batch_size = {args.batch_size}: {len(train_loader)}")

        writer = SummaryWriter('runs/' + args.dataset + f'/{args.mode}/run{run}')

        model = get_model(data, args).to(device)
        evaluator = Evaluator(name=args.dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = F.nll_loss

        train_losses = []
        valid_losses = []
        t_accuracies = []
        v_accuracies = []
        epoch_time = []

        for epoch in range(args.epochs):
            start = time()
            _ = train(model, train_loader, optimizer, device, criterion, args)
            end = time()

            if epoch % args.eval_steps == 0:
                t_accuracy, v_accuracy, _, t_loss, v_loss, _ = test(model, all_loader, criterion, device, evaluator,
                                                                    data)

                # Save stats for tensorboard
                writer.add_scalar("loss/train", t_loss, epoch)
                writer.add_scalar("loss/valid", v_loss, epoch)
                writer.add_scalar("accuracy/train", t_accuracy, epoch)
                writer.add_scalar("accuracy/valid", v_accuracy, epoch)

                t_accuracies.append(t_accuracy)
                v_accuracies.append(v_accuracy)
                train_losses.append(t_loss)
                valid_losses.append(v_loss)
                epoch_time.append(end - start)

                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {t_loss:.4f}, '
                      f'Time per Train Step: {end - start:.6f} '
                      f'Train: {100 * t_accuracy:.2f}%, '
                      f'Valid: {100 * v_accuracy:.2f}% ')

            # early stopping
            if args.es_enabled and callback_early_stopping(valid_accuracies, epoch, args):
                print(f'Early Stopping at epoch {epoch}.')
                break

        train_accuracy, valid_accuracy, test_accuracy, _, _, _ = test(model, all_loader, criterion, device, evaluator, data)
        test_accuracies.append(test_accuracy)
        valid_accuracies.append(valid_accuracy)
        train_accuracies.append(train_accuracy)

        rs = RunStats(run, train_losses, t_accuracies, valid_losses, v_accuracies, test_accuracy, epoch_time,
                      test_accuracies)
        xp_stats.add_run(rs)
        print(rs)
        wandb.log(rs.to_dict())
        wandb.run.summary["test_accuracies"] = test_accuracies
        wandb.run.summary["valid_accuracies"] = valid_accuracies
        wandb.run.summary["train_accuracies"] = train_accuracies
        writer.close()

    xp_stats.end_experiment()
    print(xp_stats)
    wandb.log(xp_stats.to_dict())


def main():
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        help='alternatively ogbn-products, ogbn-arxiv , CiteSeer, PubMed, Cora')
    parser.add_argument('--planetoid_split', type=str, default="public",
                        help="full, geom-gcn, random: see torch-geometric.data.planetoid documentation")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_sampling', type=int,
                        default=1)  # has to correspond to the number of kenn-sub/GCN Layers
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1)  # 500
    parser.add_argument('--runs', type=int, default=10)  # 10
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--mode', type=str, default='transductive',
                        help='transductive or inductive training mode ')  # inductive/transductive
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--binary_preactivation', type=float, default=500.0)
    parser.add_argument('--num_kenn_layers', type=int, default=3)
    parser.add_argument('--range_constraint_lower', type=float, default=0)
    parser.add_argument('--range_constraint_upper', type=float, default=500)
    parser.add_argument('--es_enabled', type=bool, default=False)
    parser.add_argument('--es_min_delta', type=float, default=0.001)
    parser.add_argument('--es_patience', type=int, default=10)
    parser.add_argument('--sampling_neighbor_size', type=int, default=-1)  # all neighbors will be included with -1
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--full_batch', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--train_sampling', type=str, default='graph_saint',
                        help='specify as "cluster", "graph_saint". If '
                             'not specified, standard GraphSAGE sampling '
                             'is applied')
    parser.add_argument('--cluster_partition_size', type=int, default=100,
                        help='argument for cluster sampling: Approximate size of partitions. Should be smaller than batch size. '
                             'If larger than batch size, 1 partition = 1 batch ')
    parser.add_argument('--sample_coverage', type=int, default=0, help='argument for graph saint, if sample coverage '
                                                                       'is 0, no normalization of batches is '
                                                                       'conducted ')
    parser.add_argument('--walk_length', type=int, default=3, help='argument for graph saint')
    parser.add_argument('--num_steps', type=int, default=30, help='argument for graph saint')
    parser.add_argument('--eval_steps', type=int, default=1,
                        help='How often should the model be evaluated: Default: Every epoch. Set to a higher value to reduce overall epoch time and '
                             'evaluate only every i-th step.  ')
    parser.add_argument('--knowledge_base', type=str, default='', help='specify knowledge file manually for test ')
    parser.add_argument('--create-kb', type=bool, default=True,
                        help='if true, create a clause per class. Set to false if manually added kb should be used ')

    args = parser.parse_args()
    print(args)

    run_experiment(args)


if __name__ == '__main__':
    main()