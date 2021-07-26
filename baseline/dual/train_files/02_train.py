import os
import sys
import glob
import argparse
import pathlib
import numpy as np


def pretrain(policy, pretrain_loader):
    """
    Pre-trains all PreNorm layers in the model.

    Parameters
    ----------
    policy : torch.nn.Module
        Model to pre-train.
    pretrain_loader : torch_geometric.data.DataLoader
        Pre-loaded dataset of pre-training samples.

    Returns
    -------
    i : int
        Number of pre-trained layers.
    """
    policy.pre_train_init()
    i = 0
    while True:
        for batch in pretrain_loader:
            batch.to(device)
            if not policy.pre_train(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features):
                break

        if policy.pre_train_next() is None:
            break
        i += 1
    return i


def process(policy, data_loader, top_k=[1, 3, 5, 10], optimizer=None):
    """
    Process samples. If an optimizer is given, also train on those samples.

    Parameters
    ----------
    policy : torch.nn.Module
        Model to train/evaluate.
    data_loader : torch_geometric.data.DataLoader
        Pre-loaded dataset of training samples.
    top_k : list
        Accuracy will be computed for the top k elements, for k in this list.
    optimizer : torch.optim (optional)
        Optimizer object. If not None, will be used for updating the model parameters.

    Returns
    -------
    mean_loss : float in [0, 1e+20]
        Mean cross entropy loss.
    mean_kacc : np.ndarray
        Mean top k accuracy, for k in the user-provided list top_k.
    """
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(device)
            logits = policy(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            cross_entropy_loss = F.cross_entropy(logits, batch.candidate_choices, reduction='mean')

            # if an optimizer is provided, update parameters
            if optimizer is not None:
                optimizer.zero_grad()
                cross_entropy_loss.backward()
                optimizer.step()

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            # calculate top k accuracy
            kacc = []
            for k in top_k:
                # check if there are at least k candidates
                if logits.size()[-1] < k:
                    kacc.append(1.0)
                    continue
                pred_top_k = logits.topk(k).indices
                pred_top_k_true_scores = true_scores.gather(-1, pred_top_k)
                accuracy = (pred_top_k_true_scores == true_bestscore).any(dim=-1).float().mean().item()
                kacc.append(accuracy)
            kacc = np.asarray(kacc)

            mean_loss += cross_entropy_loss.item() * batch.num_graphs
            mean_kacc += kacc * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed
    return mean_loss, mean_kacc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['item_placement', 'load_balancing', 'anonymous'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    # hyper parameters
    max_epochs = 1000
    batch_size = 12
    pretrain_batch_size = 128
    valid_batch_size = 128
    lr = 1e-3
    top_k = [1, 3, 5, 10]

    # get sample directory
    if args.problem == 'item_placement':
        train_files = glob.glob('train_files/samples/1_item_placement/train/sample_*.pkl')
        valid_files = glob.glob('train_files/samples/1_item_placement/valid/sample_*.pkl')
        running_dir = 'train_files/trained_models/item_placement'

    elif args.problem == 'load_balancing':
        train_files = glob.glob('train_files/samples/2_load_balancing/train/sample_*.pkl')
        valid_files = glob.glob('train_files/samples/2_load_balancing/valid/sample_*.pkl')
        running_dir = 'train_files/trained_models/load_balancing'

    elif args.problem == 'anonymous':
        train_files = glob.glob('train_files/samples/3_anonymous/train/sample_*.pkl')
        valid_files = glob.glob('train_files/samples/3_anonymous/valid/sample_*.pkl')
        running_dir = 'train_files/trained_models/anonymous'

    else:
        raise NotImplementedError

    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]

    # working directory setup
    os.makedirs(running_dir, exist_ok=True)

    # cuda setup
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"

    # import pytorch **after** cuda setup
    import torch
    import torch.nn.functional as F
    import torch_geometric
    from utilities import log, pad_tensor, GraphDataset, Scheduler
    sys.path.insert(0,'.')
    from model import GNNPolicy

    # randomization setup
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # logging setup
    logfile = os.path.join(running_dir, 'train_log.txt')
    if os.path.exists(logfile):
        os.remove(logfile)

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"batch_size: {batch_size}", logfile)
    log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    log(f"valid_batch_size : {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)


    # data setup
    valid_data = GraphDataset(valid_files)
    pretrain_data = GraphDataset(pretrain_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, valid_batch_size, shuffle=False)
    pretrain_loader = torch_geometric.data.DataLoader(pretrain_data, pretrain_batch_size, shuffle=False)


    policy = GNNPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = Scheduler(optimizer, mode='min', patience=10, factor=0.2, verbose=True)

    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)
        if epoch == 0:
            n = pretrain(policy, pretrain_loader)
            log(f"PRETRAINED {n} LAYERS", logfile)
        else:
            epoch_train_files = rng.choice(train_files, int(np.floor(10000/batch_size))*batch_size, replace=True)
            train_data = GraphDataset(epoch_train_files)
            train_loader = torch_geometric.data.DataLoader(train_data, batch_size, shuffle=True)
            train_loss, train_kacc = process(policy, train_loader, top_k, optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

        # validate
        valid_loss, valid_kacc = process(policy, valid_loader, top_k, None)
        log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

        scheduler.step(valid_loss)
        if scheduler.num_bad_epochs == 0:
            torch.save(policy.state_dict(), pathlib.Path(running_dir)/'best_params.pkl')
            log(f"  best model so far", logfile)
        elif scheduler.num_bad_epochs == 10:
            log(f"  10 epochs without improvement, decreasing learning rate", logfile)
        elif scheduler.num_bad_epochs == 20:
            log(f"  20 epochs without improvement, early stopping", logfile)
            break

    # load best parameters and run a final validation step
    policy.load_state_dict(torch.load(pathlib.Path(running_dir)/'best_params.pkl'))
    valid_loss, valid_kacc = process(policy, valid_loader, top_k, None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)
