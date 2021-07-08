import gzip
import pickle
import datetime
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric

def log(str, logfile=None):
    """
    Prints the provided string, and also logs it if a logfile is passed.

    Parameters
    ----------
    str : str
        String to be printed/logged.
    logfile : str (optional)
        File to log into.
    """
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    Takes a 1D tensor, splits it into slices according to pad_sizes, and pads each
    slice  with pad_value to obtain a 2D tensor of size (pad_sizes.shape[0], pad_sizes.max()).

    Parameters
    ----------
    input_ : 1D torch.Tensor
        Tensor to be sliced and padded.
    pad_sizes : 1D torch.Tensor
        Number of elements of the original tensor in each slice.
    pad_value : float (optional)
        Value to pad the tensor with.

    Returns
    -------
    output : 2D torch.Tensor
        Tensor resulting from the slicing + padding operations.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output


class BipartiteNodeData(torch_geometric.data.Data):
    """
    Data class modelling a single graph.

    Parameters
    ----------
    constraint_features : torch.float32
    edge_indices : torch.int64
    edge_features : torch.float32
    variable_features : torch.float32
    candidates : torch.int64
    candidate_choice : torch.int64
    candidate_scores : torch.float32
    """
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, candidate_choice, candidate_scores):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = len(candidates)
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)


class GraphDataset(torch_geometric.data.Dataset):
    """
    Dataset class implementing the basic methods to read samples from a file.

    Parameters
    ----------
    sample_files : list
        List containing the path to the sample files.
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.sample_files)

    def get(self, index):
        """
        Reads and returns sample at position <index> of the dataset.

        Parameters
        ----------
        index : int
            Index over the sample file list. Will return sample in this position.

        Returns
        -------
        graph : BipartiteNodeData object
            Data sample, in this case a  bipartite graph.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample['data']

        constraint_features, (edge_indices, edge_features), variable_features = sample_observation

        # mask variable features (no incumbent info)
        variable_features = np.delete(variable_features, 14, axis=1)
        variable_features = np.delete(variable_features, 13, axis=1)

        constraint_features = torch.FloatTensor(constraint_features)
        edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
        edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
        variable_features = torch.FloatTensor(variable_features)

        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_choice = torch.where(candidates == sample_action)[0][0]  # action index relative to candidates
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                  candidates, candidate_choice, candidate_scores)
        graph.num_nodes = constraint_features.shape[0]+variable_features.shape[0]
        return graph


class Scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    Inherits from pytorch's ReduceLROnPlateau scheduler.
    The behavior is the same, except that the num_bad_epochs attribute is **not** reset to
    zero whenever the learning rate is reduced. This means that it will only be reset
    to zero when an improvement on the tracked metric is reported.
    """
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch =+1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs == self.patience:
            self._reduce_lr(self.last_epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
