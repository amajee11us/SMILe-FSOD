import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def concat_all_gathered(tensor):
    """gather and concat tensor from all GPUs"""
    gathered = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor)
    output = torch.cat(gathered, dim=0)
    return output

@torch.no_grad()
def select_all_gather(tensor, idx):
    """
    args:
        idx (LongTensor), 0s and 1s.
    Performs all_gather operation on the provided tensors sliced by idx.
    """
    world_size = torch.distributed.get_world_size()

    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)

    idx_gather = [torch.ones_like(idx) for _ in range(world_size)]
    torch.distributed.all_gather(idx_gather, idx, async_op=False)
    idx_gather = torch.cat(idx_gather , dim=0)
    keep = torch.where(idx_gather)
    return output[keep]

def similarity_kernel(batch, metric='cosine'):
    """
    Computes the similarity kernel for a batch of feature vectors using specified metrics.
    Supports normalization of Euclidean distance into [0, 1] range.

    Parameters:
    batch (Tensor): A batch of feature vectors with dimensions [batch_size, feature_dimension].
    metric (str): The metric to use for computing similarity. Options are 'cosine', 'RBF', 'euclidean', or 'normalized_euclidean'.

    Returns:
    Tensor: A similarity matrix of dimensions [batch_size, batch_size].
    """
    if metric == 'cosSim':
        # Normalize the batch and compute cosine similarity
        normalized_batch = F.normalize(batch, p=2, dim=1)
        similarity = torch.mm(normalized_batch, normalized_batch.t())

    elif metric == 'rbf':
        # Compute pairwise squared Euclidean distances
        sq_dists = torch.cdist(batch, batch, p=2).pow(2)
        # Compute the RBF (Radial Basis Function) kernel
        gamma = 1.0 / batch.size(1)  # Gamma is the inverse of the number of features
        similarity = torch.exp(-gamma * sq_dists)

    elif metric == 'euclidean':
        # Compute pairwise Euclidean distances and normalize them
        euclidean_distance = torch.cdist(batch, batch, p=2)
        inverted_similarity = -euclidean_distance
        min_val = torch.min(inverted_similarity)
        max_val = torch.max(inverted_similarity)
        similarity = (inverted_similarity - min_val) / (max_val - min_val)

    else:
        raise ValueError("Unsupported metric. Choose from 'cosine', 'RBF' or 'euclidean'.")

    return similarity
    
def soft_max(similarity_matrix, axis=0):
    """
    Approximates the maximum value along a specified axis of a similarity matrix using the log-sum-exp trick.

    Parameters:
    similarity_matrix (Tensor): An n x n similarity matrix.
    axis (int): The axis along which to compute the maximum. Default is 0.

    Returns:
    Tensor: The approximated maximum values along the specified axis.
    """
    # Find the maximum value along the specified axis
    m = torch.max(similarity_matrix, axis, keepdim=True).values

    # Compute the log-sum-exp
    max_approx = m + torch.log(torch.sum(torch.exp(similarity_matrix - m), axis, keepdim=True))

    return max_approx.squeeze()

def soft_min(similarity_matrix, axis=0):
    """
    Approximates the minimum value along a specified axis of a similarity matrix using the log-sum-exp trick.

    Parameters:
    similarity_matrix (Tensor): An n x n similarity matrix.
    axis (int): The axis along which to compute the maximum. Default is 0.

    Returns:
    Tensor: The approximated minimum values along the specified axis.
    """
    # Find the maximum value along the specified axis
    m = torch.min(similarity_matrix, axis, keepdim=True).values

    # Compute the log-sum-exp
    max_approx = m - torch.log(torch.sum(torch.exp(m - similarity_matrix), axis, keepdim=True))

    return max_approx.squeeze()

def min_sets(A, B, axis=0):
    """
    Approximates the minimum value between two sets along a specified axis using the log-sum-exp negative trick.

    Parameters:
    A, B (Tensor): Two sets of n-dimensional tensors.
    axis (int): The axis along which to compute the minimum. Default is 0.

    Returns:
    Tensor: The approximated minimum values along the specified axis.
    """
    # Combine the two tensors along the specified axis
    combined = torch.cat((A.unsqueeze(axis), B.unsqueeze(axis)), dim=axis)

    # Find the global minimum across the combined tensor along the specified axis
    m = torch.min(combined, axis, keepdim=True).values

    # Compute the log-sum-exp negative
    min_approx = m - torch.log(torch.sum(torch.exp(m - combined), axis, keepdim=True))

    return min_approx.squeeze()