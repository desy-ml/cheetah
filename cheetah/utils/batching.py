import torch


def get_batch_shape(*args):
    result = torch.broadcast_tensors(*args)
    return result[0].shape
