import torch
from torch import nn, Tensor
from torchvision.models import efficientnet_b0

from Config import Config


def _forward_wrapper(model, forward_function):
    def inner(x):
        batch_size, *_, n_tasks, m_machines = x.shape
        x = x.unsqueeze(1)
        result = forward_function(x)
        if Config.max_tasks is None:
            tasks = Tensor(batch_size*[[n_tasks]])
        else:
            tasks = Tensor(batch_size*[Config.max_tasks*[0]])
            tasks[:, n_tasks] = 1
        concatenated = torch.concat((result, tasks), dim=1)
        return model.accumulator(concatenated)

    return inner


def wrapped_universal_efficientnet(*args, **kwargs):
    model = wrapped_efficientnet()
    model.accumulator = nn.Linear(1 + (1 if Config.max_tasks is None else Config.max_tasks), 1)
    model.forward = _forward_wrapper(model, model.forward)
    return model


def wrapped_efficientnet(*args, **kwargs):
    model = efficientnet_b0()
    first_layer = model.features[0]
    first_layer._modules['0'] = nn.Conv2d(
        1,
        first_layer._modules['0'].out_channels,
        first_layer._modules['0'].kernel_size,
        first_layer._modules['0'].stride,
        first_layer._modules['0'].padding,
        first_layer._modules['0'].dilation,
        first_layer._modules['0'].groups,
        first_layer._modules['0'].bias,
        first_layer._modules['0'].padding_mode,
    )
    model.features[0] = first_layer
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        1,
        model.classifier[-1].bias is not None,
    )
    return model
