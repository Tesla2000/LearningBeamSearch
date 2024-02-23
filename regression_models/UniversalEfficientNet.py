import torch
from torch import nn, Tensor
from torchvision.models import efficientnet_b0

from Config import Config


def _forward_universal_wrapper(model, forward_function):
    def inner(x):
        batch_size, *_, n_tasks, m_machines = x.shape
        x = x.unsqueeze(1)
        x = forward_function(x)
        if model.max_tasks is None:
            tasks = Tensor(batch_size * [[n_tasks]])
        else:
            tasks = Tensor(batch_size * [model.max_tasks * [0]])
            tasks[:, n_tasks - 1] = 1
        multiplayer = model.accumulator(tasks.to(x.device))
        return torch.multiply(x, multiplayer)

    return inner


def _forward_wrapper(forward_function):
    def inner(x):
        batch_size, *_, n_tasks, m_machines = x.shape
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        return forward_function(x)

    return inner


class UniversalEfficientNetAnySize:
    max_tasks = None

    def __init__(self, *args, **kwargs):
        model = wrapped_efficientnet()
        model.accumulator = nn.Linear(1 if self.max_tasks is None else self.max_tasks, 1)
        model.forward = _forward_universal_wrapper(model, model.forward)
        type(model).__name__ = type(self).__name__
        self.__dict__ = model.__dict__
        model.__class__.max_tasks = self.max_tasks
        self.__class__ = model.__class__


class UniversalEfficientNetMaxSize(UniversalEfficientNetAnySize):
    max_tasks = Config.n_tasks + 1


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
    model.forward = _forward_wrapper(model.forward)
    type(model).__name__ = "EfficientNet"
    return model
