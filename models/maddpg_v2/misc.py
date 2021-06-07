from torch.nn import Module


def soft_update(target: Module, source: Module, tau: float):

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target: Module, source: Module):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
