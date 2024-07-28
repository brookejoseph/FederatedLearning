import torch
from torch import nn
import torch.optim as optim

mu = 0.01

def fed_prox(client_models, global_model):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        for param, global_param in zip(model.parameters(), global_model.parameters()):
            param.data = param.data - mu * (param.data - global_param.data)
    return global_model
