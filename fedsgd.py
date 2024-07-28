import torch
from torch import nn
import torch.optim as optim

def fed_sgd(client_models, global_model):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def train_fed_sgd(client_models, data_loader, criterion, epochs=1):
    for epoch in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            for model in client_models:
                model.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    return client_models
