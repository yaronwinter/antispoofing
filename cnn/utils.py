import torch
import torch.optim as optim

def get_optimizer(parameters, config: dict):
    optimizer = None
    opt_name = config['optimizer_func']
    if opt_name == "adadelta":
        optimizer = optim.Adadelta(parameters,
                                   lr=config['learning_rate'],
                                   rho=config['rho'])
    elif opt_name == 'sgd':
        optimizer = optim.SGD(parameters, config['learning_rate'])
    elif opt_name == "adam":
        optimizer = optim.Adam(parameters,
                               lr=config['learning_rate'],
                               betas=(config['beta_one'],config['beta_two'],),
                               eps=config['eps'])
    else:
        print('Wrong optimizer name: ' + opt_name)
        
    return optimizer
