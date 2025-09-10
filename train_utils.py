import os
import re
import sys
import torch
import random
import numpy as np
import pandas as pd
from glob import glob
from mlcrate import LinewiseCSVWriter as mlcLinewiseCSVWriter

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence 

from sklearn.metrics import mean_squared_error

from transformers import (AdamW, 
                          get_linear_schedule_with_warmup, 
                          get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup)

import bitsandbytes as bnb
from loguru import logger
from pytorch_lightning import seed_everything
from fastai.vision.all import set_seed


def get_exp_name(config):
   
    model = config.model

    model = re.sub('microsoft/deberta-v3-', "d3", model)
    model = re.sub('microsoft/deberta-', "d", model)
    model = re.sub('roberta-', "r", model)
    model = re.sub('mosaicml/mpt-', "mp", model)

    model = re.sub('small', "s", model)
    model = re.sub('base', "b", model)
    model = re.sub('large', "l", model)
    
    exp_name = f"{config.cv}-{config.train.n_folds}{config.seed}-{model}{config.train.model}{config.preprocess}-{config.train.max_length}-{config.train_loader.batch_size}-{config.train.body_epochs}{config.train.scheduler_epochs}-{'%.e' % config.optimizer.lr}{'%.e' % config.optimizer.head_lr}-{config.optimizer.lr_decay_type}{config.optimizer.lr_decay_rate}-{config.optimizer.weight_decay}{config.scheduler.num_warmup_steps}d{config.model_config.hidden_dropout_prob}-{config.train.loss}-{config.train.num_freeze}"
    
    exp_name = re.sub('-0', '', exp_name)
    exp_name = re.sub('\.', '', exp_name)

    if config.exp_name:
        exp_name = f"{exp_name}-{config.exp_name}"

    return exp_name


def seed_everything_custom(seed):
    seed_everything(seed)
    set_seed(seed, reproducible=True)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def remove_abnormal_exp(output_path: str, exp_name: str) -> None:
    log_files = glob(os.path.join(output_path, '*.csv'))
    for log_file in log_files:
        log_df = pd.read_csv(log_file)
        if len(log_df) == 0:
            os.remove(os.path.join(output_path, f'{exp_name}.log'))
            os.remove(os.path.join(output_path, f'{exp_name}.csv'))
            os.remove(os.path.join(output_path, f'{exp_name}.yaml'))
            
    
def get_logger(config, exp_num):
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(os.path.join(config.output_path, f'{exp_num}.log'))
    header = config.header.split(' ')
    csv_log = mlcLinewiseCSVWriter(os.path.join(config.output_path, f'{exp_num}.csv'), header=header, append=True)
    return logger, csv_log

def get_optimizer_params(model, config, lr, head_lr, weight_decay=0.0, lr_decay_type='a'):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    head_layers = ["classifier"]

    # for idx, (n, p) in enumerate(model.named_parameters()):
    #     logger.info(f"{idx} - {n} - {p.requires_grad}")

    if lr_decay_type == 's':
        optimizer_parameters = filter(lambda x: x.requires_grad, model.parameters())
    
    elif lr_decay_type == 'i':
        optimizer_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(x in n for x in head_layers)], 'lr': lr, 'weight_decay': weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(x in n for x in head_layers)], 'lr': lr, 'weight_decay': 0.0},
                {'params': [p for n, p in model.named_parameters() if any(x in n for x in head_layers)], 'lr': head_lr, 'weight_decay': 0.0}
        ]

    elif lr_decay_type == 'l':
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if any(x in n for x in head_layers)], 'lr': head_lr, 'weight_decay': 0.0},
        ]
        # initialize lrs for every layer
        layers = [param for param in model.named_parameters() if not any(x in param[0] for x in head_layers)]
        # print(optimizer_parameters)
        layers.reverse()
        layerwise_learning_rate_decay = config.optimizer.lr_decay_rate
        for idx, layer in enumerate(layers):
            n, p = layer
            if not any(nd in n for nd in no_decay):
                optimizer_parameters += [
                    {"params": [p], "weight_decay": weight_decay, "lr": lr},
                ]
            if any(nd in n for nd in no_decay):
                optimizer_parameters += [
                    {"params": [p], "weight_decay": 0.0, "lr": lr},
                ]
            # if idx % 2 == 0:
            lr *= layerwise_learning_rate_decay

    elif lr_decay_type == 'L':
        optimizer_parameters = [
            # {'params': [p for n, p in model.named_parameters() if "classifier" in n], 'lr': head_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "segmentation_head" in n], 'lr': head_lr, 'weight_decay': 0.0},
        ]

        # initialize lrs for every layer
        layers = [getattr(model, "model")] 
        # layers = list(getattr(model, "model").encoder.layer)
        print(len(layers))     

        # print(layers)
        layers.reverse()
        layerwise_learning_rate_decay = config.optimizer.lr_decay_rate
        for idx, layer in enumerate(layers):
            # print(layer)
            optimizer_parameters += [
                {"params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay) and "segmentation_head" not in n], "weight_decay": weight_decay, "lr": lr},
                {"params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay) and "segmentation_head" not in n], "weight_decay": 0.0, "lr": lr},
            ]
            # if idx % 2 == 0:
            lr *= layerwise_learning_rate_decay

    elif lr_decay_type == 'C':
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if "classifier" in n], 'lr': head_lr, 'weight_decay': 0.0},
            # {'params': [p for n, p in model.named_parameters() if "segmentation_head" in n], 'lr': head_lr, 'weight_decay': 0.0},
        ]

        # initialize lrs for every layer
        layers = list(model.named_parameters())
        layers.reverse() 
        print(len(layers))     

        layerwise_learning_rate_decay = config.optimizer.lr_decay_rate
        for idx, layer in enumerate(layers):
            n, p = layer
            if "classifier" not in n:
            # if "segmentation_head" not in n:
                if not any(nd in n for nd in no_decay):
                    params = [{"params" :p, "weight_decay": weight_decay, "lr": lr}]
                    optimizer_parameters += params
                    lr *= layerwise_learning_rate_decay
                elif any(nd in n for nd in no_decay):
                    params = [{"params" :p, "weight_decay": 0.0,  "lr": lr}]
                    optimizer_parameters += params
                    lr *= layerwise_learning_rate_decay
                
           
    return optimizer_parameters


def get_scheduler(config, optimizer, train_dataloader, train_layers):
    if train_layers == 'head':
        num_training_steps = config.train.head_epochs  * len(train_dataloader)
    elif train_layers == "body":
        num_training_steps = config.train.scheduler_epochs  * (len(train_dataloader))

    if config.scheduler.name=='linear':
        print(config.scheduler.num_warmup_steps, num_training_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=config.scheduler.num_warmup_steps, 
                                                    num_training_steps=num_training_steps, 
                                                    )
    elif config.scheduler.name=='cosine':
        print(config.scheduler.num_warmup_steps, num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=config.scheduler.num_warmup_steps, 
                                                    num_training_steps=num_training_steps, 
                                                    # num_cycles=config.scheduler.num_cycles
                                                    )
                                                    
    elif config.scheduler.name=='cosine-hard':
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 
                                                                        num_warmup_steps=num_warmup_steps, 
                                                                        num_training_steps=num_training_steps, 
                                                                        # num_cycles=config.scheduler.num_cycles
                                                                        )

    elif config.scheduler.name=='one-cycle':
        scheduler = OneCycleLR(optimizer, n_epochs=config.train.body_epochs, n_batches=len(train_dataloader))

    else:
        print("scheduler None")
        scheduler = None

    return scheduler


def get_optimizer_and_scheduler(train_dataloader, model, config, train_layers):

    ## optimizer
    optimizer_parameters = get_optimizer_params(model,
                                                config,
                                                lr=config.optimizer.lr, 
                                                head_lr=config.optimizer.head_lr,
                                                weight_decay=config.optimizer.weight_decay,
                                                lr_decay_type=config.optimizer.lr_decay_type
                                                )

    if config.optimizer.name == 'Adam' :
        optimizer = Adam(optimizer_parameters, 
                            lr=config.optimizer.lr, 
                            eps=config.optimizer.eps, 
                            betas=tuple(config.optimizer.betas)
                            )

        # optimizer = bnb.optim.Adam8bit(optimizer_parameters, 
        #                     lr=config.optimizer.lr, 
        #                     eps=config.optimizer.eps, 
        #                     betas=tuple(config.optimizer.betas)
        #                     )
    elif config.optimizer.name == 'AdamW':
        optimizer = AdamW(optimizer_parameters, 
                            lr=config.optimizer.lr, 
                            eps=config.optimizer.eps, 
                            betas=tuple(config.optimizer.betas)
                            )

        # optimizer = bnb.optim.AdamW8bit(optimizer_parameters, 
        #                     lr=config.optimizer.lr, 
        #                     eps=config.optimizer.eps, 
        #                     betas=tuple(config.optimizer.betas)
        #                     )
    
    ## scheduler    
    scheduler = get_scheduler(config, optimizer, train_dataloader, train_layers)
    return optimizer, scheduler

def loss_fn(logits, labels, device, config):
    if config.train.loss == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(logits, labels)
    elif config.train.loss == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
    elif config.train.loss == 'sl1':
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(logits, labels)
    # elif config.train.loss == 'rmse':
    #     criterion = RMSELoss()
    #     loss = criterion(logits, labels) 
    elif config.train.loss == 'rmse':
        criterion = MCRMSELoss()
        loss = criterion(logits, labels)  
    
    elif config.train.loss == 'wrmse':
        criterion = MCRMSELoss()
        content_loss = criterion(logits[:, 0], labels[:, 0])  
        wording_loss = criterion(logits[:, 1], labels[:, 1])  
        loss = 0.4 * content_loss + 0.6 * wording_loss

    elif config.train.loss == 'bce_sl1':
        criterion_bce = torch.nn.BCEWithLogitsLoss()
        criterion_sl1 = torch.nn.SmoothL1Loss()

        loss_bce = criterion_bce(logits, labels) 
        loss_sl1 = criterion_sl1(logits, labels)    

        print(loss_bce) 
        print(loss_sl1)  
        loss = criterion_bce(logits, labels) + criterion_sl1(logits, labels) 
        print("loss", loss_bce+loss_sl1)  

   
    return loss

## Calculate metrics
def get_metrics1(y_true, y_pred):
    scores = []
    idxes = y_true.shape[1]
    for i in range(idxes):
        true = y_true[:,i]
        pred = y_pred[:,i]
        score = mean_squared_error(true, pred, squared=False) # RMSE
        scores.append(score)
    metric = np.mean(scores) # MCRMSE
    return scores, metric

def get_metrics(y_true, y_pred):
    # print(y_pred.shape, y_true.shape)
    col_rmse = np.sqrt(np.mean((y_pred.numpy() - y_true.numpy()) ** 2, axis=0))
    mcrmse = np.mean(col_rmse)
    # print(col_rmse.tolist(), mcrmse)
    return col_rmse.tolist(), mcrmse

# def get_metrics(y_true, y_pred):
#     rmse = mean_squared_error(y_true, y_pred, squared=False)
#     return rmse, rmse
    

def freeze(module):
    """
    Freezes module's parameters.
    """
    
    for parameter in module.parameters():
        parameter.requires_grad = False
        
def get_freezed_parameters(module):
    """
    Returns names of freezed parameters of the given module.
    """
    
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
    return freezed_parameters


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

class MCRMSELoss(nn.Module):
    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        colwise_mse = torch.mean(torch.square(y_true - y_pred), dim=0)
        return torch.mean(torch.sqrt(colwise_mse), dim=0)


