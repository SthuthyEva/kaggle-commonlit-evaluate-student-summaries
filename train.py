#!/usr/bin/env python
# coding: utf-8
import os
import re
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from gc import collect
from pathlib import Path
from atexit import register
import mlcrate.time as mlctime
from omegaconf import OmegaConf
from kaggle import api as kaggleapi
from argparse import ArgumentParser
from fastprogress import progress_bar, master_bar

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data_utils import *
from train_utils import *
from dataset import *
from model import *

tqdm.pandas()
import warnings
warnings.filterwarnings("ignore")

os.environ['TOKENIZERS_PARALLELISM']= 'true'
# ======================================================================================================================

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-config", required=True)
    parser.add_argument("-fold", required=True)
    parser.add_argument("options", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(args.options)

    config.exp_name = get_exp_name(config)
    slug = f"{config.exp_name}"
    print(len(slug))
    
    # assert (len(slug)>6) & (len(slug)<50), 'Exp name length should be greater than 6 and lesser than 50'

    config.train.fold = int(args.fold)
    config.input_path = os.path.join(config.root, config.input_path, config.competition)
    config.output_path = os.path.join(config.root, config.output_path, config.competition, config.exp_name)
    config.weights_path = os.path.join(config.output_path, config.weights_path)

    os.makedirs(config.output_path, exist_ok=True)
    os.makedirs(config.weights_path, exist_ok=True)

    register(remove_abnormal_exp, output_path=config.output_path, exp_name=config.exp_name)
   
    OmegaConf.save(config, os.path.join(config.output_path, f"{config.exp_name}.yaml"))

    global logger, csv_logger, timer, scaler

    logger, csv_logger = get_logger(config, config.exp_name)

    timer = mlctime.Timer()
    logger.info(mlctime.now())
    logger.info(f"config: {config}")
    
    seed_everything_custom(config.seed)
    logger.info(f"Seed set to {config.seed}")
    
    # SCALER FOR FP16
    scaler = GradScaler(enabled=config.train.fp16)
    logger.info(f"Running with fp16: {config.train.fp16}")

    # ===================================== TOKENIIZER =====================================
    # if "deberta-v2" in config.model or "deberta-v3" in config.model:
    #     from transformers import DebertaV2TokenizerFast
    #     tokenizer = DebertaV2TokenizerFast.from_pretrained(config.model)
    # else:
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    logger.info(f"Fast tokenizer : {tokenizer.is_fast}")
    
    # ===================================== DATA PREPARATION =====================================
    df = pd.read_csv(os.path.join(config.input_path, config.input_filename.format(config.train.n_folds, config.cv)))
    logger.info(f"shape of the train df: {df.shape}")

    # ===================================== DEBUG MODE =====================================
    if config.train.debug:
        df = df.sample(200, random_state=config.seed)
        df.reset_index(drop=True, inplace=True)
        logger.info(f"Running in debug mode with less samples")

    # ===================================== CV SPLIT ===========================================
    logger.info(df.groupby('fold').size())

    # ===================================== DATA PREPROCESSING ===========================================
    df['prompt_title_processed'] =  df['prompt_title'].progress_apply(lambda x: preprocessing(x, config.preprocess))
    df['prompt_question_processed'] =  df['prompt_question'].progress_apply(lambda x: preprocessing(x, config.preprocess))
    df['prompt_text_processed'] =  df['prompt_text'].progress_apply(lambda x: preprocessing(x, config.preprocess))        
    df['text_processed'] =  df['text'].progress_apply(lambda x: preprocessing(x, config.preprocess))
    # df['text_processed'] =  df['text'].progress_apply(lambda x: preprocessing(x, config.preprocess, spell=True))

    # sep = " " + tokenizer.sep_token + " "
    sep = " [SEP] " 
 
    
    # df['full_text_processed'] = df["prompt_title_processed"] + " " + df["prompt_question_processed"] + sep + df["text_processed"] 
    # df['full_text_processed'] = "Content Wording" + sep + df["prompt_title_processed"] + " " + df["prompt_question_processed"] + sep + df["text_processed"]  #1
    df['full_text_processed'] = "Content Wording" + sep + df["prompt_title_processed"] + " [Question] " + df["prompt_question_processed"] + " [Answer] " + df["text_processed"]  #2 finalized for now

    # df['full_text'] = "[C][W] " + df["prompt_title"] + " [BR] " + df["prompt_question"] + sep + df["text"] 
    # df['full_text'] = "[C][W] " + df["prompt_title"] + " [B] " + df["prompt_question"] + sep + df["text"] #1
    # df['full_text'] = "[C][W] " + df["text"] + sep + df["prompt_title"] + " [B] " + df["prompt_question"]  #2
    # df['full_text'] = "[C][W] " + df["prompt_question"] + sep + df["text"] + sep + df["prompt_title"] + " [B] " + df["prompt_text"]   #3
    # df['full_text'] = "[C][W] " + df["prompt_question"] + sep + df["text"] + sep + df["prompt_text"]   #4
    # df['full_text'] = "content wording " + df["prompt_title"] + " " + df["prompt_question"] + sep + df["text"] #6



    # df['full_text'] = "[C][W] " + df["prompt_title"] + sep + df["prompt_question"] + sep + df["text"] #1
    # df['full_text'] = "[C][W] " + df["text"] + sep + df["prompt_title"] + sep + df["prompt_question"] #2
    # df['full_text'] = "[C][W] " + df["prompt_title"] + " [BR] " + df["prompt_question"] + sep + df["text"] #3
    # df['full_text'] = "[C][W] " + df["prompt_question"] + sep + df["text"] #3

    # df['full_text'] = "[C][W] " + df["text"] + sep +  df["prompt_title"] + " [BR] " + df["prompt_question"]  #4
    # df['full_text'] = "[C][W] " + df["text"] + sep + df["prompt_question"]  #5
    # df['full_text'] = "[C][W] " + df["text"] + sep + df["prompt_question"]  + " [BR] " + df["prompt_text"]  #6
    # df['full_text'] = "[C][W] " + df["text"] + sep + df["prompt_text"]  #7


    # df['full_text'] = "[C][W] " + df["text"] + "[SEP]" + df["prompt_question"] + "[SEP]" + df["prompt_title"] #1
    # df['full_text'] = "[C][W] " + df["text"] + "[SEP]" + df["prompt_question"] + " " + df["prompt_title"] #2
    # df['full_text'] = "[C][W] " + df["text"] + " [SEP] " + df["prompt_question"] + " " + df["prompt_title"] #3
    # df['full_text'] = df["text"] + "[SEP]" + df["prompt_question"] + " " + df["prompt_title"] #4
    # df['full_text'] = "[C][W] " + df["text"] + " [SEP] " + df["prompt_question"] + "\n" + df["prompt_title"] #5
    # df['full_text'] = "[C][W] " + df["text"] + " [SEP] " + df["prompt_question"] + "[BR]" + df["prompt_title"] #6
    # df['full_text'] = "[C][W] " + df["text"] + " " + df["prompt_question"] + " " + df["prompt_title"] #7
    # df['full_text'] = "[C][W] " + df["text"] + " [SEP] " + df["prompt_question"] + " " + df["prompt_title"] + " " + df["prompt_text"] #8

    df['full_text_len'] = df['full_text_processed'].apply(lambda x: len(tokenizer(x)['input_ids']))

    logger.info(f"Min length in full text: {df['full_text_len'].min()}")
    logger.info(f"Max length in full text: {df['full_text_len'].max()}")
    if config.train.max_length != "None":
        logger.info(f"No.of samples longer than max_length: {df[df['full_text_len']> config.train.max_length].shape[0]}")

    logger.info(f"Tokenizer all special tokens : {len(tokenizer.all_special_tokens)}")
    # tokenizer.add_special_tokens({'additional_special_tokens': ['[BR]', '[B]', '[C]', '[W]']})
    # tokenizer.add_special_tokens({'additional_special_tokens': ['[br]', '[c]', '[w]']})
    # tokenizer.add_special_tokens({'additional_special_tokens': ['[br]', '[b]', '[c]', '[w]']})

    # tokenizer.add_special_tokens({'additional_special_tokens': ['[br]']})
    
    tokenizer.add_tokens(['[Question]', '[Answer]'])

    logger.info(f"Tokenizer all special tokens after addition : {len(tokenizer.all_special_tokens)}")

    # ===================================== KFOLD TRAINING =====================================
    
    fold = config.train.fold
    logger.info(f"------------- {fold+1} of {config.train.n_folds} Folds -------------")
    
    # TRAIN & VALID DATA

    train_set = df[df['fold'] != fold].copy()
    valid_set = df[df['fold'] == fold].copy()

    train_set.sort_values(by=['full_text_len'], inplace=True)
    valid_set.sort_values(by=['full_text_len'], inplace=True)

    train_set.reset_index(drop=True, inplace=True)      
    valid_set.reset_index(drop=True, inplace=True)      

    logger.info(f"TRAIN Dataset: {train_set.shape}")
    logger.info(f"VALID Dataset: {valid_set.shape}")

    if config.train.max_length != "None":
        logger.info(f"Train - No.of samples shorter than max_length: {train_set[train_set['full_text_len'] <= config.train.max_length].shape[0]}")
        logger.info(f"Train - No.of samples longer than max_length: {train_set[train_set['full_text_len'] > config.train.max_length].shape[0]}")    
        logger.info(f"Valid - No.of samples shorter than max_length: {valid_set[valid_set['full_text_len'] <= config.train.max_length].shape[0]}")
        logger.info(f"Valid - No.of samples longer than max_length: {valid_set[valid_set['full_text_len'] > config.train.max_length].shape[0]}")    
        
    # DATASET
    train_dataset = CommonlitDataset(train_set, 
                                    config.train.label_cols,
                                    tokenizer, 
                                    config.train.max_length, 
                                    if_collate=True,
                                    config=config
                                    )

    valid_dataset = CommonlitDataset(valid_set, 
                                    config.train.label_cols,
                                    tokenizer, 
                                    config.train.max_length, 
                                    if_collate=True,
                                    config=config
                                    )
    # logger.info(f"TRAIN Dataset: {train_dataset.__getitem__(0)}")

    # DEVICE
    device = torch.device("cuda")

    # MODEL 
    model = Model(config)
    # model = AutoModelForSequenceClassification.from_pretrained(config.model)

    for idx, (n, p) in enumerate(model.named_parameters()):
        if idx <= config.train.num_freeze:
            p.requires_grad = False
    
    # for idx, (n, p) in enumerate(model.named_parameters()):
    #     logger.info(f"{idx} - {n} - {p.requires_grad}")

    # gradient checkpointing
    model.model.gradient_checkpointing_enable()
    logger.info(f"Gradient Checkpointing: {model.model.is_gradient_checkpointing}")
        
    model.to(device)
    logger.info(f"Token Embedding shape : {model.model.get_input_embeddings()}")
    model.model._resize_token_embeddings(len(tokenizer))
    logger.info(f"Token Embedding shape after adding special tokens : {model.model.get_input_embeddings()}")

    # logger.info(f"Model configuration: {model.config}")
    # freezing embeddings and first 2 layers of encoder
    # freeze(model.model.embeddings)
    # freeze(model.model.encoder.layer[:config.train.num_freeze])

    # freezed_parameters = get_freezed_parameters(model)
    # print(f"Freezed parameters: {freezed_parameters}")

    # ===================================== BODY TRAINING =====================================
    if config.train.head:
        logger.info("Head Model Training")
        for i in range(config.train.num_freeze):
            config.train.freeze_layers.append(f"encoder.layer.{i}")

        logger.info(f"Freezed layers - {config.train.freeze_layers}")

        for idx, (name, param) in enumerate(model.named_parameters()):
            if any(layer in name for layer in config.train.freeze_layers):
                param.requires_grad=False
    else:
        logger.info("Entire Model Training")

        
    # DATALOADER
    train_dataloader = DataLoader(train_dataset, collate_fn=custom_collate, **config.train_loader)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=custom_collate, **config.val_loader)

    # # UNFREEZE ALL LAYERS
    # for idx, (name, param) in enumerate(model.named_parameters()):
    #     param.requires_grad=True

    # ================================= ACTUAL TRAINING ======================================
    # OPTIMIZER AND SCHEDULER
    optimizer, scheduler = get_optimizer_and_scheduler(train_dataloader, model, config, "body")
    
    best_score = np.inf
    mb = master_bar(range(config.train.body_epochs))
    timer.add('train')

    for epoch in mb:    
        # seed_everything_custom(config.seed + epoch)
        model, score = train_fn("body", 
                                fold, 
                                epoch, 
                                best_score, 
                                train_dataloader, 
                                valid_dataloader,
                                model,
                                device,
                                optimizer, 
                                scheduler, 
                                config,
                                mb,
                                tokenizer
                                )
        best_score = score
    logger.info(f"best score: {round(best_score.item(), 6)}")

    # ========================================= WRITE PREDICTIONS TO OUTPUT PATH =========================================
    state_dict = torch.load(os.path.join(config.weights_path, f"best_score_body_fold{fold}.pth"), map_location=device)    
    model.load_state_dict(state_dict)
    # epoch=0
    v_loss, v_score, val_targets, val_preds = val_fn(valid_dataloader, model, device, config, mb, epoch)
    
    for idx, label in enumerate(config.train.label_cols):
        valid_set[f"{label}_predictions"] = val_preds[:, idx].tolist()
       
    # valid_set.drop(["full_text", "full_text_len"], axis=1, inplace=True)
    valid_set.to_csv(os.path.join(config.output_path, f"predictions_fold{fold}.csv"), index=False)
    logger.info("Written predictions to output path")
    
    del train_set, valid_set, train_dataset, valid_dataset, train_dataloader, valid_dataloader, model, optimizer, scheduler
    torch.cuda.empty_cache()
    collect()

    logger.info(f"==================================================================================================================================")

    # ========================================= UPLOAD WEIGHTS TO KAGGLE =========================================
    if fold == (config.train.n_folds-1):
        # calculate CV
        fold_predictions=[]
        pred_cols = [f"{label}_predictions" for label in config.train.label_cols]
        for fold in range(config.train.n_folds):
            df_predictions = pd.read_csv(os.path.join(config.output_path, f"predictions_fold{fold}.csv")) 
            fold_predictions.append(df_predictions)
            fold_classwise_metric, fold_wise_metric = get_metrics(torch.tensor(df_predictions[pred_cols].values), torch.tensor(df_predictions[config.train.label_cols].values))
            logger.info(f"Classwise MCRMSE Score for fold{fold} - {['%.6f' % elem for elem in fold_classwise_metric]}")
            logger.info(f"MCRMSE Score for fold{fold} - {'%.6f' % fold_wise_metric}")
        
        
        fold_predictions = pd.concat(fold_predictions, axis=0)
        classwise_metric, overall_metric = get_metrics(torch.tensor(fold_predictions[pred_cols].values), torch.tensor(fold_predictions[config.train.label_cols].values))
        logger.info(f"Classwise MCRMSE CV Score - {['%.6f' % elem for elem in classwise_metric]}")
        logger.info(f"MCRMSE CV Score - {'%.6f' % overall_metric}")

        kaggleapi.dataset_initialize_cli(config.weights_path)
        with open(f"{config.weights_path}/dataset-metadata.json", "r+") as f:
            meta_data = json.load(f)
            meta_data['title'] = config.exp_name
            meta_data['id'] = f"{kaggleapi.config_values['username']}/{config.exp_name}"
            f.seek(0)                 # should reset file position to the beginning.
            json.dump(meta_data, f)
            f.truncate()              # remove remaining part
        kaggleapi.dataset_create_new_cli(config.weights_path, convert_to_csv=False)

    # =============================================== THE END ====================================================

@torch.no_grad()
def val_fn(dataloader, model, device, config, mb, epoch):
    model.eval()
    
    vl_loss, vl_score, vl_steps = 0, 0, 0
    vl_preds, vl_targets = [], []
    for bi, d in enumerate(progress_bar(dataloader, parent=mb)):
        inputs = d["inputs"]
        labels = d['labels']
        # meta_features = d["meta_features"]

        #load input data to device
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        # meta_features = meta_features.to(device)

        outputs = model(inputs, 
                        # meta_features,
                        labels,
                        )

        preds, loss = outputs  

        labels = labels.to('cpu')
        preds = preds.detach().to('cpu')
        
        vl_preds.append(preds)
        vl_targets.append(labels)

        vl_loss += loss.item()
        vl_steps += 1  

        avg_vl_loss = vl_loss/vl_steps
    avg_vl_classwise_score, avg_vl_score = get_metrics(torch.cat(vl_targets, dim=0), torch.cat(vl_preds, dim=0))
    # config.train.classwise_metric = [float(x) for x in avg_vl_classwise_score]
    csv_logger.write([config.train.fold, "val", epoch, bi, avg_vl_loss, avg_vl_score])

    return avg_vl_loss, avg_vl_score, torch.cat(vl_targets, dim=0), torch.cat(vl_preds, dim=0)


@torch.enable_grad()
def train_fn(train_part,
            fold,
            epoch, 
            best_score, 
            train_dataloader, 
            valid_dataloader,
            model,
            device,
            optimizer,
            scheduler,
            config,
            mb, 
            tokenizer
            ):

    tr_loss, tr_score, tr_steps = 0, 0, 0
    
    for bi, d in enumerate(progress_bar(train_dataloader, parent=mb)):
        model.train()
        inputs = d["inputs"]
        labels = d['labels']
        # meta_features = d["meta_features"]

        #load input data to device
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        # meta_features = meta_features.to(device)

        outputs = model(inputs, 
                        # meta_features,
                        labels,
                        )

        preds, loss = outputs
        # print(loss)

        labels = labels.to('cpu')
        preds = preds.detach().to('cpu')
           
        tr_loss += loss.item()
        classwise_score, score = get_metrics(labels, preds)

        tr_score += score
        tr_steps += 1  

        t_loss = tr_loss/tr_steps
        t_score = tr_score/tr_steps
        
        csv_logger.write([fold, "train", epoch, bi, loss.item(), score])

        mb.child.comment = 't_loss: {:.4f} avg_t_loss: {:.4f} tr_score: {:.4f} avg_t_score: {:.4f}'.format(loss.item(), t_loss, score, t_score) 

        # accumulating gradients over steps
        if config.train.gradient_accumulate > 1:
            loss = loss / config.train.gradient_accumulate

        scaler.scale(loss).backward()

        # perform optimization step after certain number of accumulating steps and at the end of epoch
        if bi % config.train.gradient_accumulate == 0 or bi == len(train_dataloader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
                         
        ## evaluation for every # batches
        if ((epoch <= config.train.after_eval_epoch) & ((bi+1) == len(train_dataloader)) & (bi != 0)) | ((epoch > config.train.after_eval_epoch) & ((bi % config.train.eval_steps == 0) | ((bi+1) == len(train_dataloader))) & (bi != 0)):
            tr_time = timer.fsince('train')

            timer.add('val')
            v_loss, v_score, val_targets, val_preds = val_fn(valid_dataloader, model, device, config, mb, epoch)
            
            vl_time = timer.fsince('val')

            output = f"T_tm : {tr_time} - V_tm : {vl_time} - Ep : {epoch} - Bi : {bi} - Loss : {t_loss:.4f}; {v_loss:.4f} - Score : {t_score:.4f}; {v_score:.4f}"

            if v_score < best_score:
                best_score = v_score
                torch.save(model.state_dict(), os.path.join(config.weights_path, f"best_score_{train_part}_fold{fold}.pth"))
                log_score = 'best'
            else:
                log_score = ''
            
            logger.info(f"{output} - {log_score}")

    return model, best_score

if __name__ == '__main__':
    main()


