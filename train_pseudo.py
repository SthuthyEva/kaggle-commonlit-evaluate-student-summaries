#!/usr/bin/env python
# coding: utf-8
import os
import re
import json
import shutil
import numpy as np
import pandas as pd
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
from transformers import AutoTokenizer

from data_utils import *
from train_utils import *
from dataset import *
from model import *

import warnings
warnings.filterwarnings("ignore")

os.environ['TOKENIZERS_PARALLELISM']= 'true'

# The following is necessary if you want to use the fast tokenizer for deberta v2 or v3
# This must be done before importing transformers
trfms_path = "/home/sthueva/anaconda3/envs/kaggle/lib/python3.7/site-packages/transformers"
deberta_v2_3_tok_path = "../../input/nbroad/deberta-v2-3-fast-tokenizer"

# trfms_path = "/home/yuv4r4j/anaconda3/envs/kaggle/lib/python3.7/site-packages/transformers"
# deberta_v2_3_tok_path = "/home/yuv4r4j/workspace/input/nbroad/deberta-v2-3-fast-tokenizer"

transformers_path = Path(trfms_path)
input_dir = Path(deberta_v2_3_tok_path)

convert_file = input_dir / "convert_slow_tokenizer.py"
conversion_path = transformers_path/convert_file.name

if conversion_path.exists():
    conversion_path.unlink()

shutil.copy(convert_file, transformers_path)
deberta_v2_path = transformers_path / "models" / "deberta_v2"

for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py']:
    filepath = deberta_v2_path/filename
    if filepath.exists():
        filepath.unlink()
    shutil.copy(input_dir/filename, filepath)

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

    config.exp_name = config.exp_name + f"-f{config.train.n_folds}-s{config.seed}-e{config.train.body_epochs}{config.train.scheduler_epochs}-b{config.train_loader.batch_size}-d{config.model_config.hidden_dropout_prob}-{'%.e' % config.optimizer.lr}-{'%.e' % config.optimizer.head_lr}-wd{config.optimizer.weight_decay}-wm{config.scheduler.num_warmup_steps}-{config.train.model}-x{config.train.max_length}"
    config.exp_name = re.sub('-0', '', config.exp_name)
    config.exp_name = re.sub('\.', '', config.exp_name)
    # config.exp_name = re.sub('0-', '', config.exp_name)
    slug = f"{config.exp_name}"
    
    assert (len(slug)>6) & (len(slug)<50), 'Exp name length should be greater than 6 and lesser than 50'

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
    if "deberta-v2" in config.model or "deberta-v3" in config.model:
        from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
        tokenizer = DebertaV2TokenizerFast.from_pretrained(config.model, use_fast=True)
        # tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

    elif "coco" in config.model:
        tokenizer = COCOLMTokenizer.from_pretrained(config.model, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
    logger.info(f"Fast tokenizer : {tokenizer.is_fast}")
    
    # ===================================== DATA PREPARATION =====================================
    df = pd.read_csv(os.path.join(config.input_path, config.input_filename))
    logger.info(f"shape of the train df: {df.shape}")

    label_cols = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

    # ===================================== DEBUG MODE =====================================
    if config.train.debug:
        df = df.sample(200, random_state=config.seed)
        df.reset_index(drop=True, inplace=True)
        logger.info(f"Running in debug mode with less samples")

    # ===================================== CV SPLIT ===========================================
    logger.info(df.groupby('fold').size())

    # ===================================== DATA PREPROCESSING ===========================================
    
    df['full_text'] = df['full_text'].apply(lambda x : resolve_encodings_and_normalize(x))
    # df['full_text'] = df['full_text'].apply(lambda x : tokenizer.backend_tokenizer.normalizer.normalize_str(x))
    df['full_text'] =  df['full_text'].apply(lambda x: preprocessing(x))
    df['full_text_len'] = df['full_text'].apply(lambda x: len(tokenizer(x)['input_ids']))

    if config.train.use_case == "regression":
        if config.train.loss == "sl1":
            df[label_cols] = df[label_cols]
        elif config.train.loss == "bce":
            df[label_cols] = df[label_cols]/5.0
        # df[label_cols] = (df[label_cols] - 1.0)/ 4.0
    else:
        for label in label_cols:
            df[label] = df[label].apply(lambda x: config.train.score2label[x])

    logger.info(f"Min length in full text: {df['full_text_len'].min()}")
    logger.info(f"Max length in full text: {df['full_text_len'].max()}")
    # config.train.max_length = int(df['full_text_len'].max())
    logger.info(f"No.of samples longer than max_length: {df[df['full_text_len']> config.train.max_length].shape[0]}")

    logger.info(f"Tokenizer all special tokens : {len(tokenizer.all_special_tokens)}")
    # tokenizer.add_special_tokens({'additional_special_tokens': ['[BR]']})
    # logger.info(f"Tokenizer all special tokens after addition : {len(tokenizer.all_special_tokens)}")
    
    # ===================================== KFOLD TRAINING =====================================
    
    fold = config.train.fold
    logger.info(f"------------- {fold+1} of {config.train.n_folds} Folds -------------")
    
    # TRAIN & VALID DATA
    prev_exp = re.sub("P", "", config.exp_name)
    prev_out = re.sub("P", "", config.output_path)
    prev_weights = re.sub("P", "", config.weights_path)
    
    train_set = df[df['fold'] != fold]
    pseudo_set = pd.read_csv(os.path.join(prev_out, f"{prev_exp}_pseudo_fold{fold}.csv"))
    pseudo_set.rename(columns={"essay_id":"text_id", "essay":"full_text","essay_len":"full_text_len"}, inplace=True)
    pseudo_set["fold"] = fold

    if config.train.use_case == "regression":
        if config.train.loss == "sl1":
            pseudo_set[label_cols] = pseudo_set[label_cols]
        elif config.train.loss == "bce":
            pseudo_set[label_cols] = pseudo_set[label_cols]/5.0
        # df[label_cols] = (df[label_cols] - 1.0)/ 4.0
    else:
        for label in label_cols:
            pseudo_set[label] = pseudo_set[label].apply(lambda x: config.train.score2label[x])
    
    train_set = pd.concat([train_set, pseudo_set], axis=0)
    # train_set = pseudo_set

    valid_set = df[df['fold'] == fold]
    valid_set.sort_values(by=['full_text_len'], inplace=True)

    train_set.reset_index(drop=True, inplace=True)      
    valid_set.reset_index(drop=True, inplace=True)      

    logger.info(f"TRAIN Dataset: {train_set.shape}")
    logger.info(f"VALID Dataset: {valid_set.shape}")
    logger.info(f"Train - No.of samples shorter than max_length: {train_set[train_set['full_text_len'] <= config.train.max_length].shape[0]}")
    logger.info(f"Train - No.of samples longer than max_length: {train_set[train_set['full_text_len'] > config.train.max_length].shape[0]}")    
    logger.info(f"Valid - No.of samples shorter than max_length: {valid_set[valid_set['full_text_len'] <= config.train.max_length].shape[0]}")
    logger.info(f"Valid - No.of samples longer than max_length: {valid_set[valid_set['full_text_len'] > config.train.max_length].shape[0]}")    
    
    # DATASET
    train_dataset = FeedbackDataset(train_set, 
                                    label_cols,
                                    tokenizer, 
                                    config.train.max_length, 
                                    if_collate=True, 
                                    )
    valid_dataset = FeedbackDataset(valid_set, 
                                    label_cols,
                                    tokenizer, 
                                    config.train.max_length, 
                                    if_collate=True,
                                    )
    # logger.info(f"TRAIN Dataset: {train_dataset.__getitem__(0)}")

    # DEVICE
    device = torch.device("cuda")

    # MODEL 
    if config.train.model == 'm1':
        model = Model1(config)
    elif config.train.model == 'm2':
        model = Model2(config)
    elif config.train.model == 'm3':
        model = Model3(config)
    elif config.train.model == 'm32':
        model = Model32(config)
    elif config.train.model == 'm4':
        model = Model4(config)
    elif config.train.model == 'm51':
        model = Model5(config)
    elif config.train.model == 'm6':
        model = Model6(config)
    elif config.train.model == 'm7':
        model = Model7(config)
    elif config.train.model == 'm8':
        model = Model8(config)
    elif config.train.model == 'm9':
        model = Model9(config)

    # for idx, (n, p) in enumerate(model.named_parameters()):
    #     if idx <= config.train.num_freeze:
    #         p.requires_grad = False
    
    # for idx, (n, p) in enumerate(model.named_parameters()):
    #     logger.info(f"{idx} - {n} - {p.requires_grad}")

    # gradient checkpointing
    model.model.gradient_checkpointing_enable()
    logger.info(f"Gradient Checkpointing: {model.model.is_gradient_checkpointing}")
        
    model.to(device)
    logger.info(f"Token Embedding shape : {model.model.get_input_embeddings()}")

    # state_dict = torch.load(os.path.join(prev_weights, f"best_score_body_fold{fold}.pth"), map_location=device)    
    # model.load_state_dict(state_dict)
    
    # model.model._resize_token_embeddings(len(tokenizer))
    # logger.info(f"Token Embedding shape after adding special tokens : {model.model.get_input_embeddings()}")

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
    v_loss, v_score, val_targets, val_preds = val_fn(valid_dataloader, model, device, config, mb, epoch)
    
    for idx, label in enumerate(label_cols):
        valid_set[f"{label}_predictions"] = val_preds[:, idx].tolist()
        if config.train.use_case == "regression":
            if config.train.loss == "sl1":
                valid_set[label] = valid_set[label]
            elif config.train.loss == "bce":
                valid_set[label] = 5.0 * valid_set[label]
                # valid_set[label] = (valid_set[label] + 1.0) * 4.0
        else:
            valid_set[label] = valid_set[label].apply(lambda x: config.train.label2score[x])

    valid_set.drop(["full_text", "full_text_len"], axis=1, inplace=True)
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
        pred_cols = [f"{label}_predictions" for label in label_cols]
        for fold in range(config.train.n_folds):
            df_predictions = pd.read_csv(os.path.join(config.output_path, f"predictions_fold{fold}.csv")) 
            fold_predictions.append(df_predictions)
            fold_classwise_metric, fold_wise_metric = get_metrics(torch.tensor(df_predictions[pred_cols].values), torch.tensor(df_predictions[label_cols].values))
            logger.info(f"Classwise RMSE Score for fold{fold} - {['%.6f' % elem for elem in fold_classwise_metric]}")
            logger.info(f"RMSE Score for fold{fold} - {'%.6f' % fold_wise_metric}")
        
        
        fold_predictions = pd.concat(fold_predictions, axis=0)
        classwise_metric, overall_metric = get_metrics(torch.tensor(fold_predictions[pred_cols].values), torch.tensor(fold_predictions[label_cols].values))
        logger.info(f"Classwise RMSE CV Score - {['%.6f' % elem for elem in classwise_metric]}")
        logger.info(f"RMSE CV Score - {'%.6f' % overall_metric}")

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

        #load input data to device
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)

        outputs = model(inputs, 
                        labels)

        preds, loss = outputs  

        if config.train.use_case == "regression":
            if config.train.loss == "sl1":
                labels = labels.to('cpu')
                preds = preds.detach().to('cpu')
            elif config.train.loss == "bce":
                labels = 5.0 * labels.to('cpu')
                preds = 5.0 * torch.sigmoid(preds).detach().to('cpu')
                
                # labels = 4.0 * (labels.to('cpu') + 1.0)
                # preds = 4.0 * (torch.sigmoid(preds).detach().to('cpu') + 1.0)
        else:
            labels = labels.to('cpu')
            label_list=[]
            pred_list=[]
            for idx, pred in enumerate(preds):
                pred = torch.argmax(pred, dim=-1).detach().to('cpu')
                pred = torch.tensor([config.train.label2score[x] for x in pred.tolist()])
                pred = pred.unsqueeze(-1)
                pred_list.append(pred)

                lbl = labels[:, idx]
                lbl = torch.tensor([config.train.label2score[x] for x in lbl.tolist()])
                lbl = lbl.unsqueeze(-1)
                label_list.append(lbl)
        
            labels = torch.cat(label_list, dim=-1)
            preds = torch.cat(pred_list, dim=-1)
        
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
        labels = d["labels"]

        # load input data to device
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)

        # print(inputs["input_ids"].shape, labels.shape)
        
        with autocast():  
            outputs = model(inputs, 
                            labels)

        preds, loss = outputs
        # print(loss)

        if config.train.use_case == "regression":
            if config.train.loss == "sl1":
                labels = labels.to('cpu')
                preds = preds.detach().to('cpu')
            elif config.train.loss == "bce":
                labels = 5.0 * labels.to('cpu')
                preds = 5.0 * torch.sigmoid(preds).detach().to('cpu')

            # labels = 4.0 * (labels.to('cpu') + 1.0)
            # preds = 4.0 * (torch.sigmoid(preds).detach().to('cpu') + 1.0)
        else:
            labels = labels.to('cpu')
            label_list=[]
            pred_list=[]
            for idx, pred in enumerate(preds):
                pred = torch.argmax(pred, dim=-1).detach().to('cpu')
                pred = torch.tensor([config.train.label2score[x] for x in pred.tolist()])
                pred = pred.unsqueeze(-1)
                pred_list.append(pred)

                lbl = labels[:, idx]
                lbl = torch.tensor([config.train.label2score[x] for x in lbl.tolist()])
                lbl = lbl.unsqueeze(-1)
                label_list.append(lbl)
        
            labels = torch.cat(label_list, dim=-1)
            preds = torch.cat(pred_list, dim=-1)


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


