#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import mlcrate.time as mlctime
from omegaconf import OmegaConf
from argparse import ArgumentParser
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold, KFold

from data_utils import preprocessing
from train_utils import seed_everything_custom

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-config", required=True)
    parser.add_argument("options", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(args.options)


    timer = mlctime.Timer()
    print(mlctime.now())
    print(f"config: {config}")
    
    seed_everything_custom(config.seed)
    print(f"Seed set to {config.seed}")

    config.input_path = os.path.join(config.root, config.input_path, config.competition)

    # ===================================== DATA PREPARATION =====================================
    prompts_df = pd.read_csv(f'{config.input_path}/prompts_train.csv')
    summary_df = pd.read_csv(f'{config.input_path}/summaries_train.csv')

    df = summary_df.merge(prompts_df, on="prompt_id")
    print(f"shape of the train df: {df.shape}")

    # ===================================== CV SPLIT ===========================================

    # # c1     
    # nFolds = GroupKFold(n_splits=config.train.n_folds)
    # for n, (train_index, val_index) in enumerate(nFolds.split(df, groups=df["prompt_id"])):
    #     df.loc[val_index, 'fold'] = int(n)
    # df['fold'] = df['fold'].astype(int)


    # # c2
    # df['full_text'] = df["prompt_title"] + " " + df["prompt_text"]+ " " + df["prompt_question"]+ " " + df["text"]
    # df['full_text_processed'] =  df['full_text'].apply(lambda x: preprocessing(x, "p1"))
    # df['full_text_len'] = df['full_text_processed'].apply(lambda x: len(x.split()))
    # df['unique_word_len'] = df['full_text_processed'].apply(lambda x: len(set(x.split())))

    # label_cols = ["content", "wording", "full_text_len", "unique_word_len"]
 
    # nFolds = MultilabelStratifiedKFold(n_splits=config.train.n_folds, shuffle=True, random_state=config.seed)
    # for n, (train_index, val_index) in enumerate(nFolds.split(df, y=df[label_cols].values)):
    #     df.loc[val_index, 'fold'] = int(n)
    # df['fold'] = df['fold'].astype(int)   


    # c3
    df['full_text'] = df["prompt_title"] + " " + df["prompt_text"]+ " " + df["prompt_question"]+ " " + df["text"]
    df['full_text_processed'] =  df['full_text'].apply(lambda x: preprocessing(x, "p1"))
    df['full_text_len'] = df['full_text_processed'].apply(lambda x: len(x.split()))
    df['unique_word_len'] = df['full_text_processed'].apply(lambda x: len(set(x.split())))

    num_bins = int(np.floor(1+(3.3)*(np.log2(len(df)))))
    print(num_bins)
    df[f"content_bins"] = pd.cut(df[f"content"], bins=num_bins, labels=False)
    df[f"wording_bins"] = pd.cut(df[f"wording"], bins=num_bins, labels=False)
   
    label_cols = ["content_bins", "wording_bins", "full_text_len", "unique_word_len"]
 
    nFolds = MultilabelStratifiedKFold(n_splits=config.train.n_folds, shuffle=True, random_state=config.seed)
    for n, (train_index, val_index) in enumerate(nFolds.split(df, y=df[label_cols].values)):
        df.loc[val_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)   




   

    print(df.groupby('fold').size())
    df.to_csv(os.path.join(config.input_path, f"train_fold{config.train.n_folds}_seed{config.seed}_c3.csv"), index=False)

    
if __name__ == '__main__':
    main()


