
import numpy as np
from torch import tensor, long, int32, float32, full, bernoulli
from torch.utils.data import Dataset

class CommonlitDataset(Dataset):
    def __init__(self, df, label_cols, tokenizer, max_length, if_collate, config):
        self.len = len(df)
        self.df = df
        self.label_cols = label_cols
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.if_collate = if_collate # for validation
        if self.if_collate: 
            self.padding = 'do_not_pad'
        else:
            self.padding = 'max_length' # for validation
        if config.train.max_length == "None":
            self.truncation = "do_not_truncate"
        else:
            self.truncation = True 
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        text = self.df.full_text_processed[index]
        label = self.df.loc[index, self.label_cols]
        # meta_feature = self.df.loc[index, self.meta_features]
        # print(label, meta_feature)
        
        # CREATE INPUT IDS
        encoding_inputs = self.tokenizer(text,
                                        None,
                                        add_special_tokens=True,
                                        max_length=self.max_length, 
                                        truncation=self.truncation,
                                        padding=self.padding, 
                                        )
                    
        # CONVERT TO TORCH TENSORS
        inputs = {key: tensor(val, dtype=long) for key, val in encoding_inputs.items()}

        # regression
        labels = tensor(label, dtype=float).unsqueeze(0)

        # meta_features = tensor(meta_feature, dtype=long).unsqueeze(0)
        

        # # RANDOM MASKING
        # probability_matrix = full(inputs['input_ids'].shape, 0.03)
        # masked_indices = bernoulli(probability_matrix).bool()
        # inputs['input_ids'][masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        return {'inputs':inputs, 
                'labels': labels, 
                # 'meta_features':meta_features
                }


