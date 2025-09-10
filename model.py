import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from train_utils import *

class AttentionPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x, mask):
        w = self.attention(x).float() #
        w[mask[:, 0]==0]=float('-inf')
        w = torch.softmax(w,1)
        x = torch.sum(w * x, dim=1)
        return x

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# m10 - last hidden state - only cls token
# m30 - all hidden states - only cls token

# m21 - last hidden states - all tokens - with mean pooling
# m22 - last hidden states - all tokens - with attention pooling
# m23 - last hidden states - all tokens - cls + mean pooling + attention pooling

# m41 - all hidden states - all tokens - with mean pooling
# m42 - all hidden states - all tokens - with attention pooling
# m43 - all hidden states - all tokens - mean pooling | attention pooling
# m44 - all hidden states - all tokens - mean pooling + attention pooling
# m45 - all hidden states - all tokens - with mean pooling + lstm
# m46 - all hidden states - all tokens - with mean pooling + reinit -1 encoder layer

class Model(nn.Module):
    def __init__(self, exp_config, pretrained=True):
        super().__init__()

        self.exp_config = exp_config  
        self.config = AutoConfig.from_pretrained(self.exp_config.model)

        if self.exp_config.model_config != None:
            self.config.update(self.exp_config.model_config)

        if pretrained:
            self.model = AutoModel.from_pretrained(self.exp_config.model, trust_remote_code=True, config=self.config)
        else:
            self.model = AutoModel(config=self.config)
        
        if self.exp_config.train.model[2] in ["1", "6"]:
            self.dropout1 = nn.Dropout(0.2)
            self.dropout2 = nn.Dropout(0.3)
            self.dropout3 = nn.Dropout(0.4)
            self.dropout4 = nn.Dropout(0.5)
            self.dropout5 = nn.Dropout(0.6)
            self.dropout6 = nn.Dropout(0.7)
        else:
            self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.2)
            self.dropout3 = nn.Dropout(0.3)
            self.dropout4 = nn.Dropout(0.4)
            self.dropout5 = nn.Dropout(0.5)
            self.dropout6 = nn.Dropout(0.6)

        if self.exp_config.train.model[:2] in ["m1", "m2", "m3", "m4"]:
            self.classifier = nn.Linear(self.config.hidden_size, self.exp_config.train.num_labels)
            self._init_weights(self.classifier)

        if self.exp_config.train.model[:2] in ["m3", "m4"]:
            n_weights = self.config.num_hidden_layers + 1
            weights_init = torch.zeros(n_weights).float()
            weights_init.data[:-1] = -3

            self.layer_weights = torch.nn.Parameter(weights_init)
            self._init_weights(self.layer_weights)

        # if self.exp_config.train.model[:2] in ["m5"]:
        #     self.classifier1 = nn.Linear(self.config.hidden_size, 1)
        #     self.classifier2 = nn.Linear(self.config.hidden_size, 1)
        #     self._init_weights(self.classifier1)
        #     self._init_weights(self.classifier2)


        if self.exp_config.train.model[2] in ["1"]:
            logger.info("Adding Mean Pooling layer")
            self.pool = MeanPooling()
            self._init_weights(self.pool)

        elif self.exp_config.train.model[2] in ["2"]:
            logger.info("Adding Attention Pooling layer")
            self.pool = AttentionPool(self.config.hidden_size)
            self._init_weights(self.pool)

        elif self.exp_config.train.model[2] in ["3"]:
            logger.info("Adding Mean Pooling and Attention Pooling layer")
            self.pool1 = MeanPooling()
            self.pool2 = AttentionPool(self.config.hidden_size)
            self._init_weights(self.pool1)
            self._init_weights(self.pool2)
            self.classifier = nn.Linear(self.config.hidden_size*2, self.exp_config.train.num_labels)
            self._init_weights(self.classifier)

        elif self.exp_config.train.model[2] in ["4"]:
            logger.info("Adding Mean Pooling and Attention Pooling layer")
            self.pool1 = MeanPooling()
            self.pool2 = AttentionPool(self.config.hidden_size)
            self._init_weights(self.pool1)
            self._init_weights(self.pool2)

        elif self.exp_config.train.model[2] in ["5"]:
            self.pool = MeanPooling()
            self.lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size//2, 2, batch_first=True, bidirectional=True, dropout=0.2)
            self._init_weights(self.pool)
            self._init_weights(self.lstm)

        elif self.exp_config.train.model[2] in ["6"]:
            logger.info("Adding Mean Pooling layer and Re-init 1 last encoder layer")
            self.pool = MeanPooling()
            self._init_weights(self.pool)
            self.model = self.re_initializing_layer(self.model, 1)

    
    def _init_weights(self, module):
        # print(type(module))
        if isinstance(module, nn.Linear):
            # print("L", dir(module))
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # print("E")
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def re_initializing_layer(self, model, layer_num):
        for module in model.encoder.layer[-layer_num:].modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        return model

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         nn.init.xavier_normal_(module.weight)
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)


    def feature(self, inputs):

        outputs = self.model(**inputs)

        # last hidden state - only cls token
        if self.exp_config.train.model[:2] in ["m1"]:
            features = outputs[0][:, 0, :]

        # last hidden states - all tokens
        elif self.exp_config.train.model[:2] in ["m2", "m5"]:
            features = outputs[0]

        # all hidden states - only cls token
        elif self.exp_config.train.model[:2] in ["m3"]:
            if 'deberta' in self.exp_config.model:
                hidden_layers = outputs[1]
            else:
                hidden_layers = outputs[2]
            n_hidden_states = torch.stack([layer[:, 0, :] for layer in hidden_layers], dim=-1)
            features = (torch.softmax(self.layer_weights, dim=0) * n_hidden_states).sum(-1)

        # all hidden states - all tokens
        elif self.exp_config.train.model[:2] in ["m4"]:
            if 'deberta' in self.exp_config.model:
                hidden_layers = outputs[1]
            else:
                hidden_layers = outputs[2]
            n_hidden_states = torch.stack([layer for layer in hidden_layers], dim=-1)
            features = (torch.softmax(self.layer_weights, dim=0) * n_hidden_states).sum(-1)

        return features

    def forward(self, inputs, labels=None):
        feature = self.feature(inputs)
        out = feature
        # print(self.exp_config.train.model[:2], self.exp_config.train.model[2])

        if self.exp_config.train.model[2] in ["1", "2", "6"]:
            out = self.pool(out, inputs['attention_mask'])
            # out = self.pool(out[:, 1:3, :], inputs['attention_mask'][:, 1:3])


        elif self.exp_config.train.model[2] in ["3"]:
            # cls_emd = out[:, 0, :]
            avg_pool = self.pool1(out[:, :, :], inputs['attention_mask'][:, :])
            att_pool = self.pool2(out[:, :, :], inputs['attention_mask'][:, :])
            out = torch.cat([avg_pool, att_pool], dim=-1)

        elif self.exp_config.train.model[2] in ["4"]:
            # cls_emd = out[:, 0, :]
            avg_pool = self.pool1(out[:, :, :], inputs['attention_mask'][:, :])
            att_pool = self.pool2(out[:, :, :], inputs['attention_mask'][:, :])
            out = avg_pool + att_pool


        elif self.exp_config.train.model[2] in ["5"]: 
            out = self.pool(out, inputs['attention_mask'])
            lstm, (_, _) = self.lstm(out.unsqueeze(1))
            out = lstm.squeeze(1)

        logits1 = self.classifier(self.dropout1(out))
        logits2 = self.classifier(self.dropout2(out))
        logits3 = self.classifier(self.dropout3(out))
        logits4 = self.classifier(self.dropout4(out))
        logits5 = self.classifier(self.dropout5(out))
        logits6 = self.classifier(self.dropout6(out))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5 + logits6) / 6
        # print(logits.shape, labels.shape)

        if labels is not None:
            loss1 = loss_fn(logits1, labels, self.model.device, self.exp_config)
            loss2 = loss_fn(logits2, labels, self.model.device, self.exp_config)
            loss3 = loss_fn(logits3, labels, self.model.device, self.exp_config)
            loss4 = loss_fn(logits4, labels, self.model.device, self.exp_config)
            loss5 = loss_fn(logits5, labels, self.model.device, self.exp_config)
            loss6 = loss_fn(logits6, labels, self.model.device, self.exp_config)

            loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6) / 6
        return logits, loss