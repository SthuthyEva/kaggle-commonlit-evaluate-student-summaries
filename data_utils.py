import re
import string
import numpy as np
import pandas as pd
from torch import tensor, cat
from torch.nn.utils.rnn import pad_sequence 

import codecs
from text_unidecode import unidecode
from typing import Dict, List, Tuple
from bs4 import BeautifulSoup

# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords


# lemmatizer = WordNetLemmatizer()
# stop = stopwords.words('english')
 
from autocorrect import Speller
speller = Speller(lang='en')

## Preprocessing if required
def preprocessing(text, preprocess, spell=False):
    
    if preprocess == 'p1':
        text = re.sub('\r\n', ' [BR] ', text)
        text = re.sub(' +', ' ', text)
        text = text.strip()

    elif preprocess == 'p2':
        text = re.sub('\r\n', ' [BR] ', text)
        text = re.sub(' +', ' ', text)
        text = text.strip()
        text = text.lower()

    elif preprocess == 'p3':
        text = re.sub('\r\n', ' [BR] ', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(' +', ' ', text)
        text = text.strip()
        text = text.lower()

    elif preprocess == 'P4':
        '''
        Cleans text into a basic form for NLP. Operations include the following:-
        1. Remove special charecters like &, #, etc
        2. Removes extra spaces
        3. Removes embedded URL links
        4. Removes HTML tags
        5. Removes emojis

        text - Text piece to be cleaned.
        '''
        template = re.compile(r'https?://\S+|www\.\S+')  # Removes website links
        text = template.sub(r'', text)

        soup = BeautifulSoup(text, 'lxml')  # Removes HTML tags
        only_text = soup.get_text()
        text = only_text

        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        text = re.sub(r"[^a-zA-Z\d]", " ", text) # Remove special Charecters
        text = re.sub('\n+', '\n', text) 
        text = re.sub('\.+', '.', text) 
        text = re.sub(' +', ' ', text) # Remove Extra Spaces 

    elif preprocess == 'p4':
        text = re.sub('\r\n', ' [BR] ', text)
        text = re.sub(r'https?://\S+|www\.\S+', ' social medium ', text) 
        text = re.sub(' +', ' ', text)
        text = text.strip()

    elif preprocess == 'p5':
        text = re.sub('\r\n', ' [BR] ', text)
        text = re.sub(r'https?://\S+|www\.\S+', ' social medium ', text) # Removes website links
        text = re.sub(r"[^a-zA-Z\d]", " ", text) # Remove special Charecters
        # text = ' '.join([word for word in text.split(' ') if word not in stop]) # del stopwords
        text = re.sub(' +', ' ', text)
        text = text.strip()

    elif preprocess == 'p6':
        text = re.sub('\r\n', ' [BR] ', text)
        text = re.sub(r'https?://\S+|www\.\S+', ' social medium ', text) 
        if spell:
            text = speller(text)
        text = re.sub(' +', ' ', text)
        text = text.strip()

    elif preprocess == 'p7':
        text = re.sub('\r\n', ' [BR] ', text)
        text = re.sub('\n\n', ' [BR] ', text)
        text = re.sub(r'https?://\S+|www\.\S+', ' social medium ', text) 
        text = re.sub(' +', ' ', text)
        text = text.strip()

    elif preprocess == 'p8':
        text = re.sub('\r\n', ' [BR] ', text)
        text = re.sub(r'https?://\S+|www\.\S+', ' social medium ', text) 
        text = re.sub('\.+', '.', text) 
        text = re.sub(' +', ' ', text)
        text = text.strip()

    # elif preprocess == 'p6':
    #     text = re.sub('\r\n', ' [BR] ', text)
    #     text = re.sub(r'https?://\S+|www\.\S+', ' social medium ', text) # Removes website links


    #     text = re.sub(' +', ' ', text)
    #     text = text.strip()

    return text


def custom_collate(data): 
    # inputs
    inputs={}
    for _key, _value in data[0]['inputs'].items():
        inputs_list = []
        for item in data:
            inputs_list.append(item['inputs'][_key])
        inputs[_key] = pad_sequence(inputs_list, batch_first=True)
    
    # labels   
    labels=[]
    for item in data:
        labels.append(item['labels'])
    labels = cat(labels, dim=0)

    # # meta_features   
    # meta_features=[]
    # for item in data:
    #     meta_features.append(item['meta_features'])
    # meta_features = cat(meta_features, dim=0)

    return {'inputs':inputs, 
            'labels': labels, 
            # 'meta_features':meta_features,
            } 


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end

# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text

    
