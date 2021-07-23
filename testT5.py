"""
test finetuned T5 model
"""


import re
import numpy as np
import pandas as pd
import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda
from sentenceninja import split


device = 'cuda' if cuda.is_available() else 'cpu'


class CustomTestDataset(Dataset):

    def __init__(self, data, tokenizer, source_len):
        self.tokenizer = tokenizer
        self.data = data
        self.source_len = source_len
        self.source = self.data.source
    def __len__(self):
        return len(self.source)
    def __getitem__(self, index):
        source = self.source[index]
        source = self.tokenizer.batch_encode_plus([source], max_length= self.source_len, pad_to_max_length=True, truncation=True, return_tensors='pt')
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
        }



def test(epoch, tokenizer, model, device, loader, SUMMARY_LEN):

    model.eval()
    predictions = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            generated_ids = model.generate(
                input_ids          = ids,
                attention_mask     = mask,
                max_length         = SUMMARY_LEN,
                num_beams          = 12,
                repetition_penalty = 2.5,
                length_penalty     = 1.0,
                early_stopping     = True,
                #do_sample         = True


                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            predictions.extend(preds)
    return predictions



def main(input):

    TEST_BATCH_SIZE = 2
    SEED            = 777
    SUMMARY_LEN     = 100


    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True


    tokenizer = T5Tokenizer.from_pretrained("t5-base")


    # available trained model names
    #len_100_100_50k means source and target max length is 100 respectively and dataset contains 50k sentences

    # models_len_100_100_50k
    # models_len_100_100_200k          # i prefer
    # models_len_90_90_200k_try2
    # models_len_100_100_300k          # i prefer
    # models_len_100_100_300k_utf8
    # models_len_100_100_500k   # was trained only for 1 epoch, and it needs to be trained on 1 more epoch to get better result


    MODEL_PATH = 'models_len_100_100_300k'
    #model = torch.load(MODEL_PATH, map_location=torch.device('cpu')) # if cpu
    model = torch.load(MODEL_PATH) # if gpu
    model = model.to(device)


    test_data = pd.DataFrame(input, columns= ['source'])
    test_data.source = 'summarize: ' + test_data.source
    test_set = CustomTestDataset(test_data, tokenizer, SUMMARY_LEN)
    test_params = {
        'batch_size': TEST_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    test_loader = DataLoader(test_set, **test_params)
    epoch       = 1
    predictions = test(epoch, tokenizer, model, device, test_loader, SUMMARY_LEN)
    return predictions
