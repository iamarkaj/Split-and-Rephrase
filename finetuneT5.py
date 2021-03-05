"""
fine tune Hugging Face's T5 model
"""

!pip install -q transformers
!pip install -q sentencepiece


import torch
import warnings
import numpy as np
import pandas as pd
from torch import cuda
torch.cuda.empty_cache()
import torch.nn.functional as F
warnings.filterwarnings(action='ignore')
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration


device = 'cuda' if cuda.is_available() else 'cpu'


class CustomDataset(Dataset):
 
    def __init__(self, data, tokenizer, source_len, target_len):
        self.tokenizer  = tokenizer
        self.data       = data
        self.source_len = source_len
        self.target_len = target_len
        self.source     = self.data.source
        self.target     = self.data.target
 
    def __len__(self):
        return len(self.target)
 
    def __getitem__(self, index):
        source = self.source[index]
        target = self.target[index]
 
        source = self.tokenizer.batch_encode_plus([source], max_length= self.source_len, pad_to_max_length=True, truncation=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([target], max_length= self.target_len, pad_to_max_length=True, truncation=True, return_tensors='pt')
 
        source_ids  = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids  = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
 
        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }



def train(epoch, tokenizer, model, device, loader, optimizer):
 
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)
        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
 
        if _%4000==0:
            print('Epoch: {}, Loss:  {:.3f}'.format((epoch+1), loss.item()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def validate(epoch, tokenizer, model, device, loader, SUMMARY_LEN):
  
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]
            
        print('Epoch: {}, Val Loss {:.3f}'.format((epoch+1), loss.item()))



def main():

    TRAIN_BATCH_SIZE = 16  
    VALID_BATCH_SIZE = 16 
    EPOCHS           = 1      
    LEARNING_RATE    = 5e-5
    SEED             = 777             
    MAX_LEN          = 100
    SUMMARY_LEN      = 100
    TRAIN_DATA_PATH  = '/content/drive/MyDrive/split-and-rephrase/dataCSV/train300k.csv'
    VALID_DATA_PATH  = '/content/drive/MyDrive/split-and-rephrase/dataCSV/val5k.csv'
    MODEL_SAVE_PATH  = '/content/drive/MyDrive/split-and-rephrase/models_len_100_100_300k_utf8'
    

    torch.manual_seed(SEED)     # pytorch random seed
    np.random.seed(SEED)        # numpy random seed
    torch.backends.cudnn.deterministic = True
    

    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    

    train_data        = pd.read_csv(TRAIN_DATA_PATH,encoding='utf-8')
    train_data.source = 'summarize: ' + train_data.source
    print("TRAIN Dataset: {}".format(train_data.shape)) 
    train_set         = CustomDataset(train_data, tokenizer, MAX_LEN, SUMMARY_LEN)
    train_loader      = DataLoader(
        train_set,
        batch_size  = TRAIN_BATCH_SIZE,
        shuffle     = True,
        num_workers = 0


        )


    validation_data        = pd.read_csv(VALID_DATA_PATH,encoding='utf-8')
    validation_data.source = 'summarize: ' + validation_data.source
    print("VALIDATION Dataset: {}".format(validation_data.shape))
    validation_set        = CustomDataset(validation_data, tokenizer, MAX_LEN, SUMMARY_LEN)
    validation_loader     = DataLoader(
        validation_set, 
        batch_size  = VALID_BATCH_SIZE,
        shuffle     = False,
        num_workers = 0


        )
    

    model     = T5ForConditionalGeneration.from_pretrained("t5-base")
    model     = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    

    print('Initiating Fine-Tuning for the model on our dataset')
    for epoch in range(EPOCHS):
        train(epoch, tokenizer, model, device, train_loader, optimizer)  
        validate(epoch, tokenizer, model, device, validation_loader, SUMMARY_LEN)
        torch.save(model, MODEL_SAVE_PATH)


if __name__ == "__main__":
    print("Starting ...")
    main()
    print("End")