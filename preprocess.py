"""
Saving the data into csv
Format: Adding the target text in front of the source text
        because this is how T5 model was trained for summarization task
"""

!pip install -q unidecode
import unidecode
import pandas as pd


TRAIN_SAMPLE = 300_000
TRAIN_SRC    = '/content/drive/MyDrive/split-and-rephrase/data/train.source'
TRAIN_TARGET = '/content/drive/MyDrive/split-and-rephrase/data/train.target'
TRAIN_SAVE   = '/content/drive/MyDrive/split-and-rephrase/dataCSV/train300k.csv'


VALID_SAMPLE = 5_000
VALID_SRC    = '/content/drive/MyDrive/split-and-rephrase/data/validation.source'
VALID_TARGET = '/content/drive/MyDrive/split-and-rephrase/data/validation.target'
VALID_SAVE   = '/content/drive/MyDrive/split-and-rephrase/dataCSV/val5k.csv'


def save_to_csv(SRC, TARGET, SAMPLE, SAVE):

    source = pd.read_csv(SRC, header=None, names=['source'], delimiter="\n", encoding='utf-8')
    target = pd.read_csv(TARGET, header=None, names=['target'], delimiter="\n", encoding='utf-8')

    data           = pd.DataFrame()
    data['source'] = np.squeeze(source.to_numpy())
    data['target'] = np.squeeze(target.to_numpy())
    data           = data.sample(SAMPLE) # randomly select data
    data['source'] = data['source'].apply(lambda x: unidecode.unidecode(x))
    data['target'] = data['target'].apply(lambda x: unidecode.unidecode(x))
    
    data.to_csv(SAVE,index=False)


save_to_csv(TRAIN_SRC, TRAIN_TARGET, TRAIN_SAMPLE, TRAIN_SAVE)
save_to_csv(VALID_SRC, VALID_TARGET, VALID_SAMPLE, VALID_SAVE)