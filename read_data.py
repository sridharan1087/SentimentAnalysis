import pandas as pd
from pandas.io.parsers import count_empty_vals
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch import tensor
import tensorflow as tf

class sentiment_data(Dataset):
    """Sentiment Analysis Data"""

    def __init__(self,data):
        """
        Args:
            data (string): path to csv file.
        """
        self.data = pd.read_csv(data)
        self.word_dictionary = dict()
        self.count = 0
        self.max_len = 150
        self.vocab_size = 0

    def __len__(self):
        return len(self.data)

    def vocabsize(self):
        for j,i in self.data.iterrows():
            words = i['tweet'].split()
            for n,word in enumerate(words):
                if word in self.word_dictionary:
                    continue
                self.word_dictionary[word] = self.vocab_size
                self.vocab_size+=1
            

        vocab_size = self.vocab_size
        return vocab_size

    def __getitem__(self,idx):
        """
        Args:
        idx (int): index value of the data frame.
        return: tensors of tweet of size 150 and label
        """
        words = self.data.loc[idx,'tweet'].split()
        for n,word in enumerate(words):
            if word in self.word_dictionary:
                continue
            self.word_dictionary[word] = self.count
            self.count+=1
        sentence2vec = [self.word_dictionary[w] for w in words]   
        padding = self.max_len - len(sentence2vec)
        if padding > 0:
            sentence2vec.extend([0]*padding)
        else:
            sentence2vec[:padding]

        return tensor(sentence2vec),tensor(self.data.loc[idx,'label'])


"""
#Initializing setiment_data
sentiment_dataset = sentiment_data('train_2kmZucJ.CSV')

print(sentiment_dataset.__len__())
"""


"""
#Uncomment to iterate through dataset.
for i in range(sentiment_dataset.__len__()):
    print(sentiment_dataset[i])
    if i == 10:
        break

"""

"""
#How to use DataLoader
dataloader = DataLoader(sentiment_dataset,batch_size=64)
for i in dataloader:
    print(tf.argmax(i[0]))  #64*150
"""
        



    