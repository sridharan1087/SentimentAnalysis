import torch.nn as nn
import torch.optim as optim
from model_implementation import LSTM_Model
from read_data import sentiment_data
from torch.utils.data import DataLoader
import torch
import numpy as np

sentiment_dataset = sentiment_data('train_2kmZucJ.CSV')
loss_function = nn.BCELoss()
model = LSTM_Model(sentiment_dataset.vocabsize(),100)
optimizer = optim.Adam(model.parameters(), lr=0.001)
dataloader = DataLoader(sentiment_dataset,batch_size=64)

losses = []
for epoch in range(10):
    print('Epoch',epoch)
    total_loss = 0
    for word in dataloader:
        sentence,label = word[0],word[1]
        model.zero_grad()
        
        word_embedding = model(sentence.type(torch.LongTensor))
        #print(word_embedding.shape)
        #print(word_embedding)
        loss = loss_function(word_embedding,label.float())
        #print('loss',loss.item())
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()
    losses.append(total_loss)

print(losses)
#torch.save(model,r'E:\AI-DL-ML\SentimentAnalysis\model.pkl')

#############################Testing a model #####################
# test_sentiment_dataset = sentiment_data(r'E:\AI-DL-ML\SentimentAnalysis\test_oJQbWVk.csv')
# test_dataloader = DataLoader(sentiment_dataset,batch_size=64)
# num_correct = 0 

# for word in test_dataloader:
#     sentence,label = word[0],word[1]
#     predicted = model(sentence)
#     #print(torch.round(predicted),label)
#     correct_tensor = predicted.eq(label.float().view_as(predicted))
#     correct = np.squeeze(correct_tensor.numpy()) 
#     num_correct += np.sum(correct)
#     num_correct/len(word[0])
#     print(num_correct)
