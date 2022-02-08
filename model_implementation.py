import torch.nn as nn
import torch.nn.functional as ff
import torch

class LSTM_Model(nn.Module):
    """
    Lstm Implementaion
    """

    def __init__(self,vocab_size,embedding_dim) -> None:
        """
        Args:
        vocab_size (int): Vocabulary size
        embedding_dim (int): Size of embedding dim
        """
        super(LSTM_Model,self).__init__()

        #Initializing random hidden state and cell state values.
        #150 -> is the length of the input sentence along with padding
        #20 -> is the hidden dimensition to provide.
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html.
        self.hidden_dim = (torch.randn(1,64,20),torch.randn(1,64,20))
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim,hidden_size = 20,num_layers=1,batch_first=True)
        self.linear = nn.Linear(3000,2)

    def forward(self,inputs):
        self.inputs = inputs
        #print("Input Shape",inputs.shape)
        embedding_vector = self.embedding(inputs)
        #print("Embedding Shape",embedding_vector.shape)
        lstm_out,hidden = self.lstm(embedding_vector.view(len(inputs),150,-1))
        #print("lstm_out shape",lstm_out.shape)
        hidden_state,cell_state = hidden
        #print('hidden state shape',hidden_state.shape)
        #print('cell state shape',cell_state.shape)
        linear = self.linear(lstm_out.reshape((lstm_out.shape[0],-1)))
        out1 = torch.sigmoid(linear)
        return out1[:,-1]

