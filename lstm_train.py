######## IMPORTS ########
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

#########################
# pre-processing functions

def create_lookup_tables(text):
    """
    Usage -- Create lookup tables for vocabulary
    Arguments -- 
    text: The text of tv scripts split into words
    Output --
    A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    int_to_vocab = dict(enumerate(tuple(set(text))))
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    
    return (vocab_to_int, int_to_vocab)
    
def token_lookup():
    """
    Usage -- Generate a dictionary to turn punctuations into a tokens, this is to prevent different words
             generations from the same punctuation. For example, 'bye!' and 'bye' can be tokenized as different tokens
    Output -- Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    punc_dict = {'.': "||Period||", ',' :"||Comma||", 
                     '"': "||QuotationMark||", ';': "||Semicolon||",
                     '!': "||Exclamationmark||", '?': "||Questionmark||",
                     '(': "||LeftParentheses||", ')': "||RightParentheses||",
                     '-': "||Dash||", '\n': "||Return||"}
        
    return punc_dict
 
 # check access to GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
    
 # batching input and wrap them up into generators using dataloaders
 def batch_data(words, sequence_length, batch_size):
    """
    Usage -- Batch the neural network data using DataLoader
    Arguments -- 
    words: The word ids of the text - text is cleaned and tokenized
    sequence_length: The sequence length of each batch
    batch_size: The size of each batch; the number of sequences in a batch
    Output --  DataLoader with batched data
    """
    # number of total batches we can make
    total_per_batch = sequence_length * batch_size
    n_batch = len(words)//total_per_batch
    words = words[:n_batch * total_per_batch]
    
    # reshape into batch_size rows
    words = np.array(words)
    feature_tensors = words.reshape((-1, sequence_length))
    target_tensors = feature_tensors[:,-1] + 1
   
    data = TensorDataset(torch.from_numpy(feature_tensors), torch.from_numpy(target_tensors))
    data_loader = DataLoader(data, batch_size = batch_size)
    return data_loader
    
 # Build the Neural Network - Network architecture, forward propagation method, and hidden state initiation
 class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Usage -- Initialize the PyTorch RNN Module
        Arguments -- 
        vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        output_size: The number of output dimensions of the neural network
        embedding_dim: The size of embeddings, should you choose to use them        
        hidden_dim: The size of the hidden layer outputs
        dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # define model layers
        # Embedding and LSTM layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout = dropout, batch_first = True)
    
        # dropout layer - a layer after LSTM
        self.dropout = nn.Dropout(0.3)
        
        # fc and sigmoid layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        
    
    
    def forward(self, nn_input, hidden):
        """
        Usage -- Forward propagation of the neural network
        Arguments -- 
        nn_input: The input to the neural network
        hidden: The hidden state   
        Output -- Two Tensors, the output of the neural network and the latest hidden state
        """
        
        # first, grab batch size from each dataloader iteration
        batch_size = nn_input.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(nn_input)
        lstm_out, hidden = self.lstm(embeds, hidden)
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
         # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)


        # reshape to be batch_size first
        out = out.view(batch_size, -1, self.output_size)
        # get last batch of prediction
        out = out[:, -1]
            
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        '''
        Usage -- Initialize the hidden state of an LSTM/GRU
        Arguments -- 
        batch_size: The batch_size of the hidden state
        Output --  hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
   
        # initialize hidden state with zero weights, and move to GPU if available
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
              weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
        
def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Usage -- Forward and backward propagation on the neural network
    Arguments -- 
    rnn: The PyTorch Module that holds the neural network
    optimizer: The PyTorch optimizer for the neural network
    criterion: The PyTorch loss function
    inp: A batch of input to the neural network
    target: The target output for the batch of input
    Output -- The loss and the latest hidden state Tensor
    """
   
    # move data to GPU, if available
    if (train_on_gpu):
        inp = inp.cuda()
        target = target.cuda()
    
    # zero accumulated gradients
    rnn.zero_grad()
    # get the output from the rnn
    output = rnn(inp, hidden)[0]
    hidden = rnn(inp, hidden)[1]
    
    # perform backpropagation and optimization
    loss = criterion(output, target)
    loss.backward(retain_graph=True)
    
    # prevent exploding gradients typical in RNNs and LSTMs
    #nn.utils.clip_grad_norm(rnn.parameters(), clip) - Future Improvement
    optimizer.step()
    loss = loss.item()
    # return the loss over a batch and the hidden state produced by our model
    
    return loss, hidden
    
    #TRAINING the Network
    
def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn
    
###### Set Hyperparameters
sequence_length = 15 # of words in a sequence
batch_size = 30
train_loader = batch_data(int_text, sequence_length, batch_size)
# Training parameters
num_epochs = 150
learning_rate = 0.001
# Vocab size
vocab_size = len(vocab_to_int)
output_size = vocab_size
embedding_dim = 280
hidden_dim = 350
# Number of RNN Layers -- 2 to 3 is better than more layers
n_layers = 2
# Show stats for every n number of batches
show_every_n_batches = 20

# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
helper.save_model('./save/trained_rnn', trained_rnn)
print('Model Trained and Saved')
        
    
