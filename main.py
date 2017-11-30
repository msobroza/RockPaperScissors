import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import os

# Hyper Parameters
sequence_length = 1
input_size = 4
hidden_size = 20
num_layers = 1
num_classes = 3
batch_size = 100
learning_rate = 0.01


def save_checkpoint(state, filename='rnn.pth.tar'):
    torch.save(state, filename)

# RNN Model (One-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hidden_state):
        # Forward propagate RNN
        x = x.view(1, 1, -1)
        out, hidden = self.rnn(x, hidden_state)

        # Decode hidden state of last time step
        out = self.fc(out.view(-1,self.hidden_size))
        return out, hidden 


rnn = RNN(input_size, hidden_size, num_layers, num_classes)


optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

file_name = 'rnn.pth.tar'

if os.path.isfile(file_name):
    print("=> loading checkpoint =>")
    checkpoint = torch.load(file_name)
    rnn.load_state_dict(checkpoint['rnn'])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    print("=> checkpoint not found =>")

MOVES = {0:'r', 1:'p', 2:'s'}
rev_dict = dict()
for i in MOVES:
    rev_dict[MOVES[i]] = i

def best_choice(mov):
    assert(mov in MOVES)
    if MOVES[mov]=='r':
        return rev_dict['p']
    elif MOVES[mov]=='p':
        return rev_dict['s']
    else:
        return rev_dict['r']

def evaluate_moves(machine_prediction, your_move, score_machine, score_user):
   if best_choice(your_move) == machine_prediction:
       print('Machine wins !')
       score_machine+=1
   elif your_move==machine_prediction:
       print('Equality !')
   else:
       print('You win !')
       score_user+=1
   print('Machine prediction is '+str(MOVES[machine_prediction]))
   print('Your move is '+str(MOVES[your_move]))
   return score_machine, score_user

PREVIOUS_MOUVEMENT = np.array([4])
# Train the Model
valid=True
hidden_state= Variable(torch.zeros(1, 1, hidden_size))
total_loss= 0.0
count = 0
criterion = nn.CrossEntropyLoss()
score_user = 0
score_machine = 0
while valid :
    # Input initialization

    input = Variable(torch.from_numpy((np.arange(input_size) == PREVIOUS_MOUVEMENT).astype(np.float32)))
    
    # Predicting winner mouvement
    optimizer.zero_grad()
    outputs, hidden_state = rnn(input, hidden_state)
    _, predicted = torch.max(outputs.data, 1)
    machine_prediction = int(predicted.numpy())

    # Getting answer from user
    mode = str(raw_input('Choose r (rock) | p (paper) | s (scissor) : '))
    if mode not in rev_dict:
        valid = False
        break

    # Optimizing network

    next_mouvement = rev_dict[mode]
    labels = Variable(torch.from_numpy(np.array([best_choice(next_mouvement)])))
    loss = criterion(outputs, labels)
    loss.backward(retain_graph=True)
    optimizer.step()
    
    # Evaluating mouvement
    score_machine, score_user = evaluate_moves(machine_prediction, rev_dict[mode], score_machine, score_user)
    
    # Previous initialization for next step
    PREVIOUS_MOUVEMENT = np.array([next_mouvement])
    count +=1
    if count%30 == 0:
       print('Machine win rate: %f'%(float(score_machine)/score_user))
    

save_checkpoint({'rnn':rnn.state_dict(), 'optimizer' : optimizer.state_dict()})
