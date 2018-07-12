import torch
from torch import nn

import torch.nn.functional as F


class FeedForwardClassifier(nn.Module):

    def __init__(self, hidden_units, dropout=0.5):
        """ Form the model

        Keyword arguments:
        hidden_layers -- The size of each unit, array
        dropout -- Dropout rate
        """

        # Call constructor of parent
        super(FeedForwardClassifier, self).__init__()


        input_size = 25088
        output_size = 102

        # Start defining the architecture of the model
        # Create hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size,
                                                      hidden_units[0])])

        # Create arbitary layers
        layers = zip(hidden_units[0:-1], hidden_units[1:])
        self.hidden_layers.extend([ nn.Linear(h1, h2) for h1, h2 in layers])

        # set the output
        self.output = nn.Linear(hidden_units[-1], output_size)

        # Set the dropout
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        """
        Define forward function of the model

        Keyword arguments
        x - input features
        """
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)
        return F.log_softmax(x, dim=1)
