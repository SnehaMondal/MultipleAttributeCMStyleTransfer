import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierModel(nn.Module):
    def __init__(self, style_vector, num_attr, hidden_dim):
        super(ClassifierModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.style_vector = style_vector
        input_size = hidden_dim*(num_attr+1)
        self.ff = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1)
        )
        ## TODO: Add weight initialisations
    
    def forward(self, hidden_states, input_style_scores):
        batch_size = input_style_scores.shape[0]
        assert(hidden_states.shape==batch_size*self.hidden_dim)
        style_scaled = self.style_vector.repeat(batch_size, 1, 1) * input_style_scores[:,:, None]
        style_scaled = style_scaled.view(batch_size, -1)
        hidden_states = torch.cat((hidden_states, style_scaled), -1)
        hidden_states = self.ff(hidden_states)
        return hidden_states
