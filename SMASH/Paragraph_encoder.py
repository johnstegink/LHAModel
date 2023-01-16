# Class to encode the paragraph layer
import torch
import torch.nn as nn

class ParagraphEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, hidden_layers=2, output_dim=1, keep_prob=0.2):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.GRU_Sent = nn.GRU(input_dim, int(hidden_dim/2), hidden_layers, batch_first=True, dropout=keep_prob, bidirectional=True)
        self.GRU_para = nn.GRU(input_dim, int(hidden_dim/2), hidden_layers, batch_first=True, dropout=keep_prob, bidirectional=True)
        self.FC1=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,output_dim),nn.Softmax(dim=-2))
        self.FC2=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,output_dim),nn.Softmax(dim=-2))
        self.FC3=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,output_dim),nn.Softmax(dim=-2))
    def forward(self, X): #(4,10,50,1024)
        out=X.float() #(4,10,50,1024)
        alpha_word=self.FC1(out)
        first_level_attention=torch.sum(alpha_word*out,dim=-2)
        first_level_attention,_=self.GRU_Sent(first_level_attention)
        alpha_sent=self.FC2(first_level_attention)
        second_level_attention=torch.sum(alpha_sent*first_level_attention,dim=-2).unsqueeze(0)
        second_level_attention,_=self.GRU_para(second_level_attention)
        alpha_para=self.FC3(second_level_attention)
        third_level_output=torch.sum(alpha_para*second_level_attention,dim=-2)
        return third_level_output