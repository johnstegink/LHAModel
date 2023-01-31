# Bert implementation, strongly inspired by: https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d


# Importing the relevant modules
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import torch
# Loading the pre-trained BERT model
###################################
# Embeddings will be derived from
# the outputs of this model
model = BertModel.from_pretrained(‘bert-base-uncased’,
           output_hidden_states = True,)
# Setting up the tokenizer
###################################
# This is the same tokenizer that
# was used in the model to generate
# embeddings to ensure consistency
tokenizer = BertTokenizer.from_pretrained(‘bert-base-uncased’)
