
# https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#1-loading-pre-trained-bert
# https://huggingface.co/jgammack/distilbert-base-mean-pooling

import torch
from transformers import AutoTokenizer, AutoModel

modelname = "bert-base-uncased"

zinnen = ["The man went to the store.","The man walked to the shop.", "The weather is very bad."]


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

print("Load tokenizer...")
tokenizer = AutoTokenizer.from_pretrained( modelname)

print("Load model...")
model = AutoModel.from_pretrained(modelname, output_hidden_states=True)
model.eval()



# Preprocess the sentence and return torch tensors  Op basis
embeddings = []
for zin in zinnen:
    encoded_input = tokenizer( zin, add_special_tokens=True, max_length=64, padding='max_length', return_tensors='pt')
    with torch.no_grad():
        outputs = model( **encoded_input)
        sentence_embeddings = mean_pooling(outputs, encoded_input['attention_mask'])
        embeddings.append(sentence_embeddings)

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

a = cos( embeddings[0], embeddings[1])
b = cos( embeddings[0], embeddings[2])

print( a)
print( b)
