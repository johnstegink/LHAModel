# Create BERT embeddings
# https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#1-loading-pre-trained-bert
# https://huggingface.co/jgammack/distilbert-base-mean-pooling

import torch
from transformers import AutoTokenizer, AutoModel

class BertEmbedder:
    def __init__(self, modelname, device):
        self.modelname = modelname
        self.device = device

        # Load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained( self.modelname)
        self.model = AutoModel.from_pretrained(modelname, output_hidden_states=True)
        self.model.eval()
        self.model.to( self.device)

    def __mean_pooling(self, model_output, attention_mask):
        """
        Take the attention mask into account for averagin out the token embeddings
        :param model_output:    Output of the model
        :param attention_mask:  The attention mask
        :return: Average
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def embed_sentences(self, sentences, number_of_tokens):
        """
        Embed one sentence, it is not optimal, but suffices for now
        :param sentences: the sentences to get the embeddings from, a string with words seperated by " " ending with a .
        :param max_length: the number of tokens to be retrieved, using padding
        :return: a Pytorch tensor with the dimensions(<nrofsentences>, 768)
        """
        encoded_input = self.tokenizer(sentences, add_special_tokens=True, max_length=number_of_tokens, padding='max_length',
                                  return_tensors='pt', truncation=True).to( self.device)
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            embeddings = self.__mean_pooling(outputs, encoded_input['attention_mask'])
            return embeddings
