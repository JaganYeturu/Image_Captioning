
#need to add more comments
#need to add more comments
#need to add more comments
#need to add more comments
#need to add more comments
import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linearLayer = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        batch_size = features.shape[0]
        captions_trimmed = captions[..., :-1]
        embed = self.embedding(captions_trimmed)
        inputs = torch.cat([features.unsqueeze(1), embed], 1)
        lstm_out, self.hidden = self.lstm(inputs)
        outputs = self.linearLayer(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        for i in range(max_len):
            lstm_output, states = self.lstm(inputs, states)
            outputs = self.linearLayer(lstm_output)
            outputs = outputs.squeeze(1)
            outputid  = outputs.argmax(dim=1)
            tokens.append(outputid.item())
            inputs = self.embedding(outputid.unsqueeze(0))
        return tokens
#need to add more comments
#need to add more comments
#need to add more comments
#need to add more comments
#need to add more comments
#need to add more comments
#need to add more comments
#need to add more comments
#need to add more comments
#need to add more comments

#need to add more comments
#need to add more comments