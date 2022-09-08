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
        
        self.word_embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        
        self.lstm = nn.LSTM(
            input_size=self.embed_size, 
            hidden_size=self.hidden_size, 
            batch_first=True, 
            dropout=0.5, 
            num_layers=self.num_layers)
        
        self.fc_linear = nn.Linear(
            in_features=self.hidden_size, 
            out_features=self.vocab_size)
        
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        
        # output shape : (batch_size, caption length , embed_size)
        captions_embed = self.word_embed(captions) 
        
        # Features shape : (batch_size, embed_size)
        # Word embeddings shape : (batch_size, caption length , embed_size)
        # output shape : (batch_size, caption length, embed_size)
        inputs = torch.cat((features.unsqueeze(1), captions_embed), dim=1)
        
        # output shape : (batch_size, caption length, hidden_size)
        outputs, _ = self.lstm(inputs)
        
        # output shape : (batch_size, caption length, vocab_size)
        outputs = self.fc_linear(outputs)
        
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        output_length = 0
        
        while output_length <= max_len:
            # LSTM layer
            # input shape:  (1, 1, embed_size)
            # output shaep: (1, 1, hidden_size)
            output, states = self.lstm(inputs, states)
            
            # Linear layer
            # input shape:  (1, hidden_size)
            # output shape: (1, vocab_size)
            output = self.fc_linear(output.squeeze(dim=1))
            _, predicted_idx = torch.max(output, 1)
            
            # Since Numpy doesn't support CUDA directly, 
            # we should convert CUDA tensor to cpu and then to numpy
            outputs.append(predicted_idx.cpu().numpy()[0].item())
            
            # If <end> word appear, then break the loop as it ends the sentence
            if predicted_idx == 1:
                break
                
            # For next iteration, embed the last predicted word 
            # to be the new input of the LSTM
            inputs = self.word_embed(predicted_idx)
            inputs = inputs.unsqueeze(1)
            
            output_length += 1
            
        return outputs