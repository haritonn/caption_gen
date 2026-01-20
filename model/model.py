import torch as nn
import torchvision.models as models

# class Attention(nn.Module):
#     def __init__(self):
#         pass
    
#     def forward(self, x, hidden):
#         pass

class CaptionEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(CaptionEncoder, self).__init__()
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model = nn.Sequential(*list(self.model.children())[:-2])
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), x.size(1), -1)
        
        x = x.permute(0, 2, 1)
        return x


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(CaptionDecoder).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, h):
        x = self.embedding(x)
        x, h = self.lstm(x, h)
        out = self.fc(x)
        return out, h


class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim ,num_layers):
        super(CaptionGenerator, self).__init__()
        self.encoder = CaptionEncoder(embedding_dim)
        self.decoder = CaptionDecoder(vocab_size, embedding_dim, hidden_dim, num_layers)
    
    def forward(self, x, captions):
        features = self.encoder(x)
        captions = captions[:, :-1]
        embeddings = self.decoder.embedding(captions)
        outputs, _ = self.decoder.lstm(embeddings, None)
        outputs = self.decoder.fc(outputs)
        return outputs