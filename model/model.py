import torch
import torch.nn as nn
import torchvision.models as models

# class Attention(nn.Module):
#     def __init__(self):
#         pass
    
#     def forward(self, x, hidden):
#         pass

class CaptionEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(CaptionEncoder, self).__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        
        self.features = nn.Sequential(*list(model.children())[:-2])
    
    def forward(self, x):
        ft = self.features(x) #[batch_size, 2048, 7, 7]
        x = ft.view(ft.size(0), ft.size(1), -1) #[batch_size, 2048, 49]
        out = x.permute(0, 2, 1) #[batch_size, 49, 2048]
        return out


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(CaptionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, h_prev):
        embs = self.embedding(x) #[batch_size, seq_len, embedding_dim]
        probs, h_next = self.lstm(embs, h_prev) 
        #probs: [batch_size, seq_len, hidden_dim] h_next: [batch_size, num_layers, hidden_dim]
        out = self.fc(probs) #[batch_size, seq_len, vocab_size]
        return out, h_next


class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim ,num_layers):
        super(CaptionGenerator, self).__init__()
        self.encoder = CaptionEncoder(embedding_dim)
        self.decoder = CaptionDecoder(vocab_size, embedding_dim, hidden_dim, num_layers)
    
    def forward(self, x, captions):
        features = self.encoder(x)
        captions = captions[:, :-1] #without <end> token
        embeddings = self.decoder.embedding(captions)
        outputs, _ = self.decoder.lstm(embeddings, None)
        outputs = self.decoder.fc(outputs)
        return outputs