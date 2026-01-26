import torch
import torch.nn as nn
import torchvision.models as models


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # [batch_size, num_pixels, attention_dim]
        att2 = self.decoder_att(decoder_hidden)  # [batch_size, attention_dim]
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(
            2
        )  # [batch_size, num_pixels]
        alpha = self.softmax(att)  # [batch_size, num_pixels]
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(
            dim=1
        )  # [batch_size, encoder_dim]
        return attention_weighted_encoding, alpha


class CaptionEncoder(nn.Module):
    def __init__(self, encoded_image_size=7):
        super(CaptionEncoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        out = self.resnet(images)  # [batch_size, 2048, image_size/32, image_size/32]
        out = self.adaptive_pool(
            out
        )  # [batch_size, 2048, encoded_image_size, encoded_image_size]
        out = out.permute(
            0, 2, 3, 1
        )  # [batch_size, encoded_image_size, encoded_image_size, 2048]
        return out


class CaptionDecoder(nn.Module):
    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=2048,
        dropout=0.5,
    ):
        super(CaptionDecoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # [batch_size, decoder_dim]
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(
            batch_size, -1, encoder_dim
        )  # [batch_size, num_pixels, encoder_dim]
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(
            encoded_captions
        )  # [batch_size, max_caption_length, embed_dim]

        h, c = self.init_hidden_state(encoder_out)  # [batch_size, decoder_dim]
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(
            encoder_out.device
        )
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(
            encoder_out.device
        )
        for t in range(max(decode_lengths)):
            batch_size_t = sum([length > t for length in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )
            gate = self.sigmoid(
                self.f_beta(h[:batch_size_t])
            )  # gating scalar, [batch_size_t, encoder_dim]
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat(
                    [embeddings[:batch_size_t, t, :], attention_weighted_encoding],
                    dim=1,
                ),
                (h[:batch_size_t], c[:batch_size_t]),
            )  # [batch_size_t, decoder_dim]
            preds = self.fc(self.dropout_layer(h))  # [batch_size_t, vocab_size]
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


class CaptionGenerator(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        attention_dim=512,
        dropout=0.5,
    ):
        super(CaptionGenerator, self).__init__()
        self.encoder = CaptionEncoder()
        self.decoder = CaptionDecoder(
            attention_dim=attention_dim,
            embed_dim=embedding_dim,
            decoder_dim=hidden_dim,
            vocab_size=vocab_size,
            dropout=dropout,
        )

    def forward(self, images, captions, caption_lengths):
        encoder_out = self.encoder(images)
        predictions, encoded_captions, decode_lengths, alphas, sort_ind = self.decoder(
            encoder_out, captions, caption_lengths
        )
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
