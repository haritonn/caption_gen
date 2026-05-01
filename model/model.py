import torch
import torch.nn as nn
import torchvision.models as models


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


class CaptionEncoder(nn.Module):
    def __init__(self, encoded_image_size=7, pretrained=False):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        for parameter in self.resnet.parameters():
            parameter.requires_grad = False

    def forward(self, images):
        encoded = self.resnet(images)
        encoded = self.adaptive_pool(encoded)
        return encoded.permute(0, 2, 3, 1)


class CaptionDecoder(nn.Module):
    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=2048,
        dropout=0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.layer_norm = nn.LayerNorm(embed_dim + encoder_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.fc.bias, 0)
        nn.init.xavier_uniform_(self.f_beta.weight)
        nn.init.constant_(self.f_beta.bias, 0)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        return self.init_h(mean_encoder_out), self.init_c(mean_encoder_out)

    def forward(
        self, encoder_out, encoded_captions, caption_lengths, sampling_prob=0.0
    ):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(
            encoder_out.device
        )
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(
            encoder_out.device
        )

        for step in range(max(decode_lengths)):
            active_batch_size = sum(length > step for length in decode_lengths)

            if step == 0 or torch.rand(1).item() > sampling_prob or not self.training:
                word_embeddings = embeddings[:active_batch_size, step, :]
            else:
                previous_word_ids = torch.argmax(
                    predictions[:active_batch_size, step - 1, :],
                    dim=1,
                )
                word_embeddings = self.embedding(previous_word_ids)

            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:active_batch_size],
                h[:active_batch_size],
            )
            gate = self.sigmoid(self.f_beta(h[:active_batch_size]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            lstm_input = torch.cat(
                [word_embeddings, attention_weighted_encoding], dim=1
            )
            lstm_input = self.layer_norm(lstm_input)
            h, c = self.decode_step(
                lstm_input,
                (h[:active_batch_size], c[:active_batch_size]),
            )
            predictions[:active_batch_size, step, :] = self.fc(self.dropout_layer(h))
            alphas[:active_batch_size, step, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


class CaptionGenerator(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        attention_dim=512,
        dropout=0.3,
        encoder_pretrained=False,
    ):
        super().__init__()
        self.encoder = CaptionEncoder(pretrained=encoder_pretrained)
        self.decoder = CaptionDecoder(
            attention_dim=attention_dim,
            embed_dim=embedding_dim,
            decoder_dim=hidden_dim,
            vocab_size=vocab_size,
            dropout=dropout,
        )

    def forward(self, images, captions, caption_lengths, sampling_prob=0.0):
        encoder_out = self.encoder(images)
        return self.decoder(encoder_out, captions, caption_lengths, sampling_prob)
