import torch
import torch.nn as nn
import torchvision.models as models


class CaptionEncoder(nn.Module):
    """
    Encoder based on extracting features from images (via pre-trained resnet50)
    """

    def __init__(self, embedding_dim):
        super(CaptionEncoder, self).__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        ft = self.features(x)  # [batch_size, 2048, 7, 7]
        x = ft.view(ft.size(0), ft.size(1), -1)  # [batch_size, 2048, 49]
        out = x.permute(0, 2, 1)  # [batch_size, 49, 2048]
        return out


class AttentionMechanism(nn.Module):
    """
    Getting context vectors from encoder outputs & current hidden state of decoder
    """

    def __init__(self, encoder_dim, decoder_dim, attn_dim):
        super().__init__()
        self.encoder_map = nn.Linear(encoder_dim, attn_dim)
        self.decoder_map = nn.Linear(decoder_dim, attn_dim)

        self.attn_map = nn.Linear(attn_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        encoder_map = self.encoder_map(
            encoder_out
        )  # [batch_size, num_pixels, attention_dim]
        decoder_map = self.decoder_map(decoder_hidden)  # [batch_size, attention_dim]

        decoder_map = decoder_map.unsqueeze(1)
        attn_map = self.attn_map(
            torch.tanh(encoder_map + decoder_map)
        )  # [batch_size, num_pixels, 1]
        attn_map = attn_map.squeeze(2)

        alphas = nn.functional.softmax(attn_map, dim=1)
        context_vector = (encoder_out * alphas.unsqueeze(2)).sum(dim=1)

        return context_vector, alphas


class CaptionDecoder(nn.Module):
    """
    Generating output from context vector, concatenated with current word embedding.
    """

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, encoder_dim, attn_dim, num_layers
    ):
        super(CaptionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim + encoder_dim, hidden_dim, num_layers, batch_first=True
        )
        self.attention = AttentionMechanism(encoder_dim, hidden_dim, attn_dim)

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.drop = nn.Dropout(p=0.5)

        self.num_layers = num_layers

    def forward(self, encoder_out, captions, hidden=None):
        batch_size = encoder_out.size(0)
        seq_len = captions.size(1)

        embs = self.embedding(captions)  # [batch_size, seq_len, embedding_dim]
        if hidden is None:
            h = torch.zeros(self.num_layers, batch_size, self.lstm.hidden_size).to(
                encoder_out.device
            )
            c = torch.zeros(self.num_layers, batch_size, self.lstm.hidden_size).to(
                encoder_out.device
            )
            hidden = (h, c)

        outputs, all_attn_weights = [], []

        for t in range(seq_len):
            word_emb = embs[:, t, :]
            h_prev = hidden[0][-1]

            context, attn_weights = self.attention(encoder_out, h_prev)
            lstm_input = torch.cat([context, word_emb], dim=1)
            lstm_input = lstm_input.unsqueeze(1)

            lstm_output, hidden = self.lstm(lstm_input, hidden)

            output = self.fc(self.drop(lstm_output))
            outputs.append(output)
            all_attn_weights.append(attn_weights)

        outputs = torch.cat(outputs, dim=1)
        all_attn_weights = torch.stack(all_attn_weights, dim=1)

        return outputs, hidden, all_attn_weights


class CaptionGenerator(nn.Module):
    """
    Full encoder-decoder architecture with attention mechanism
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        attn_dim,
        encoder_dim=2048,
    ):
        super(CaptionGenerator, self).__init__()
        self.encoder = CaptionEncoder(embedding_dim)
        self.decoder = CaptionDecoder(
            vocab_size, embedding_dim, hidden_dim, encoder_dim, attn_dim, num_layers
        )

    def forward(self, images, captions):
        encoder_out = self.encoder(images)
        captions = captions[:, :-1]  # removing <END> token

        outputs, _, attn_weights = self.decoder(encoder_out, captions)

        return outputs, attn_weights

    def beam_search(self, images, vocab, beam_size=3, max_length=50):
        batch_size = images.size(0)
        encoder_out = self.encoder(images)

        start_idx = vocab.word2idx["<START>"]
        end_idx = vocab.word2idx["<END>"]

        results = []

        for i in range(batch_size):
            img_encoder_out = encoder_out[i : i + 1]

            sequences = torch.full((1, 1), start_idx, device=images.device)
            scores = torch.zeros(1, device=images.device)

            h = torch.zeros(
                self.decoder.num_layers, 1, self.decoder.lstm.hidden_size
            ).to(images.device)
            c = torch.zeros(
                self.decoder.num_layers, 1, self.decoder.lstm.hidden_size
            ).to(images.device)
            hidden = (h, c)

            for step in range(max_length):
                current_beam_size = sequences.size(0)

                if current_beam_size < beam_size:
                    img_encoder_out_expanded = img_encoder_out.expand(
                        current_beam_size, -1, -1
                    )
                else:
                    img_encoder_out_expanded = img_encoder_out.expand(beam_size, -1, -1)

                word_emb = self.decoder.embedding(sequences[:, -1])
                h_prev = hidden[0][-1]

                context, _ = self.decoder.attention(img_encoder_out_expanded, h_prev)
                lstm_input = torch.cat([context, word_emb], dim=1).unsqueeze(1)

                lstm_output, hidden = self.decoder.lstm(lstm_input, hidden)
                output = self.decoder.fc(self.decoder.drop(lstm_output))
                logits = output.squeeze(1)

                log_probs = torch.log_softmax(logits, dim=1)
                candidates_scores = scores.unsqueeze(1) + log_probs
                candidates_scores = candidates_scores.view(-1)

                top_scores, top_indices = torch.topk(
                    candidates_scores, min(beam_size, candidates_scores.size(0))
                )

                beam_indices = top_indices // logits.size(1)
                word_indices = top_indices % logits.size(1)

                new_sequences = []
                new_h = []
                new_c = []

                for j in range(len(top_scores)):
                    beam_idx = beam_indices[j]
                    word_idx = word_indices[j]

                    new_seq = torch.cat([sequences[beam_idx], word_idx.unsqueeze(0)])
                    new_sequences.append(new_seq)

                    new_h.append(hidden[0][:, beam_idx : beam_idx + 1])
                    new_c.append(hidden[1][:, beam_idx : beam_idx + 1])

                sequences = torch.stack(new_sequences)
                scores = top_scores
                hidden = (torch.cat(new_h, dim=1), torch.cat(new_c, dim=1))

                if word_indices[0].item() == end_idx:
                    break

            best_sequence = sequences[0].cpu().tolist()
            results.append(best_sequence)

        return results
