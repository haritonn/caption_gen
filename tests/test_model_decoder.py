import torch

from model.model import CaptionDecoder


class TestCaptionDecoder:
    def test_decoder_forward_keeps_sorted_alignment(self):
        decoder = CaptionDecoder(
            attention_dim=32,
            embed_dim=16,
            decoder_dim=20,
            vocab_size=11,
            encoder_dim=8,
            dropout=0.0,
        )
        decoder.eval()

        batch_size = 3
        encoder_out = torch.randn(batch_size, 2, 2, 8)
        encoded_captions = torch.tensor(
            [
                [1, 4, 5, 2, 0],
                [1, 6, 2, 0, 0],
                [1, 7, 8, 9, 2],
            ],
            dtype=torch.long,
        )
        caption_lengths = torch.tensor([4, 3, 5], dtype=torch.long)

        predictions, sorted_captions, decode_lengths, alphas, sort_ind = decoder(
            encoder_out, encoded_captions, caption_lengths
        )

        expected_sort_ind = torch.tensor([2, 0, 1], dtype=torch.long)
        assert torch.equal(sort_ind, expected_sort_ind)
        assert decode_lengths == [4, 3, 2]
        assert torch.equal(sorted_captions, encoded_captions[expected_sort_ind])
        assert predictions.shape == (batch_size, max(decode_lengths), 11)
        assert alphas.shape == (batch_size, max(decode_lengths), 4)

    def test_decoder_outputs_can_be_restored_with_sort_indices(self):
        decoder = CaptionDecoder(
            attention_dim=16,
            embed_dim=8,
            decoder_dim=10,
            vocab_size=7,
            encoder_dim=6,
            dropout=0.0,
        )
        decoder.eval()

        encoder_out = torch.randn(3, 2, 2, 6)
        encoded_captions = torch.tensor(
            [
                [1, 4, 2, 0],
                [1, 5, 6, 2],
                [1, 3, 4, 5],
            ],
            dtype=torch.long,
        )
        caption_lengths = torch.tensor([3, 4, 4], dtype=torch.long)

        predictions, sorted_captions, _, _, sort_ind = decoder(
            encoder_out, encoded_captions, caption_lengths
        )

        restored_captions = sorted_captions.new_zeros(sorted_captions.shape)
        restored_captions[sort_ind] = sorted_captions

        restored_predictions = predictions.new_zeros(predictions.shape)
        restored_predictions[sort_ind] = predictions

        assert torch.equal(restored_captions, encoded_captions)
        assert restored_predictions.shape == predictions.shape
