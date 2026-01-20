import pytest
from model.model import CaptionDecoder
import torch

class TestCaptionDecoder:
    @pytest.fixture
    def decoder(self):
        return CaptionDecoder(
            vocab_size=8,
            embedding_dim=128,
            hidden_dim=130,
            num_layers=2,
        )
    
    @pytest.fixture
    def text(self):
        return torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        
    SEQ_LEN = 6
    
    def test_decoder_embedding_shape(self, decoder, text):
        output = decoder.embedding(text)
        assert output.shape == (1, self.SEQ_LEN, 128)
        
    def test_decoder_lstm_shape(self, decoder, text):
        embs = decoder.embedding(text)
        h0 = torch.zeros(2, 1, 130)
        c0 = torch.zeros(2, 1, 130)
        
        output, _ = decoder.lstm(embs, (h0, c0))
        assert output.shape == (1, self.SEQ_LEN, 130)
        
    def test_decoder_hidden_shape(self, decoder, text):
        embs = decoder.embedding(text)
        h0 = torch.zeros(2, 1, 130)
        c0 = torch.zeros(2, 1, 130)
        
        output, (h_next, _) = decoder.lstm(embs, (h0, c0))
        assert h_next.shape == (2, 1, 130)
        
        
    def test_decoder_forward(self, decoder, text):
        h0 = torch.zeros(2, 1, 130)
        c0 = torch.zeros(2, 1, 130)
        
        output, _ = decoder(text, (h0, c0))
        assert output.shape == (1, self.SEQ_LEN, 8)
        assert output.dtype == torch.float32
    
    def test_decoder_unfreezed(self, decoder):
        for param in decoder.parameters():
            assert param.requires_grad