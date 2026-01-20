import pytest
import torch
from model.model import CaptionEncoder

class TestCaptionEncoder:
    @pytest.fixture
    def encoder(self):
        return CaptionEncoder(512)
    
    def test_encoder_forward(self, encoder):
        x = torch.randn(32, 3, 224, 224)
        output = encoder(x)
        
        assert output.shape == (32, 49, 2048)
        assert output.dtype == torch.float32
    
    # spatial structure validation (different inputs = different outputs)
    def test_encoder_spatial_structure(self, encoder):
        x1 = torch.randn(1, 3, 224, 224)
        x2 = torch.randn(1, 3, 224, 224)
        features1 = encoder(x1)
        features2 = encoder(x2)
        
        # also validate that features (pixels) are different
        pixels_norm1 = features1.norm(dim=1)
        pixels_norm2 = features2.norm(dim=1)
        
        assert not torch.allclose(features1, features2, atol=1e-6)
        assert pixels_norm1.std() >= 0.01
        assert pixels_norm2.std() >= 0.01
        
    def test_encoder_freezing(self, encoder):
        for param in encoder.parameters():
            assert not param.requires_grad
        
    def test_encoder_backward(self, encoder):
        pass