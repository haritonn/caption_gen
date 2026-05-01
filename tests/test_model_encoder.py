import torch

from model.model import CaptionEncoder


class TestCaptionEncoder:
    def test_encoder_forward_shape(self):
        encoder = CaptionEncoder()
        x = torch.randn(2, 3, 224, 224)

        output = encoder(x)

        assert output.shape == (2, 7, 7, 2048)
        assert output.dtype == torch.float32

    def test_encoder_parameters_are_frozen(self):
        encoder = CaptionEncoder()

        for param in encoder.resnet.parameters():
            assert not param.requires_grad
