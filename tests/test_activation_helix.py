import pytest
import torch
from activation_helix import ActivationHELIX, create_activation_helix

class TestActivationHELIX:
    def test_init(self):
        m = ActivationHELIX(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = ActivationHELIX(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_activation_helix()
        assert isinstance(m, ActivationHELIX)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
