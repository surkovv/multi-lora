from multilora import MultiLoRALayerMaskingHom, MultiLoRALayerMaskingHomEfficient, MultiLoRALayerMasking
import torch
import pytest

def test_homs():
    n_adapters = 4
    batch_size = 8
    inp = 10
    out = 12
    adapter_ids = torch.randint(low=0, high=n_adapters, size=(batch_size,))
    hom = MultiLoRALayerMaskingHom(inp, out, adapter_ids=adapter_ids, n_adapters=n_adapters)
    hom.eval()
    hom_eff = MultiLoRALayerMaskingHomEfficient(inp, out, adapter_ids=adapter_ids, n_adapters=n_adapters)
    hom_eff.eval()

    hom.A.data = hom_eff.A.detach().clone()
    hom.B.data = torch.rand_like(hom.B)
    hom_eff.B.data = hom.B.detach().clone()
    x = torch.rand((batch_size, 3, inp))

    x1 = hom(x)
    x2 = hom_eff(x)
    
    x1.sum().backward()
    x2.sum().backward()

    assert torch.allclose(x1, x2)
    assert torch.allclose(hom.A.grad, hom_eff.A.grad)

def test_homeff_het():
    r = 8
    n_adapters = 4
    batch_size = 8
    inp = 10
    out = 12

    adapter_ids = torch.randint(low=0, high=n_adapters, size=(batch_size,))
    hom_eff = MultiLoRALayerMaskingHomEfficient(inp, out, adapter_ids=adapter_ids, n_adapters=n_adapters)
    hom_eff.eval()

    het = MultiLoRALayerMasking(inp, out, adapter_ids, [r] * n_adapters)
    for i in range(n_adapters):
        A = torch.rand_like(hom_eff.A[i])
        B = torch.rand_like(hom_eff.B[i])
        hom_eff.A.data[i] = A.detach().clone()
        hom_eff.B.data[i] = B.detach().clone()
        het.loras[i].A.data = A.detach().clone()
        het.loras[i].B.data = B.detach().clone()
    
    x = torch.rand((batch_size, 3, inp))

    x1 = hom_eff(x)
    x2 = het(x)
    x1.sum().backward()
    x2.sum().backward()
    
    assert torch.allclose(x1, x2)
    assert hom_eff.A.requires_grad
    assert torch.allclose(het.loras[2].A.grad, hom_eff.A.grad[2])
    assert torch.allclose(het.loras[2].B.grad, hom_eff.B.grad[2])
    assert not torch.allclose(hom_eff.B.grad.abs().sum(), torch.tensor(0.0))
