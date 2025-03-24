from multilora import MultiLoRALayerMaskingHom, MultiLoRALayerMaskingHomEfficient, MultiLoRALayerMasking, MultiLoRALayerSTK
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
    batch_size = 27
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
    assert het.loras[2].A.requires_grad
    assert torch.allclose(het.loras[2].A.grad, hom_eff.A.grad[2])
    assert torch.allclose(het.loras[2].B.grad, hom_eff.B.grad[2])
    assert not torch.allclose(hom_eff.B.grad.abs().sum(), torch.tensor(0.0))

def test_heteff_het():
    r = 16
    n_adapters = 4
    batch_size = 27
    inp = 32
    out = 32

    adapter_ids = torch.randint(low=0, high=n_adapters, size=(batch_size,)).to('cuda')
    # adapter_ids = torch.zeros(size=(batch_size,), dtype=int).to('cuda')
    het_eff = MultiLoRALayerSTK(inp, out, adapter_ids, [r] * n_adapters)
    het_eff.eval()

    het = MultiLoRALayerMasking(inp, out, adapter_ids, [r] * n_adapters)
    het.cuda()
    for i in range(n_adapters):
        A = torch.randn((out, r), dtype=torch.bfloat16)
        B = torch.randn((r, inp), dtype=torch.bfloat16)
        het_eff.A.data[i * r:i * r + r] = A.detach().clone().T
        het_eff.B.data[:,i * r:i * r + r] = B.detach().clone().T
        het.loras[i].A.data = het.loras[i].A.to(torch.bfloat16)
        het.loras[i].B.data = het.loras[i].B.to(torch.bfloat16)
        het.loras[i].A.data[:] = A.detach().clone()
        het.loras[i].B.data[:] = B.detach().clone()
    
    x = torch.rand((batch_size, 32, inp)).to('cuda', dtype=torch.bfloat16)

    x1 = het_eff(x)
    x2 = het(x)
    x1.sum().backward()
    x2.sum().backward()

    assert torch.allclose(x1, x2)
    assert het_eff.A.requires_grad
    assert torch.allclose(het.loras[2].A.grad, het_eff.A.grad[2*r:3*r, :].T)
    assert torch.allclose(het.loras[2].B.grad, het_eff.B.grad[:, 2*r:3*r].T)
    assert not torch.allclose(het_eff.B.grad.abs().sum(), torch.tensor(0.0, dtype=torch.bfloat16))
