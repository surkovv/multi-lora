import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < size
    x = tl.load(x_ptr + block_start, mask=mask)
    y = tl.load(y_ptr + block_start, mask=mask)
    tl.store(output_ptr + block_start, x + y, mask=mask)

x = torch.ones(1024, device="cuda")
y = torch.ones(1024, device="cuda")
output = torch.empty_like(x)

add_kernel[(triton.cdiv(1024, 128),)](
    x, y, output, 1024, BLOCK_SIZE=128
)
print(output)