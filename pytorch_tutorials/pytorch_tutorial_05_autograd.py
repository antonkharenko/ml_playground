import torch

x = torch.ones(5)  # input tensor
print(f'x: {x} {x.shape}')
y = torch.zeros(3)  # expected output
print(f'y: {y} {y.shape}')
w = torch.randn(5, 3, requires_grad=True)
print(f'w: {w} {w.shape}')
b = torch.randn(3, requires_grad=True)
print(f'b: {b} {b.shape}')
z = torch.matmul(x, w)+b
print(f'z = x * w + b: {z} {z.shape}')
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(f'loss = binary_cross_entropy_with_logits(z, y): {loss} {loss.shape}')

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(w.grad)
print(b.grad)

z = torch.matmul(x, w)+b
print(z.requires_grad)

# with torch.no_grad():
#     z = torch.matmul(x, w)+b
# print(z.requires_grad)

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")