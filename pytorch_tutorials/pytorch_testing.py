import torch
from torch import nn, optim
from torchvision.models import resnet18, ResNet18_Weights

print(f"torch.backends.mps.is_available() = {torch.backends.mps.is_available()}") #the MacOS is higher than 12.3+
print(f"torch.backends.mps.is_built() = {torch.backends.mps.is_built()}") #MPS is activated


a = torch.tensor([2., 3.], requires_grad=True)
print("a = ", a)
b = torch.tensor([6., 4.], requires_grad=True)
print("b = ", b)

Q = 3*a**3 - b**2

external_grad = torch.tensor([.1, .1])
Q.backward(gradient=external_grad)

# check if collected gradients are correct
print("a.grad = ", a.grad)
print(9*a**2 == a.grad)
print("b.grad = ", b.grad)
print(-2*b == b.grad)


model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False


# Let’s say we want to finetune the model on a new dataset with 10 labels.
# In resnet, the classifier is the last linear layer model.fc.
# We can simply replace it with a new linear layer (unfrozen by default) that acts as our classifier.
model.fc = nn.Linear(512, 10)

# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
