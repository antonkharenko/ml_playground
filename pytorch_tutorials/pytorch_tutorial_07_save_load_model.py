import torch
import torchvision.models as models

# Save/load just weights
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model2_weights.pth')

model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model2_weights.pth'))
model.eval()

# Save/load with shapes
torch.save(model, 'model2.pth')
model = torch.load('model2.pth')


