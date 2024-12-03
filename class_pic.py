from torchvision import models
from torchvision import transforms
from PIL import Image
import torch


resnet_weights = models.ResNet101_Weights.DEFAULT
categories = resnet_weights.meta['categories']
transforms = resnet_weights.transforms()

resnet = models.resnet101(weights=resnet_weights)

img = Image.open('image/lisa.jpg')
img_t = transforms(img)

batch_t = torch.unsqueeze(img_t, 0)
resnet.eval()
p = resnet(batch_t).squeeze()

res = p.softmax(dim=0).sort(descending=True)

for s, i in zip(res[0][:5], res[1][:5]):
    print(f'{categories[i]}: {s:.4f}')