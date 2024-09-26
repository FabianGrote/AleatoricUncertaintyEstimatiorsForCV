import aleatoric, neural_net

import torch
from torchvision import transforms, datasets
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import sys

def show_images(images, w, h):
  for i, image in enumerate(images):
    plt.subplot(w, h, i+1)
    image = image.swapaxes(0,1)
    image = image.swapaxes(1,2)
    plt.imshow(image)
    plt.axis('off')
 
def slice_by_class(images, digits, uncertainty, n=10):
  high = []
  low = []
  for i in range(1, 10+1):
    images_i = images[digits == i]
    uncertainty_i = uncertainty[digits == i]
    indices = uncertainty_i.argsort()
    high.extend(images_i[indices[::-1][:n]])
    low.extend(images_i[indices[:n]])
  return high, low

if __name__ == '__main__':
  variant = sys.argv[1]

  test_loader = torch.utils.data.DataLoader(
    datasets.Imagenette( # ImageNet(
      root = "/datasets/Imagenette-download", # ImageNet2012",
      split = "val",
      transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
    ),
    batch_size=256, shuffle=True
  )
  # test_loader = torch.utils.data.DataLoader(
  # datasets.MNIST('data', train=False, 
  #   transform=transforms.Compose([
  #   transforms.ToTensor(),
  #   transforms.Normalize((0.1307,), (0.3081,))
  # ])),
  # batch_size=256, shuffle=False)



  if variant == 'aleatoric':
    net = torch.load('aleatoric.pt')
    aleatoric = []
    images, digits = [], []

    with torch.no_grad():
      for data, target in tqdm(test_loader):
        x = data
        mu, log_sigma2 = net(x)
        aleatoric.extend([ np.linalg.norm(s) for s in np.exp(0.5*log_sigma2.detach().numpy()) ])
        images.extend(data.detach().numpy())
        digits.extend(target.detach().numpy())

      images, digits = np.array(images), np.array(digits)
      aleatoric = np.array(aleatoric)

    high, low = slice_by_class(images, digits, aleatoric)
    
    plt.figure()
    show_images(high, 10, 10)
    plt.savefig('aleatoric_high.png')

    plt.figure()
    show_images(low, 10, 10)
    plt.savefig('aleatoric_low.png')

  else:
    print ('variant must be one of combined,aleatoric,epistemic')