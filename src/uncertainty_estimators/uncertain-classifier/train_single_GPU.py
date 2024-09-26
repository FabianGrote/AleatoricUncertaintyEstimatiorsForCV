import sys

import torch
from torch import nn
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ExponentialLR

from matplotlib import pyplot as plt

import aleatoric_uncertainty_estimator_model, trainer, loss_functions

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import os
print("Working dir:", os.getcwd())

def ddp_setup(rank: int, world_size: int):
  """
    Args:
    rank: Unique identifier of each process
    world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  torch.cuda.set_device(rank)
  # backend for distributed stuff
  init_process_group(backend="nccl", rank=rank, world_size=world_size)


train_dataset = datasets.Imagenette( # ImageNet(
    root = "/mnt/HDD1/datasets/Imagenet2012", # ImageNet2012",
    split = "train",
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
  )

val_dataset = datasets.Imagenette(
    root = "/mnt/HDD1/datasets/Imagenet2012",
    split = "val",
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
  )

train_loader = torch.utils.data.DataLoader(
  train_dataset, batch_size=512, shuffle=False, num_workers=20, sampler=DistributedSampler(train_dataset)
)

test_loader = torch.utils.data.DataLoader(
  val_dataset, batch_size=512, shuffle=False, num_workers=20, sampler=DistributedSampler(val_dataset)
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = DDP(aleatoric_uncertainty_estimator_model.Net(
  image_size = (3, 224, 224), # channels x width x height
  num_classes = 10, 
  encoder = "resnet50"
), device_ids=[gpu_id])

# TODO: für was ist das? 
# net.apply(neural_net.init_weights)

# UC uses 2 times num classes as linear output in loss, kyle uses logits_variance and softmax in loss
criterion_dict = {
  # Loss used by "Uncertainty classifier"-GitHub Repo
  "criterion_kendall_and_gal": loss_functions.Loss(device=device),

  # For logits_variance network output in Kyles version
  "criterion_kyles_variance": loss_functions.BayesianCategoricalCrossEntropy_KylesVersion(),
  # For softmax network output in Kyles version
  "criterion_kyles_softmax": nn.CrossEntropyLoss(),
}

predict = trainer.predict

kwargs = dict(lr=1e-4, weight_decay=0.0001)
optimizer = torch.optim.Adam(net.parameters(), **kwargs)

scheduler = ExponentialLR(optimizer, gamma=0.9999)

net.train()
trainingEpoch_loss = []
validationEpoch_loss = []
validationEpoch_Softmaxloss = []
criterion_to_use = "kendall and gal" # or "kyles version"
for epoch in range(500):
  train_losses = trainer.train(train_loader, net, criterion_to_use, criterion_dict, optimizer, scheduler, device)  
  print('Epoch: ' + str(epoch) + ' Train loss = %s' % (sum(train_losses) / len(train_losses)) )
  trainingEpoch_loss.append(train_losses)

  if criterion_to_use == "kendall and gal":
    score, validation_loss = trainer.test(test_loader, predict, net, criterion_to_use, criterion_dict, device)
    print('Epoch: ' + str(epoch) + ' Testing: Accuracy = %.2f%%, Loss %.4f' % (score*100, validation_loss))
    validationEpoch_loss.append(validation_loss)
  # TODO
  # elif criterion_to_use == "kyles version":
  #   score, validationEpoch_loss = train_and_test_routines.test(test_loader, predict, net, criterion_to_use, criterion_dict, device)
  #   print ('Testing: Accuracy = %.2f%%, Loss %.4f' % (score*100, validationEpoch_loss))
  #   validationEpoch_loss.append(validationEpoch_loss)
  
  if self.gpu_id == 0 and epoch%10 == 0:
    torch.save(net, "new_checkpoints/UncertaintyEstimator_" + criterion_to_use + "_Epoch_" + str(epoch) + "_test_loss_" + str(validationEpoch_loss[-1]) + "_test_accuracy_" + str(score) + ".pt")


plt.plot(trainingEpoch_loss, label='train_loss')
plt.plot(validationEpoch_loss, label='validation_loss')
plt.legend()
plt.show


torch.save(net, '%s.pt' % "aleatoric_final")