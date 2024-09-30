import sys

import torch
from torch import nn
from torchvision import transforms, datasets
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import argparse
import time


import aleatoric_uncertainty_estimator_model, trainer, loss_functions

import os
print("Working dir:", os.getcwd())

# local conda env: conda activate fabis_uncertainty_env
# local execution: python3 train.py --accelerator='gpu' --devices=1 --num_nodes=1 --max_epochs=1000 --dataset="Imagenette"
parser = argparse.ArgumentParser()
parser.add_argument("--accelerator", default="cpu", help="cpu or gpu", type=str)
parser.add_argument("--devices", default=1, help="Number of GPU nodes for distributed training.", type=int)
parser.add_argument("--num_nodes", default=1, help="Number of GPU nodes for distributed training.", type=int)
parser.add_argument("--max_epochs", default=100, help="Stop training once this number of epochs is reached.", type=int)
parser.add_argument("--dataset", default="Imagenette", help="Dataset to use. Can be: Imagenette, MNIST, ...", type=str)
parser.add_argument("--freeze_encoder_params", default=True, help="If encoder parameters should be freezed or not.", type=bool)
args = parser.parse_args()

dataset = args.dataset

if dataset == "Imagenette":
  train_dataset = datasets.Imagenette(
      # root = "~/datasets/ImageNet2012",   # BwUniCloud nette-download", #Net2012", # ImageNet2012",
      root = "/mnt/HDD1/datasets/Imagenette",   # local workstation
      split = "train",
      transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
    )
  val_dataset = datasets.Imagenette(
      # root = "~/datasets/ImageNet2012",   # BwUniCloud nette-download", # Net2012", mnt/HDD1
      root = "/mnt/HDD1/datasets/Imagenette",   # local workstation
      split = "val",
      transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
    )
  class_labels = {'tench': 0, 'English springer': 1, 'cassette player': 2, 'chainsaw': 3, 'church': 4, 'French horn': 5, 'garbage truck': 6, 'gas pump': 7, 'golf ball': 8, 'parachute': 9}
else:
  train_dataset = datasets.Imagenette( # ImageNet or Imagenette
    # root = "~/datasets/ImageNet2012",   # BwUniCloud nette-download", #Net2012", # ImageNet2012",
    root = "/mnt/HDD1/datasets/Imagenette",   # local workstation   or: ImageNet2012
    split = "train", #"val", or "train",
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
  )
  val_dataset = datasets.Imagenette( # ImageNet or Imagenette
    # root = "~/datasets/ImageNet2012",   # BwUniCloud nette-download", # Net2012", mnt/HDD1
    root = "/mnt/HDD1/datasets/Imagenette",   # local workstation   or: ImageNet2012
    split = "val",
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
  )
  class_labels = train_dataset.class_to_idx


train_loader = torch.utils.data.DataLoader(
  train_dataset, batch_size=40, shuffle=True, num_workers=8 # sampler=DistributedSampler(train_dataset)
)

val_loader = torch.utils.data.DataLoader(
  val_dataset, batch_size=40, shuffle=False, num_workers=8 #, sampler=DistributedSampler(val_dataset)
)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = len(train_dataset.classes)
freeze_encoder_params = args.freeze_encoder_params
net = aleatoric_uncertainty_estimator_model.Net(
  image_size = (3, 224, 224), # channels x width x height
  num_classes = num_classes, # imagenette: 10, 
  encoder = "resnet50",
  freeze_encoder_params = freeze_encoder_params
)

lr_monitor = LearningRateMonitor(logging_interval='step')

# UC uses 2 times num classes as linear output in loss, kyle uses logits_variance and softmax in loss
criterion_dict = {
  # Loss used by "Uncertainty classifier"-GitHub Repo
  "criterion_kendall_and_gal": loss_functions.Loss(num_classes=num_classes, T=1000),

  # For logits_variance network output in Kyles version
  "criterion_kyles_variance": loss_functions.BayesianCategoricalCrossEntropy_KylesVersion(),
  # For softmax network output in Kyles version
  "criterion_kyles_softmax": nn.CrossEntropyLoss(),
}
criterion_to_use = "kyles_version" # "kyles_version" # "kendall_and_gal" or "kyles_version" or "softmax_only"

time = time.strftime("%Y%m%d_%H-%M")
# default logger used by trainer (if tensorboard is installed)
logger = TensorBoardLogger(
  save_dir=os.getcwd(),
  name="lightning_logs",
  version=dataset + "_" + criterion_to_use + "_freeze_encoder-" + freeze_encoder_params + "_" + time
)


predict = aleatoric_uncertainty_estimator_model.predict

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)#
aleatoricUncertaintyEstimator = trainer.AleatoricUncertaintyEstimator(
  net = net,
  criterion_to_use = criterion_to_use,
  criterion_dict = criterion_dict, 
  predict = predict,
  num_classes = num_classes,
  class_labels = class_labels,
  freezer_encoder_params = freeze_encoder_params
)
trainer = L.Trainer(
  check_val_every_n_epoch=5,
  max_epochs=args.max_epochs,
  devices=args.devices,
  num_nodes=args.num_nodes,
  accelerator=args.accelerator,
  enable_checkpointing=True,
  # log_every_n_steps = 100,
  limit_train_batches=1.0,
  limit_val_batches=1.0,
  logger=logger,
  callbacks=[lr_monitor],
  )
trainer.fit(  
  model=aleatoricUncertaintyEstimator, 
  train_dataloaders=train_loader,
  val_dataloaders=val_loader,
)