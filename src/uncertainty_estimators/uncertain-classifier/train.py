import sys

import torch
from torch import nn
from torchvision import transforms, datasets
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import aleatoric_uncertainty_estimator_model, trainer, loss_functions

# import torch.multiprocessing as mp
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group

import os
print("Working dir:", os.getcwd())

# def ddp_setup(rank: int, world_size: int):
#   """
#     Args:
#     rank: Unique identifier of each process
#     world_size: Total number of processes
#   """
#   os.environ["MASTER_ADDR"] = "localhost"
#   os.environ["MASTER_PORT"] = "12355"
#   torch.cuda.set_device(rank)
#   # backend for distributed stuff
#   init_process_group(backend="nccl", rank=rank, world_size=world_size)


train_dataset = datasets.ImageNet( # Imagenette
    root = "/mnt/HDD1/datasets/ImageNet2012", # nette-download", #Net2012", # ImageNet2012",
    split = "val", # "train",
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
  )

val_dataset = datasets.ImageNet( # Imagenette
    root = "/mnt/HDD1/datasets/ImageNet2012", #nette-download", # Net2012",
    split = "val",
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
  )

train_loader = torch.utils.data.DataLoader(
  train_dataset, batch_size=256, shuffle=False, num_workers=8 # sampler=DistributedSampler(train_dataset)
)

val_loader = torch.utils.data.DataLoader(
  val_dataset, batch_size=256, shuffle=False, num_workers=8 #, sampler=DistributedSampler(val_dataset)
)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = len(train_dataset.classes)
net = aleatoric_uncertainty_estimator_model.Net(
  image_size = (3, 224, 224), # channels x width x height
  num_classes = num_classes, # imagenette: 10, 
  encoder = "resnet50"
)

# default logger used by trainer (if tensorboard is installed)
logger = TensorBoardLogger(
  save_dir=os.getcwd(),
  # version=1, 
  name="lightning_logs")

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
criterion_to_use = "kendall and gal" # or "kyles version"

predict = aleatoric_uncertainty_estimator_model.predict

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)#
aleatoricUncertaintyEstimator = trainer.AleatoricUncertaintyEstimator(
  net = net,
  criterion_to_use = criterion_to_use,
  criterion_dict = criterion_dict, 
  predict = predict
)
trainer = L.Trainer(
  check_val_every_n_epoch=5,
  max_epochs=500, 
  devices=1,
  num_nodes=1,
  accelerator="gpu",
  enable_checkpointing=True,
  log_every_n_steps = 100,
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