import sys

import torch
from torch import nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import argparse
import time

import aleatoric_uncertainty_estimator_model, trainer, loss_functions
from dataloader import get_dataset

import yaml
from pathlib import Path

import os
print("Working dir:", os.getcwd())

# local conda env: conda activate fabis_uncertainty_env
# local execution: python3 train.py --accelerator='gpu' --devices=1 --num_nodes=1 --max_epochs=200 --dataset="MNIST" --freeze_encoder_params=True --criterion_to_use="softmax_only"
# parser = argparse.ArgumentParser()
# parser.add_argument("--accelerator", default="cpu", help="cpu or gpu", type=str)
# parser.add_argument("--devices", default=1, help="Number of GPU nodes for distributed training.", type=int)
# parser.add_argument("--num_nodes", default=1, help="Number of GPU nodes for distributed training.", type=int)
# parser.add_argument("--max_epochs", default=100, help="Stop training once this number of epochs is reached.", type=int)
# parser.add_argument("--dataset", default="Imagenette", help="Dataset to use. Can be: Imagenet, Imagenette, MNIST, GTSRB, ...", type=str)
# parser.add_argument("--freeze_encoder_params", default=True, help="If encoder parameters should be freezed or not.", type=bool)
# parser.add_argument("--criterion_to_use", default="softmax_only", help="Criterion to use. Can be 'kendall_and_gal', 'kyles_version' or 'softmax_only'", type=str)
# parser.add_argument("--data_augmentation", default="False", help="Set if test time data augmentation should be performed. Can be true or false.", type=bool)
# args = parser.parse_args()

# dataset_name = args.dataset

my_path = Path(__file__).resolve()
config_path = my_path.parent / 'config.yaml'
with open(config_path, 'r') as file:
  config = yaml.safe_load(file)

dataset_name = config["dataset_name"]
freeze_encoder_params = config["freeze_encoder_params"]

train_dataset, val_dataset, class_labels, image_size = get_dataset(dataset_name=dataset_name, 
                                                                   augment_data=config["augment_data"], 
                                                                   num_data_augmentations = config["num_data_augmentations"]
                                                                  )

train_loader = torch.utils.data.DataLoader(
  train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8 # sampler=DistributedSampler(train_dataset)
)
val_loader = torch.utils.data.DataLoader(
  val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8 #, sampler=DistributedSampler(val_dataset)
)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#
# GTSRB pytorch dataset module doesn't have .classes parameter
if dataset_name == "GTSRB":
  num_classes = 43
else:
  num_classes = len(train_dataset.classes)

net = aleatoric_uncertainty_estimator_model.Net(
  image_size = image_size, # channels x width x height
  num_classes = num_classes, # imagenette: 10, 
  encoder = "resnet50",
  freeze_encoder_params = freeze_encoder_params
)

lr_monitor = LearningRateMonitor(logging_interval='step')

# UC uses 2 times num classes as linear output in loss, kyle uses logits_variance and softmax in loss
criterion_dict = {
  # Loss used by "Uncertainty classifier"-GitHub Repo
  "criterion_kendall_and_gal": loss_functions.Loss(num_classes=num_classes, T=config["num_mc_sims"]),

  # For logits_variance network output in Kyles version
  "criterion_kyles_variance": loss_functions.BayesianCategoricalCrossEntropy_KylesVersion(T=config["num_mc_sims"], num_classes=num_classes),
  # For softmax network output in Kyles version
  "criterion_kyles_softmax": nn.CrossEntropyLoss(),
}
criterion_to_use = config["criterion_to_use"] # "kendall_and_gal" or "kyles_version" or "softmax_only"

time = time.strftime("%Y%m%d_%H-%M")
# default logger used by trainer (if tensorboard is installed)
logger = TensorBoardLogger(
  save_dir=os.getcwd(),
  name="lightning_logs",
  version=dataset_name + "_" + criterion_to_use + "_freeze_encoder-" + str(freeze_encoder_params) + "_augmentation_data-" + str(config["augment_data"]) + "_" + time
)


predict = aleatoric_uncertainty_estimator_model.predict

if dataset_name == "ImageNet":
  log_confusion_matrix = False
else:
  log_confusion_matrix = True

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)#
aleatoricUncertaintyEstimator = trainer.AleatoricUncertaintyEstimator(
  net = net,
  criterion_to_use = criterion_to_use,
  criterion_dict = criterion_dict, 
  predict = predict,
  num_classes = num_classes,
  class_labels = class_labels,
  log_confusion_matrix=log_confusion_matrix
)
trainer = L.Trainer(
  check_val_every_n_epoch=5,
  max_epochs=config["max_epochs"],
  devices=config["devices"],
  num_nodes=config["num_nodes"],
  accelerator=config["accelerator"],
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