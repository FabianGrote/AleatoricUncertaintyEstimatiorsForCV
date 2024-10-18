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
# local execution: python3 train.py --config_name=

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", default="False", help="Which config to use for loading parameters.", type=str)
args = parser.parse_args()

my_path = Path(__file__).resolve()
config_path = my_path.parent / (args.config_name + '.yaml')
with open(config_path, 'r') as file:
  config = yaml.safe_load(file)

dataset_name = config["dataset_name"]
print("Training on dataset: ", dataset_name)
freeze_encoder_params = config["freeze_encoder_params"]

train_dataset, val_dataset, class_labels, image_size = get_dataset(
   dataset_root_path = config["dataset_root_path"],
   dataset_name=dataset_name, 
   augment_data=config["augment_data"], 
   num_data_augmentations=config["num_data_augmentations"],
   rotation_and_flip = config["rotation_and_flip"]
)

train_loader = torch.utils.data.DataLoader(
  train_dataset, batch_size=config["train_batch_size"], shuffle=True, num_workers=8 # sampler=DistributedSampler(train_dataset)
)

val_loader = torch.utils.data.DataLoader(
  val_dataset, batch_size=config["val_batch_size"], shuffle=False, num_workers=8 #, sampler=DistributedSampler(val_dataset)
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
log_folder = (
  dataset_name + "_" + criterion_to_use + "_freeze_encoder-" + str(freeze_encoder_params) 
  + "_augmentation_data-" + str(config["augment_data"]) + "_rotation_and_flip-" + str(config["rotation_and_flip"]) + "_" + time
)
logger = TensorBoardLogger(
  save_dir=os.getcwd(),
  name="lightning_logs",
  version=log_folder
)

log_path = my_path.parent / "lightning_logs" / log_folder
if not os.path.exists(log_path):
    os.makedirs(log_path)
with open(log_path / 'config_save.yaml', 'w') as f:
    yaml.dump(config, f,  default_flow_style=False)

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
  log_confusion_matrix=log_confusion_matrix,
  augment_val_data=config["augment_data"],
  num_data_augmentations=config["num_data_augmentations"],
)
trainer = L.Trainer(
  check_val_every_n_epoch=config["check_val_every_n_epoch"],
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
