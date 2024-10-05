import torch
from tqdm import tqdm
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import ExponentialLR

from torchmetrics.classification import MulticlassAccuracy, MulticlassCalibrationError, MulticlassConfusionMatrix, MulticlassROC, MulticlassAUROC
from torchmetrics import MeanSquaredError
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import io
from torchvision import transforms
from PIL import Image
from data_augmentation import DataAugmentation

# define the LightningModule
class AleatoricUncertaintyEstimator(L.LightningModule):
  def __init__(self, net, criterion_to_use, criterion_dict, predict, num_classes, class_labels, log_confusion_matrix, augment_val_data, num_data_augmentations):
    super().__init__()
    self.net = net
    self.criterion_to_use = criterion_to_use
    self.criterion_dict = criterion_dict
    self.predict = predict
    self.num_classes = num_classes
    self.class_labels = class_labels
    self.augment_val_data = augment_val_data
    self.num_data_augmentations = num_data_augmentations
    self.data_augmentation = DataAugmentation(augment_data=True)


    self.multiclass_top1_accuracy = MulticlassAccuracy(num_classes = self.num_classes, top_k=1)
    self.multiclass_top5_accuracy = MulticlassAccuracy(num_classes = self.num_classes, top_k=5)
    # "l1" as norm is expected calibration error 
    self.multiclass_ece = MulticlassCalibrationError(num_classes = self.num_classes, n_bins=10, norm="l1")
    # "max" as norm is maximum calibration error 
    self.multiclass_mce = MulticlassCalibrationError(num_classes = self.num_classes, n_bins=10, norm="max")
    self.mean_squared_error = MeanSquaredError()
    self.multiclass_roc = MulticlassROC(num_classes=self.num_classes)
    self.multiclass_auroc = MulticlassAUROC(num_classes=self.num_classes)
    self.multiclass_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)
    self.log_confusion_matrix = log_confusion_matrix

  def on_train_epoch_start(self) -> None:
    super().on_train_epoch_start()
    self.train_output_dict = {
        "loss": [],
        "logits": [],
        "softmax_pred": [],
        "target":  []
    }
    return

  def training_step(self, batch, batch_idx):
    data, target = batch
    output = self.net(data)
      
    if self.criterion_to_use == "kendall_and_gal":
      loss = self.criterion_dict["criterion_kendall_and_gal"]([output["logits_variance"][:, :-1], output["sigma2_uc_github_approach"]], target, device=self.device)
      # loss_variance = criterion_dict["criterion_kendall_and_gal_variance"](output["logits_variance"], target)
      # loss_softmax = criterion_dict["criterion_kendall_and_gal_softmax"](output["softmax_output"], target)
      # loss = 0.2 * loss_variance + 1*loss_softmax
  
    elif self.criterion_to_use == "kyles_version":
      loss_variance = self.criterion_dict["criterion_kyles_variance"](output["logits_variance"], target)
      loss_softmax = self.criterion_dict["criterion_kyles_softmax"](output["softmax_output"], target)
      loss = 0.2 * loss_variance + 0.8*loss_softmax  # before: 1*loss_softmax
      # acc_top_1 = self.multiclass_top1_accuracy(output["softmax_output"], target)
      # acc_top_5 = self.multiclass_top5_accuracy(output["softmax_output"], target)
      # ece = self.multiclass_ece(preds=output["logits_variance"][:, 0:self.num_classes], target=target)
      # confusion_matrix = self.multiclass_confusion_matrix()
    elif self.criterion_to_use == "softmax_only":
      loss = self.criterion_dict["criterion_kyles_softmax"](output["softmax_output"], target)
    
    self.train_output_dict["loss"].append(loss)
    self.train_output_dict["logits"].append(output["logits_variance"][:, 0:self.num_classes])
    self.train_output_dict["softmax_pred"].append(torch.argmax(output["softmax_output"], dim=1))
    self.train_output_dict["target"].append(target) # acc_top_1": acc_top_1, "acc_top_5": acc_top_5, "ece": ece}

    return loss

  # Logging to TensorBoard
  def on_train_epoch_end(self):
    self.log_metrics(output_dict = self.train_output_dict, prefix="train")
    self.train_output_dict.clear()  # free memory

  def on_validation_epoch_start(self) -> None:
    super().on_validation_epoch_start()
    self.val_output_dict = {
      "loss": [],
      "logits": [],
      "softmax_pred": [],
      "target":  []
    }
    return

  def validation_step(self, batch, batch_idx):
    data, target = batch
    # output = self.predict(data, self.net, device=self.device, T=1000, class_count=self.net.num_classes)
      
    # predictions = torch.argmax(output.data, 1)
    # score = (predictions == target).float().mean().detach().item()
      
    # # Logging to TensorBoard
    # self.log("val_score", score, sync_dist=True) #, on_step=False, on_epoch=True, sync_dist=True)
    if self.augment_val_data:
      softmax_output_all = []
      logits_variance_all = []
      sigma2_uc_github_approach_all = []

      for i in range(self.num_data_augmentations):
        augmented_data = self.data_augmentation(data)
        single_output = self.net(augmented_data)
        softmax_output_all.append(single_output["softmax_output"])
        logits_variance_all.append(single_output["logits_variance"])
        sigma2_uc_github_approach_all.append(single_output["sigma2_uc_github_approach"])

      # Convert from a list of T items to tensor   
      softmax_output = torch.vstack(softmax_output_all)
      logits_variance = torch.vstack(logits_variance_all)
      sigma2_uc_github_approach = torch.vstack(sigma2_uc_github_approach_all)

      # use the median of T predictions for the final class membership: Mx1x5
      output = {
        "softmax_output": torch.median(softmax_output, dim=0).values.unsqueeze(dim=0),
        "logits_variance": torch.median(logits_variance, dim=0).values.unsqueeze(dim=0),
        "sigma2_uc_github_approach": torch.median(sigma2_uc_github_approach, dim=0).values.unsqueeze(dim=0)
      }
    
    else:
      output = self.net(data)
    
    if self.criterion_to_use == "kendall_and_gal":
      loss = self.criterion_dict["criterion_kendall_and_gal"]([output["logits_variance"][:, :-1], output["sigma2_uc_github_approach"]], target, device=self.device)
      # loss_variance = criterion_dict["criterion_kendall_and_gal_variance"](output["logits_variance"], target)
      # loss_softmax = criterion_dict["criterion_kendall_and_gal_softmax"](output["softmax_output"], target)
      # loss = 0.2 * loss_variance + 1*loss_softmax
  
    elif self.criterion_to_use == "kyles_version":
      loss_variance = self.criterion_dict["criterion_kyles_variance"](output["logits_variance"], target)
      loss_softmax = self.criterion_dict["criterion_kyles_softmax"](output["softmax_output"], target)
      loss = 0.2 * loss_variance + 0.8*loss_softmax  # before: 1*loss_softmax
    
    elif self.criterion_to_use == "softmax_only":
      loss = self.criterion_dict["criterion_kyles_softmax"](output["softmax_output"], target)

    self.val_output_dict["loss"].append(loss)
    self.val_output_dict["logits"].append(output["logits_variance"][:, 0:self.num_classes])
    self.val_output_dict["softmax_pred"].append(torch.argmax(output["softmax_output"], dim=1))
    self.val_output_dict["target"].append(target) # acc_top_1": acc_top_1, "acc_top_5": acc_top_5, "ece": ece}

    return loss
      # return sum(scores)/len(test_loader), sum(losses)/len(test_loader)
    
  # Logging to TensorBoard
  def on_validation_epoch_end(self):
    self.log_metrics(output_dict = self.val_output_dict, prefix="val")
    self.val_output_dict.clear()  # free memory



  def configure_optimizers(self):
    kwargs = dict(lr=1e-4, weight_decay=0.0001)
    # optimizer = torch.optim.RMSprop(self.net.parameters(), lr=0.001, weight_decay=0.0001)
    optimizer = torch.optim.Adam(self.net.parameters(), **kwargs)
    scheduler = ExponentialLR(optimizer, gamma=0.9999)
    return [optimizer], [scheduler]


  def log_metrics(self, output_dict, prefix):
    all_loss = torch.stack(output_dict["loss"])
    all_logits=torch.vstack(output_dict["logits"])
    all_softmax_pred=torch.hstack(output_dict["softmax_pred"])
    all_targets=torch.hstack(output_dict["target"])

    acc_top_1 = self.multiclass_top1_accuracy(preds=all_softmax_pred, target=all_targets)
    acc_top_5 = self.multiclass_top5_accuracy(preds=all_logits, target=all_targets)
    ece = self.multiclass_ece(preds=all_logits, target=all_targets)
    mce = self.multiclass_mce(preds=all_logits, target=all_targets)
    mse = self.mean_squared_error(preds=all_softmax_pred, target=all_targets)
    roc = self.multiclass_roc(preds=all_softmax_pred, target=all_targets)
    auroc = self.multiclass_auroc(preds=all_softmax_pred, target=all_targets)
    
    self.multiclass_confusion_matrix.update(preds=all_softmax_pred, target=all_targets)

    self.log(prefix + "_loss_epoch_level", all_loss.mean(), on_step=False, on_epoch=True)
    self.log(prefix + "_accuracy_top-1", acc_top_1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log(prefix + "_accuracy_top-5", acc_top_5, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log(prefix + "_ece", ece, on_step=False, on_epoch=True, prog_bar=False, logger=True)
    self.log(prefix + "_mce", mce, on_step=False, on_epoch=True, prog_bar=False, logger=True)
    self.log(prefix + "_mse_brier-score", mse, on_step=False, on_epoch=True, prog_bar=False, logger=True)
    self.log(prefix + "_roc", roc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
    self.log(prefix + "_auroc" , auroc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    # log confusion matrix only for val and every x train epoch because it creates and saves an memory expensive image every time
    if self.log_confusion_matrix and (prefix == "val" or self.current_epoch%25==0):
      fig, ax = plt.subplots(figsize=(self.num_classes, self.num_classes))

      self.multiclass_confusion_matrix.plot(ax=ax, labels=self.class_labels.keys(), cmap="OrRd")

      buf = io.BytesIO()
      fig.savefig(buf, format="png", bbox_inches="tight")
      buf.seek(0)
      im = transforms.ToTensor()(Image.open(buf))

      self.logger.experiment.add_image(
          prefix + "_confusion_matrix",
          im,
          global_step=self.current_epoch,
      )
      self.multiclass_confusion_matrix.reset()

    # self.log("train_accuracy_top-1", training_step_outputs.acc_top_1.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    # self.log("train_accuracy_top-5", training_step_outputs.acc_top_5.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    # self.log("train_ece", training_step_outputs.ece.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    # self.validation_step_outputs.clear()  # free memory
