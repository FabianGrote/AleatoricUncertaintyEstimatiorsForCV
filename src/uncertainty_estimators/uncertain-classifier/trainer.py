import torch
from tqdm import tqdm
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import ExponentialLR

from torchmetrics.classification import MulticlassAccuracy, MulticlassCalibrationError, MulticlassConfusionMatrix
from torchmetrics import MeanSquaredError
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import io
from torchvision import transforms
from PIL import Image

# define the LightningModule
class AleatoricUncertaintyEstimator(L.LightningModule):
  def __init__(self, net, criterion_to_use, criterion_dict, predict, num_classes, class_labels):
    super().__init__()
    self.net = net
    self.criterion_to_use = criterion_to_use
    self.criterion_dict = criterion_dict
    self.predict = predict
    self.num_classes = num_classes
    self.class_labels = class_labels

    self.multiclass_top1_accuracy = MulticlassAccuracy(num_classes = self.num_classes, top_k=1)
    self.multiclass_top5_accuracy = MulticlassAccuracy(num_classes = self.num_classes, top_k=5)
    # "l1" as norm is expected calibration error 
    self.multiclass_ece = MulticlassCalibrationError(num_classes = self.num_classes, n_bins=10, norm="l1")
    # "max" as norm is maximum calibration error 
    self.multiclass_mce = MulticlassCalibrationError(num_classes = self.num_classes, n_bins=10, norm="max")
    self.mean_squared_error = MeanSquaredError()
    self.multiclass_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

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
      
    if self.criterion_to_use == "kendall and gal":
      loss = self.criterion_dict["criterion_kendall_and_gal"]([output["logits_variance"][:, :-1], output["sigma2_uc_github_approach"]], target, device=self.device)
      # loss_variance = criterion_dict["criterion_kendall_and_gal_variance"](output["logits_variance"], target)
      # loss_softmax = criterion_dict["criterion_kendall_and_gal_softmax"](output["softmax_output"], target)
      # loss = 0.2 * loss_variance + 1*loss_softmax
  
    elif self.criterion_to_use == "kyles version":
      loss_variance = self.criterion_dict["criterion_kyles_variance"](output["logits_variance"], target)
      loss_softmax = self.criterion_dict["criterion_kyles_softmax"](output["softmax_output"], target)
      loss = 0.2 * loss_variance + 1*loss_softmax
      # acc_top_1 = self.multiclass_top1_accuracy(output["softmax_output"], target)
      # acc_top_5 = self.multiclass_top5_accuracy(output["softmax_output"], target)
      # ece = self.multiclass_ece(preds=output["logits_variance"][:, 0:self.num_classes], target=target)
      # confusion_matrix = self.multiclass_confusion_matrix()
      self.log("train_loss_step_level", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
    
    self.train_output_dict["loss"].append(loss)
    self.train_output_dict["logits"].append(output["logits_variance"][:, 0:self.num_classes])
    self.train_output_dict["softmax_pred"].append(torch.argmax(output["softmax_output"], dim=1))
    self.train_output_dict["target"].append(target) # acc_top_1": acc_top_1, "acc_top_5": acc_top_5, "ece": ece}

    # return loss
    return loss

  # Logging to TensorBoard
  def on_train_epoch_end(self):
    all_loss = torch.stack(self.train_output_dict["loss"])
    all_logits=torch.vstack(self.train_output_dict["logits"])
    all_softmax_pred=torch.hstack(self.train_output_dict["softmax_pred"])
    all_targets=torch.hstack(self.train_output_dict["target"])

    acc_top_1 = self.multiclass_top1_accuracy(preds=all_softmax_pred, target=all_targets)
    acc_top_5 = self.multiclass_top5_accuracy(preds=all_logits, target=all_targets)
    ece = self.multiclass_ece(preds=all_logits, target=all_targets)
    mce = self.multiclass_mce(preds=all_logits, target=all_targets)
    mse = self.mean_squared_error(preds=all_softmax_pred, target=all_targets)
    self.multiclass_confusion_matrix.update(preds=all_softmax_pred, target=all_targets)

    #confusion_matrix = self.multiclass_confusion_matrix(preds=all_softmax_pred, target=all_targets)
    self.log("train_loss_epoch_level", all_loss.mean(), on_step=False, on_epoch=True)
    self.log("train_accuracy_top-1", acc_top_1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log("train_accuracy_top-5", acc_top_5, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log("train_ece", ece, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log("train_mce", mce, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log("train_mse_brier-score", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    fig, ax = plt.subplots(figsize=(self.num_classes, self.num_classes))

    self.multiclass_confusion_matrix.plot(ax=ax, labels=self.class_labels.keys())

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    im = transforms.ToTensor()(Image.open(buf))

    self.logger.experiment.add_image(
        "train_confusion_matrix",
        im,
        global_step=self.current_epoch,
    )

    # self.log("train_accuracy_top-1", training_step_outputs.acc_top_1.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    # self.log("train_accuracy_top-5", training_step_outputs.acc_top_5.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    # self.log("train_ece", training_step_outputs.ece.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    # self.training_step_outputs.clear()  # free memory

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
    
    output = self.net(data)
    
    if self.criterion_to_use == "kendall and gal":
      loss = self.criterion_dict["criterion_kendall_and_gal"]([output["logits_variance"][:, :-1], output["sigma2_uc_github_approach"]], target, device=self.device)
      # loss_variance = criterion_dict["criterion_kendall_and_gal_variance"](output["logits_variance"], target)
      # loss_softmax = criterion_dict["criterion_kendall_and_gal_softmax"](output["softmax_output"], target)
      # loss = 0.2 * loss_variance + 1*loss_softmax
  
    elif self.criterion_to_use == "kyles version":
      loss_variance = self.criterion_dict["criterion_kyles_variance"](output["logits_variance"], target)
      loss_softmax = self.criterion_dict["criterion_kyles_softmax"](output["softmax_output"], target)
      loss = 0.2 * loss_variance + 1*loss_softmax

      # Logging to TensorBoard
      self.log("val_loss", loss.detach().item(), sync_dist=True) #, on_step=False, on_epoch=True, sync_dist=True)

    self.val_output_dict["loss"].append(loss)
    self.val_output_dict["logits"].append(output["logits_variance"][:, 0:self.num_classes])
    self.val_output_dict["softmax_pred"].append(torch.argmax(output["softmax_output"], dim=1))
    self.val_output_dict["target"].append(target) # acc_top_1": acc_top_1, "acc_top_5": acc_top_5, "ece": ece}

    #return {"loss": loss, "logits":output["logits_variance"][:, 0:self.num_classes], "softmax_pred":output["softmax_output"], "target": target} # acc_top_1": acc_top_1, "acc_top_5": acc_top_5, "ece": ece}
    return loss
      # return sum(scores)/len(test_loader), sum(losses)/len(test_loader)
    
  # Logging to TensorBoard
  def on_validation_epoch_end(self):
    all_loss = torch.stack(self.val_output_dict["loss"])
    all_logits=torch.vstack(self.val_output_dict["logits"])
    all_softmax_pred=torch.hstack(self.val_output_dict["softmax_pred"])
    all_targets=torch.hstack(self.val_output_dict["target"])

    acc_top_1 = self.multiclass_top1_accuracy(preds=all_softmax_pred, target=all_targets)
    acc_top_5 = self.multiclass_top5_accuracy(preds=all_logits, target=all_targets)
    ece = self.multiclass_ece(preds=all_logits, target=all_targets)
    mce = self.multiclass_mce(preds=all_logits, target=all_targets)
    mse = self.mean_squared_error(preds=all_softmax_pred, target=all_targets)
    self.multiclass_confusion_matrix.update(preds=all_softmax_pred, target=all_targets)

    self.log("val_loss_epoch_level", all_loss.mean(), on_step=False, on_epoch=True)
    self.log("val_accuracy_top-1", acc_top_1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_accuracy_top-5", acc_top_5, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_ece", ece, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_mce", mce, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_mse_brier-score", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    fig, ax = plt.subplots(figsize=(self.num_classes, self.num_classes))

    self.multiclass_confusion_matrix.plot(ax=ax, labels=self.class_labels.keys())

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    im = transforms.ToTensor()(Image.open(buf))

    self.logger.experiment.add_image(
        "val_confusion_matrix",
        im,
        global_step=self.current_epoch,
    )

    # self.log("train_accuracy_top-1", training_step_outputs.acc_top_1.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    # self.log("train_accuracy_top-5", training_step_outputs.acc_top_5.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    # self.log("train_ece", training_step_outputs.ece.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    # self.validation_step_outputs.clear()  # free memory


  def configure_optimizers(self):
    kwargs = dict(lr=1e-4, weight_decay=0.0001)
    optimizer = torch.optim.RMSprop(self.net.parameters(), lr=0.001, weight_decay=0.0001)
    #optimizer = torch.optim.Adam(self.net.parameters(), **kwargs)
    scheduler = ExponentialLR(optimizer, gamma=0.9999)
    return [optimizer], [scheduler]
