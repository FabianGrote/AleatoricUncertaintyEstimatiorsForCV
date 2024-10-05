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
  def __init__(self, net, criterion_to_use, criterion_dict, predict, num_classes, class_labels, log_confusion_matrix):
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
    self.multiclass_confusion_matrix.update(preds=all_softmax_pred, target=all_targets)

    self.log(prefix + "_loss_epoch_level", all_loss.mean(), on_step=False, on_epoch=True)
    self.log(prefix + "_accuracy_top-1", acc_top_1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log(prefix + "_accuracy_top-5", acc_top_5, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log(prefix + "_ece", ece, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log(prefix + "_mce", mce, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log(prefix + "_mse_brier-score", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

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




  # def inference_with_test_time_data_aug(reader, model, test_input, test_input_aug, T=32, k=-1):
  #   labels_all = []
  #   predictions_all = []
    
  #   kkk = 0
    
  #   while not reader.exhausted_test_cases:
  #       org_ex, label, _ = reader.next_batch(batch_size=1, normalize=True, shuffle=False)
        
  #       feed_img = {test_input: org_ex}
  #       images = []
  #       labels = []
  #       for i in range(T):
  #           aug_ex = np.squeeze(model.session.run([test_input_aug], feed_dict=feed_img))
  #           images.append(aug_ex)
  #           labels.append(label)
            
  #       x_batch = np.reshape(np.asarray(images, dtype=np.float32), [-1, 512, 512, 3])
  #       y_batch = np.reshape(np.asarray(labels, dtype=np.float32), [-1, 1])
        
  #       predictions, labels = model.session.run([model.predictions_1hot, model.labels_1hot], 
  #                                               feed_dict=feed_dict(model, x_batch, y_batch)
  #                                              )        
  #       labels_all.append(labels[0])
  #       predictions_all.append(predictions)        
        
  #       if k != -1:
  #           k = k - 1
  #           if k == 0:
  #               dr.exhausted_test_cases = True
  #           print('k = %d' % k)
  #       kkk = kkk + 1
  #       if kkk % 1000 == 0:
  #           print('kkk = %d' % kkk)
  #   print('kkk = %d' % kkk)
    
  #   # Convert from a list of M items of size Tx5 to an array of dims MxTx5. For labels_1hot: Mx5.   
  #   labels_1hot = np.asarray(labels_all)
    
  #   predictions_all = np.asarray(predictions_all)
    
  #   # use the median of T predictions for the final class membership: Mx1x5
  #   predictions_1hot_median = np.median(predictions_all, axis=1)
    
  #   correct = np.equal(np.argmax(labels_1hot, axis=1), np.argmax(predictions_1hot_median, axis=1))
  #   acc = np.mean(np.asarray(correct, dtype=np.float32))
  #   print('Accuracy : %.5f' % acc)
        
  #   onset_level = 1
  #   labels_bin = np.greater_equal(np.argmax(labels_1hot, axis=1), onset_level)
  #   pred_bin = np.sum(predictions_all[:, :, onset_level:], axis=2) # MxTx1
  #   pred_bin_median = np.median(pred_bin, axis=1) # Mx1x1  
  #   fpr, tpr, _ = roc_curve(labels_bin, np.squeeze(pred_bin_median))
  #   roc_auc_onset1 = auc(fpr, tpr)
  #   print('Onset level = %d\t ROC-AUC: %.5f' % (onset_level, roc_auc_onset1))
            
  #   onset_level = 2
  #   labels_bin = np.greater_equal(np.argmax(labels_1hot, axis=1), onset_level)
  #   pred_bin = np.sum(predictions_all[:, :, onset_level:], axis=2) # MxTx1
  #   pred_bin_median = np.median(pred_bin, axis=1) # Mx1x1  
  #   fpr, tpr, _ = roc_curve(labels_bin, np.squeeze(pred_bin_median))
  #   roc_auc_onset1 = auc(fpr, tpr)
  #   print('Onset level = %d\t ROC-AUC: %.5f' % (onset_level, roc_auc_onset1))
        
  #   return labels_1hot, predictions_all