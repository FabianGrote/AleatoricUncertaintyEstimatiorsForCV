import torch
from tqdm import tqdm
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import ExponentialLR

# define the LightningModule
class AleatoricUncertaintyEstimator(L.LightningModule):
  def __init__(self, net, criterion_to_use, criterion_dict, predict):
    super().__init__()
    self.net = net
    self.criterion_to_use = criterion_to_use
    self.criterion_dict = criterion_dict
    self.predict = predict

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
    
    # Logging to TensorBoard
    self.log("train_loss", loss) # , on_step=False, on_epoch=True)
    return loss

  def validation_step(self, batch, batch_idx):
    data, target = batch
    output = self.predict(data, self.net, device=self.device, T=1000, class_count=self.net.num_classes)
      
    predictions = torch.argmax(output.data, 1)
    score = (predictions == target).float().mean().detach().item()
      
    # Logging to TensorBoard
    self.log("val_score", score, sync_dist=True) #, on_step=False, on_epoch=True, sync_dist=True)
    
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

      return loss
      # return sum(scores)/len(test_loader), sum(losses)/len(test_loader)

  def configure_optimizers(self):
    kwargs = dict(lr=1e-4, weight_decay=0.0001)
    optimizer = torch.optim.Adam(self.net.parameters(), **kwargs)
    scheduler = ExponentialLR(optimizer, gamma=0.9999)
    return [optimizer], [scheduler]