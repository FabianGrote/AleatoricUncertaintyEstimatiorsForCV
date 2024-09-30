import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from torchvision.models import resnet50


class Net(nn.Module):
  def __init__(self, image_size, num_classes, encoder): # input_size, output_size, hidden_size, hidden_count):
    super(Net, self).__init__()
    self.image_size = image_size
    self.num_classes = num_classes
    self.backbone = self.create_encoder_model(encoder)

    # bayesian network part
    backbone_output_size = 2048 # Imagenette with image size 3 x 224 x 224
    self.batch_norm_1 = nn.BatchNorm1d(num_features=backbone_output_size)
    self.batch_norm_2 = nn.BatchNorm1d(num_features=500)
    self.batch_norm_3 = nn.BatchNorm1d(num_features=100)
    self.linear_1 = nn.Linear(in_features=backbone_output_size, out_features=500)
    self.linear_2 = nn.Linear(in_features=500, out_features=100)
    self.linear_3 = nn.Linear(in_features=100, out_features=num_classes)
    self.relu_1 = nn.ReLU()
    self.relu_2 = nn.ReLU()
    self.linear_variance = nn.Linear(in_features=100, out_features=1)
    self.softplus = nn.Softplus()
    self.softmax = nn.Softmax(dim=-1)

    # # Layer only used for Uncertainty Classifier GitHub that takes 2*num_classes as output size
    self.sigma2_uc_github_approach_linear = nn.Linear(in_features=100, out_features=num_classes)

  def forward(self, x):
    backbone_output = self.backbone(x)
    output_dict = self.bayesian_model(backbone_output) #, backbone_output_size=backbone_output.shape[1], num_classes=self.num_classes)
    return output_dict
    # self.mu(x), self.log_sigma2(x)
  
  
  def create_encoder_model(self, encoder):
    # Resnet50 as backbone without last fully connected layer
    if encoder == 'resnet50':
      base_model = torch.nn.Sequential(*(list(
        resnet50(
          weights = "ResNet50_Weights.IMAGENET1K_V1"     
        ).children())[:-1]),
        nn.Flatten(start_dim=1, end_dim=-1)
      )
    else:
      raise ValueError('Unexpected encoder model ' + encoder + ".")

    # freeze encoder layers to prevent over fitting
    # for param in base_model.parameters():
    #   param.requires_grad = False

    return base_model
  
  def bayesian_model(self, x):    
    x = self.batch_norm_1(x)
    #x = nn.Dropout(p=0.5)(x)
    x = self.linear_1(x)
    x = self.relu_1(x)

    x = self.batch_norm_2(x)
    #x = nn.Dropout(p=0.5)(x)
    x = self.linear_2(x)
    x = self.relu_2(x)
    
    x = self.batch_norm_3(x)
    #x = nn.Dropout(p=0.5)(x)
    logits = self.linear_3(x)

    variance_pre = self.linear_variance(x)
    variance = self.softplus(variance_pre)
    logits_variance = torch.cat((logits, variance), dim=-1)
    softmax_output = self.softmax(logits)

    sigma2_uc_github_approach = self.sigma2_uc_github_approach_linear (x)

    # UC uses ?, kyle uses logits_variance and softmax
    return {"softmax_output": softmax_output, "logits_variance": logits_variance, "sigma2_uc_github_approach": sigma2_uc_github_approach}
  


def predict(data, net, device, T=1000, class_count=10):
  mvn = MultivariateNormal(torch.zeros(class_count), torch.eye(class_count))
  output = net(data)
  mu = output["logits_variance"][:, :-1]
  log_sigma2 = output["sigma2_uc_github_approach"]
  y_hat = torch.zeros_like(mu)
    
  for t in range(T):
    y_hat += F.softmax(mu + torch.exp(0.5*log_sigma2)*mvn.sample((len(mu),)).to(device), dim=1).detach() / T

  return y_hat / T
