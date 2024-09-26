import torch
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal

def train(train_loader, net, criterion_to_use, criterion_dict, optimizer, scheduler, device):
  train_losses = []
  for data, target in tqdm(train_loader):
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    output = net(data)
    
    if criterion_to_use == "kendall and gal":
      loss = criterion_dict["criterion_kendall_and_gal"]([output["logits_variance"][:, :-1], output["sigma2_uc_github_approach"]], target)
      # loss_variance = criterion_dict["criterion_kendall_and_gal_variance"](output["logits_variance"], target)
      # loss_softmax = criterion_dict["criterion_kendall_and_gal_softmax"](output["softmax_output"], target)
      # loss = 0.2 * loss_variance + 1*loss_softmax
  
    elif criterion_to_use == "kyles version":
      loss_variance = criterion_dict["criterion_kyles_variance"](output["logits_variance"], target)
      loss_softmax = criterion_dict["criterion_kyles_softmax"](output["softmax_output"], target)
      loss = 0.2 * loss_variance + 1*loss_softmax
  
    train_losses.append(loss.item())
    loss.backward()
    optimizer.step()
  return train_losses

def test(test_loader, predict, net, criterion_to_use, criterion_dict, device):
  scores = []
  losses = []
  net.train()
  for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    output = predict(data, net)
    
    predictions = torch.argmax(output.data, 1)
    scores.append((predictions == target).float().mean().detach().item())
    
    output = net(data)
    
    if criterion_to_use == "kendall and gal":
      loss = criterion_dict["criterion_kendall_and_gal"]([output["logits_variance"][:, :-1], output["sigma2_uc_github_approach"]], target)
      # loss_variance = criterion_dict["criterion_kendall_and_gal_variance"](output["logits_variance"], target)
      # loss_softmax = criterion_dict["criterion_kendall_and_gal_softmax"](output["softmax_output"], target)
      # loss = 0.2 * loss_variance + 1*loss_softmax
  
    elif criterion_to_use == "kyles version":
      loss_variance = criterion_dict["criterion_kyles_variance"](output["logits_variance"], target)
      loss_softmax = criterion_dict["criterion_kyles_softmax"](output["softmax_output"], target)
      loss = 0.2 * loss_variance + 1*loss_softmax

    losses.append(loss.detach().item())
  return sum(scores)/len(test_loader), sum(losses)/len(test_loader)




def predict(data, net, T=100, class_count=10):
  mvn = MultivariateNormal(torch.zeros(class_count), torch.eye(class_count))
  output = net(data)
  mu = output["logits_variance"][:, :-1]
  log_sigma2 = output["sigma2_uc_github_approach"]
  y_hat = torch.zeros_like(mu)
    
  for t in range(T):
    y_hat += F.softmax(mu + torch.exp(0.5*log_sigma2)*mvn.sample((len(mu),)), dim=1).detach() / T

  return y_hat / T
