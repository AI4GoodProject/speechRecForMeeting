import torch
from sklearn.metrics import roc_curve

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, 1)     
         
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

def initialize(features=None):
    n = len(features)
    p = features[0].size(1)*features[0].size(2)
    features = torch.reshape(features, (n, p))
    return [LogisticRegression(n), features]


def train(model, train_dataloader, val_dataloader, criterion, optimizer=None, num_epochs=5, use_gpu = False):
  train_losses = []
  test_losses = []

  train_error_rates = []
  test_error_rates = []
  
  if use_gpu:
    # switch model to GPU
    CNN.cuda()

  for epoch in range(num_epochs): 
    train_loss = 0 
    n_iter = 0 
    total = 0
    correct = 0

    for i, (images, labels) in enumerate(train_dataloader): 

      if use_gpu: 
        images = images.cuda()
        labels = labels.cuda()

      # print(images.shape, 'train')
      outputs = model(images)
      print('outputs: ', type(outputs), outputs[0:10])
      
      # to compute the train_error_rates  
      _, predictions = torch.max(outputs, 1)
      print('predictions: ', type(predictions), predictions[0:10])
      correct += torch.sum(labels == predictions).item()
      total += labels.shape[0]
      
      # compute loss
      loss_bs = criterion(outputs, labels)
      # compute gradients
      loss_bs.backward()
      # update weights
      optimizer.step()

      train_loss += loss_bs.detach().item()

      n_iter += 1

    train_error_rate = 1 - correct/total

    with torch.no_grad():
      predictions, test_loss, test_error_rate = prediction(val_dataloader, CNN, criterion)

    train_error_rates.append(train_error_rate)
    test_error_rates.append(test_error_rate)
    train_losses.append(train_loss) # train_losses.append(train_loss/n_iter)
    test_losses.append(test_loss)

    if epoch%1 == 0:
      print('Epoch: {}/{}, Loss: {:.4f}, Error Rate: {:.1f}%'.format(epoch+1, num_epochs, train_loss/n_iter, 100*train_error_rate))
  print('Finished Training')
  return train_error_rates, train_losses, test_error_rates, test_losses #list of y_true and y_hat

