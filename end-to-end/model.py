import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn.functional as F
from copy import deepcopy
from sklearn.metrics import confusion_matrix

LEARNING_RATE = 1e-3
BATCH_SIZE = 256
PATIENCE_WINDOW = 10
CLASSIFIER_AUTHENTICATOR_SPLIT = 0.75
TEST_TRAIN_SPLIT = 0.2
RANDOM_SEED = 70

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def get_accuracy(model, x, y):
    out = model(x)
    preds = out.argmax(dim=1)
    correct = preds.eq(y.argmax(dim=1)).sum().item()
    return correct / len(preds)

def create_dataset(data):
  classifier_X = []
  classifier_y = []
  authenticator_session_1_X = []
  authenticator_session_1_y = []
  authenticator_session_2_X = []
  authenticator_session_2_y = []

  np.random.shuffle(data)

  classifier_participants = np.arange(int(CLASSIFIER_AUTHENTICATOR_SPLIT*len(data))) 
  authenticator_participants = np.arange(int(CLASSIFIER_AUTHENTICATOR_SPLIT*len(data)), len(data))

  classifier_data = data[classifier_participants]
  authenticator_data = data[authenticator_participants]

  for x in range(len(classifier_data)):
    for y in range(len(classifier_data[x])):
      for row in classifier_data[x, y, 0]:
        classifier_X.append(torch.Tensor(row))
        classifier_y.append(torch.Tensor([x]).long())

      for row in classifier_data[x, y, 1]:
        classifier_X.append(torch.Tensor(row))
        classifier_y.append(torch.Tensor([x]).long())

  for x in range(len(authenticator_data)):
    for y in range(len(authenticator_data[x])):
      for row in authenticator_data[x, y, 0]:
        authenticator_session_1_X.append(torch.Tensor(row))
        authenticator_session_1_y.append(torch.Tensor([x]).long())

      for row in authenticator_data[x, y, 1]:
        authenticator_session_2_X.append(torch.Tensor(row))
        authenticator_session_2_y.append(torch.Tensor([x]).long())

  return torch.stack(classifier_X).unsqueeze(1), F.one_hot(torch.stack(classifier_y)).squeeze().float(), authenticator_data


def train(model, X_train, y_train, X_val, y_val, n_epochs = 10, learn_rate=LEARNING_RATE, batch_size=BATCH_SIZE, patience_window=PATIENCE_WINDOW, class_weights=None, print_statements=True, doPlot=False):
  optimizer = optim.Adam(model.parameters(), lr=learn_rate)

  if class_weights is None: criterion = nn.CrossEntropyLoss()
  else: criterion = nn.CrossEntropyLoss(weight=class_weights)

  avg_train_loss_epoch = []
  avg_val_loss_epoch = []
  train_acc_epoch = []
  val_acc_epoch = []
  epochs = list(range(n_epochs))
  train_indices = np.arange(len(X_train)) 
  val_indices = np.arange(len(X_val)) 

  num_fails = 0
  
  for epoch in epochs:
    avg_train_loss = 0 # (averaged) training loss per batch
    avg_val_loss =  0  # (averaged) validation loss per batch
    train_acc = 0      # training accuracy per batch
    val_acc = 0        # validation accuracy per batch
    
    np.random.shuffle(train_indices)
    for it in range(0, X_train.shape[0], batch_size):
      model.train()
      batch = np.random.choice(train_indices, size=batch_size, replace=False)
      preds = model(X_train[batch])
      loss = criterion(preds, y_train[batch])
      avg_train_loss += loss.item() * len(batch)

      loss.backward()               
      optimizer.step()              
      optimizer.zero_grad()
    
    train_acc_epoch.append(get_accuracy(model, X_train, y_train)*100)
    avg_train_loss_epoch.append(avg_train_loss/X_train.shape[0])

    # run validation
      
    np.random.shuffle(val_indices)    
    for it in range(0, X_val.shape[0], batch_size):
      model.eval()
      batch = np.random.choice(val_indices, size=(batch_size if batch_size < len(val_indices) else len(val_indices)), replace=False)
      preds = model(X_val[batch])
      loss = criterion(preds, y_val[batch])
      avg_val_loss += loss.item() * len(batch)
      
    val_acc = get_accuracy(model, X_val, y_val)*100

    if len(val_acc_epoch) > patience_window and val_acc < max(val_acc_epoch):
      num_fails += 1
      if num_fails >= patience_window: 
        val_acc_epoch.append(val_acc)
        avg_val_loss_epoch.append(avg_val_loss/X_val.shape[0])
        if print_statements:
          print('Epoch %04d  Training Loss %.2f Validation Loss %.2f Training Accuracy %.2f Validation Accuracy %.2f' % (epoch + 1, avg_train_loss/X_train.shape[0], avg_val_loss/X_val.shape[0], train_acc_epoch[-1], val_acc_epoch[-1]))
        break
    else:
      num_fails = 0

    val_acc_epoch.append(val_acc)
    avg_val_loss_epoch.append(avg_val_loss/X_val.shape[0])

    if print_statements:
      print('Epoch %04d  Training Loss %.2f Validation Loss %.2f Training Accuracy %.2f Validation Accuracy %.2f' % (epoch + 1, avg_train_loss/X_train.shape[0], avg_val_loss/X_val.shape[0], train_acc_epoch[-1], val_acc_epoch[-1]))
  
    #Plot training loss
    if doPlot:
      plt.title("Train vs Validation Loss")
      plt.plot(avg_train_loss_epoch, label="Train")
      plt.plot(avg_val_loss_epoch, label="Validation")
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.legend(loc='best')
      plt.show()

      plt.title("Train vs Validation Accuracy")
      plt.plot(train_acc_epoch, label="Train")
      plt.plot(val_acc_epoch, label="Validation")
      plt.xlabel("Epoch")
      plt.ylabel("Accuracy (%)")
      plt.legend(loc='best')
      plt.show()

  print("Best Validation Accuracy is  {} at Epoch {}.".format(max(val_acc_epoch), val_acc_epoch.index(max(val_acc_epoch)) + 1))
  return train_acc_epoch[-1], val_acc_epoch[-1]

  
class Classifier(nn.Module):
  def __init__(self, num_filters = [40, 40], filter_sizes = [30, 50], pool_size = 4, dropout = 0.5, hidden_units = 60, num_classes = 100):
    super(Classifier, self).__init__()

    # may be good idea to use Sigmoid or tanh instead of ReLU for signal processing networks
    self.conv1 = nn.Sequential(nn.Conv1d(1, num_filters[0], filter_sizes[0]), nn.LeakyReLU(), nn.MaxPool1d(pool_size), nn.Dropout(p=dropout))
    self.conv2 = nn.Sequential(nn.Conv1d(num_filters[0], num_filters[1], filter_sizes[1]), nn.LeakyReLU(), nn.MaxPool1d(pool_size), nn.Dropout(p=dropout))
    self.lstm = nn.LSTM(input_size=num_filters[1], hidden_size=hidden_units, num_layers=3, batch_first=True)
    self.classifier = nn.Sequential(nn.Linear(48*60, 5*num_classes), nn.LeakyReLU(), nn.Linear(5*num_classes, num_classes))
    self.output_size = num_classes

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x).transpose(1, 2)
    x, (h_n, c_n) = self.lstm(x)
    x = x.reshape(-1, 48*60)
    x = self.classifier(x)
    return x # since nn.CrossEntropyLoss does the softmax part there

class Authenticator(nn.Module):
  def __init__(self, trained_classifier, freeze_classifier = True):
    super(Authenticator, self).__init__()

    # may be good idea to use Sigmoid or tanh instead of ReLU for signal processing networks
    self.classifier = deepcopy(trained_classifier)
    self.authenticator = nn.Sequential(nn.Softmax(dim=1), nn.Linear(self.classifier.output_size, 2))

    if freeze_classifier:
      for param in self.classifier.parameters():
        param.requires_grad = False

  def forward(self, x):
    x = self.classifier(x)
    x = self.authenticator(x)
    return x

def build_dataset_for_participant(authenticator_data, participant):
  authenticator_train_X = []
  authenticator_train_y = []
  authenticator_test_X = []
  authenticator_test_y = []

  counts = [0, 0]

  for x in range(len(authenticator_data)):
    for y in range(len(authenticator_data[x])):
      for row in authenticator_data[x, y, 0]:
        authenticator_train_X.append(torch.Tensor(row))
        if x == participant: 
          authenticator_train_y.append(torch.Tensor([1]).long())
          counts[1] += 1
        else: 
          authenticator_train_y.append(torch.Tensor([0]).long())
          counts[0] += 1

      for row in authenticator_data[x, y, 1]:
        authenticator_test_X.append(torch.Tensor(row))
        if x == participant: authenticator_test_y.append(torch.Tensor([1]).long())
        else: authenticator_test_y.append(torch.Tensor([0]).long())

  # class_weights = [1/count for count in counts]
  class_weights = [(sum(counts)/(2*count)) for count in counts]
  # class_weights = [1, 100]

  return torch.stack(authenticator_train_X).unsqueeze(1), F.one_hot(torch.stack(authenticator_train_y)).squeeze().float(), torch.stack(authenticator_test_X).unsqueeze(1), F.one_hot(torch.stack(authenticator_test_y)).squeeze().float(), torch.Tensor(class_weights)

def test_real_time_authentication(authenticator_cls, trained_classifier_model, authenticator_data,participant,n_epochs = 50, freeze_classifier=True):
  final_train_accs = []
  final_val_accs = []
  test_accs = []

  confusion_matrices = []
  X_train, y_train, X_test, y_test, class_weights = build_dataset_for_participant(authenticator_data, participant)
  print("Class Weights: {}".format(class_weights))

  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=TEST_TRAIN_SPLIT, random_state=RANDOM_SEED)

  model = authenticator_cls(trained_classifier_model, freeze_classifier=freeze_classifier)

  last_train_acc, last_val_acc = train(model, X_train, y_train, X_val, y_val, n_epochs=n_epochs, class_weights=class_weights, print_statements=False)

  model.eval()
  test_acc = get_accuracy(model, X_test, y_test)*100
  
  final_train_accs.append(last_train_acc)
  final_val_accs.append(last_val_acc)
  test_accs.append(test_acc)

  CM = confusion_matrix(y_test.argmax(dim=1).cpu(), model(X_test).argmax(dim=1).cpu(), labels=[0,1])
  print(CM)
  confusion_matrices.append(CM)
  
  return final_train_accs, final_val_accs, test_accs, confusion_matrices, model

def train_model_for_participant(Authenticator, classifier,authenticator_data,cleaned_ppg_data,participant,freeze_classifier=False):
    
    '''
    Authenticator: The Authenticator Class
    classifier: The trained classifier model
    authenticator_date: The Authenticator Dataset
    participant_data_path: The path to the Cleaned PPG Signal of Participant
    participant: An ID to assign to the participant
    freeze_classifier: Default is False
    '''

    count = 0
    # Add Participant's Data to Authenticator Dataset
    for i in range(len(authenticator_data[participant])):
        for j in range(len(authenticator_data[participant][i])):
            for k in range(len(authenticator_data[participant][i][j])):
                authenticator_data[participant][i][j][k] = cleaned_ppg_data[count]
                count = (count + 1) % 9

    
    

    train_accs, val_accs, test_accs, confusion_matrices , model= test_real_time_authentication(Authenticator, classifier, authenticator_data, participant, freeze_classifier=False)

    mean = lambda x: sum(x)/len(x)

    print("Average final train accuracy: {}".format(mean(train_accs)))
    print("Average final validation accuracy: {}".format(mean(val_accs)))
    print("Average test accuracy: {}".format(mean(test_accs)))
    print(sum(confusion_matrices))


    return train_accs, val_accs, test_accs, confusion_matrices , model

def prediction(data, model):
  model.eval()
  preds = model(torch.Tensor(data).reshape(data.shape[0],1,-1))
  return preds
