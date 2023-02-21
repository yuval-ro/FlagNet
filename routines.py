# vanilla:
import os
import time
# external:
import torch
import torchvision
# custom:
import consts


''' Routines for Model Training  '''


# Training sequence for a single batch.
#  Returns the calculated loss for metadata collection:
def train(hyperparams, inputs, labels):
    model, optimizer, device, criterion = hyperparams
    # Cast inputs and labels Tensors to hardware:
    inputs, labels = inputs.to(device), labels.to(device)
    # Zero the layer's gradient:
    optimizer.zero_grad()
    # Get the probability matrix for this batch:
    log_probabilities = model.forward(inputs)
    # Calculate the loss of the probability matrix:
    loss = criterion(log_probabilities, labels)
    # Perform back-propagation with the calculated loss using the defined optimizer:
    loss.backward()
    optimizer.step()
    return loss.item()


# Testing sequence for a single batch, which also applies for validation.
#  Returns a tuple of loss and accuracy for metadata collection:
def test(hyperparams, inputs, labels):
    model, optimizer, device, criterion = hyperparams
    # Cast inputs and labels Tensors to hardware:
    inputs, labels = inputs.to(device), labels.to(device)
    # Get the probability matrix for this batch:
    log_probabilities = model.forward(inputs)
    # Calculate the loss of the probability matrix:
    loss = criterion(log_probabilities, labels)
    # Calculate prediction accuracy:
    probabilities = torch.exp(log_probabilities)
    top_probability, top_class = probabilities.topk(1, dim = 1)
    equals = (top_class == labels.view(*top_class.shape))
    acc = torch.mean(equals.type(torch.FloatTensor)).item()
    return (loss.item(), acc)


# Aggregating a single epoch's metadata:
def collect(list, metadata):
    idx, stime, etime, tloss, vloss, acc, loaders = metadata
    list.append([(idx + 1),                  # epoch index
                (etime - stime),             # epoch time in seconds
                (tloss / len(loaders[0])),   # epoch training loss
                (vloss / len(loaders[1])),   # epoch validation loss
                (acc / len(loaders[1]))])    # epoch accuracy


# Template for loading a .pth file checkpoint.
#  returns a model object:
def loadCheckpoint(file_path, weights):
    checkpoint = torch.load(file_path)
    model = getattr(torchvision.models, checkpoint['network'])(weights=weights)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


# Sorts all checkpoints in root dir by their filename,
#  which tells their creation date (ascending order).
# Returns the latest checkpoint path from root:
def latestCheckpoint(path=consts.checkpoints_path, format=consts.checkpoint_timestamp_format):
    checkpoints = []
    for file in os.listdir(path):
        if file.endswith('.pth'):
            filename, _ = os.path.splitext(file) # strips file extension from string
            filename_date_tuple = (time.strptime(filename, format), file)
            checkpoints.append(filename_date_tuple)
    if checkpoints == []:
        return None
    else:
        latest_checkpoint = (sorted(checkpoints, key=lambda x: x[0])[-1])[1]
        return path + '/' + latest_checkpoint # string - path to .pth file
