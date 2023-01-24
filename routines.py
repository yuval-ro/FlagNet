import torch

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
