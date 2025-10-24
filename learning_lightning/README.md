## Summary from what I learnt from train_a_model.py 

HOOKS = Model-SPECIFIC logic (THIS model only)

CALLBACKS = Model-AGNOSTIC features (ANY model)

# Learnings from train_a_model.py

- Did not use any AI in writing the code in this file, wrote the whole code by self.

## Concepts learned from writing the code train_a_model.py, the leanrings below are solely after writing the code in train_a_model.py only. 

- how an auto encoder model works (basic level), trained on MNIST dataset
- About Lightning workflow - Dataloaders was created using torch.data.utils.DataLoader class and passed in arguments inside trainer of Lightning, lightning handles the loops, and boiler plate code, repetative stuff involved in training a model which is for every epoch and inside every epoch for every batch (step), particularly telling about the workflow involved in autoencoder so getting the output from the encoder, ignoring labels involved (if any) because in autoencoder they are not of use, then passing the output of encoder which is called latent vector to decoder and then calculating the loss between X and output given by decoder, for every datapoint in that batch.
- So it was lightning which handled every step processing involved in a epoch for batches namely - The loop involved for processing every batch of dataloader (loss calculation i.e training_step, val_step ,configure_optimizer etc are handled by LightningModule), loss.backward() (handled by Trainer class) for that batch, optimizer.step() (handled by Trainer Class) for that batch, optimizer.zero_grad() (handled by Trainer Class) for that batch
- Trainer handles the above namely things and loss calculation for every batch for trainer is handled by hook/method from LightningModule which is def training_step , from here we the whole process of giving the input to encoder then latent vector then to decoder happens and loss is calculated for the batch.
- The training_step method of LightningModule is applied on every batch which trainer calls 
- The input to training_step is batch, batch_idx and output or return is loss for that batch
- The model was instantiated by nn of torch, dataloaders was used from torch, Lightning module is used for training step (loss calculation), optimizer and trainer from lightning L.Trainer is used to remove boiler plate code of training loops, loss.backward() etc etc.
- When the LitAutoEnocder was instantiated, the input to this class is given the argument accepeted by the def __init__ , here in this case it accepts the encoder and decoder class that is model in the input argument for LitAutoEncoder (L.LightningModule)

If we use the Lightning module for training then we pass in the model architecture either inside the Lightning module or if complex can define somewhere else and then pass the instance of the architecture to the Lightning module, and the basic essentials needed to define in Lightning module like training_step and forward method. Dataloaders are defined from torch and then in trainer.fit we pass in the instance of this Lightning module and dataloaders. The lightning module contains the code for training loop, model architecture, 


**Summary:**

The dataloader of PyTorch accepts as argument -> The instance of the Dataset Object, or (A class which has getitem and length functions inside it), which mostly would be PyTorch Dataset object (inherited from PyTorch Dataset object). That instance of the Dataset object must have these 3 methods inside them - __init__, getitem, len while defining the custom Dataset (there may be some scenario where one may not need to define custom Dataset like when we can use ImageFolder which assumes that directory structure is already there etc etc), but for specific use cases we need to define our Dataset class, in which most important is the getitem method inside it because it defines how the loading for one sample and label will happen.

init function -> While instantiating the Dataset class before dataloader, the arguments need to be passed in the instance will be the same as the init method will ask for and it is like the data directory path and stuff.

length (len) -> it returns the length of the total data points in the data directory, that is the count of all data points which are possible (counting each unique only once) from the data directory we have.

getitem - The most important, it loads one pair of data point, label. The logic of loading and labelling is happening from here that's why it's most important.

The dataloader needs a Dataset object in the argument which points towards a data directory from which the mini batches of data points, labels get created by the logic written inside the getitem of the Dataset Object and data directory which init provides. For creating dataloader for an epoch the getitem of the Dataset gets called multiple times (batch size times, this call can be made parallel) for creating one batch of that data and that's how dataloaders are created each of batch size, inside each batch size there are batch size number of data points, labels are there in one batch and total batches are length/batch size which are further used in training.


 