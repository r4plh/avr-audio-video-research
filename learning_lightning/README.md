HOOKS = Model-SPECIFIC logic (THIS model only)

CALLBACKS = Model-AGNOSTIC features (ANY model)

# Learnings from train_a_model.py

- Did not use any AI in writing the code in this file, wrote the whole code by self.

## Concepts learned from writing the code train_a_model.py, the leanrings below are solely after writing the code in train_a_model.py only. 

- how an auto encoder model works (basic level), trained on MNIST dataset
- About Lightning workflow - Dataloaders was created using torch.data.utils.DataLoader class and passed in arguments inside trainer of Lightning, lightning handles the loops, and boiler plate code, repetative stuff involved in training a model which is for every epoch and inside every epoch for every batch (step), particularly telling about the workflow involved in autoencoder so getting the output from the encoder, ignoring labels involved (if any) because in autoencoder they are not of use, then passing the output of encoder which is called latent vector to decoder and then calculating the loss between X and output given by decoder, for every datapoint in that batch.
- So it was lightning which handled every step processing involved in a epoch for batches namely - The loop involved for processing every batch of dataloader (loss calculation i.e training_step, val_step ,configure_optimizer etc are handled by LightningModule ), loss.backward() (handled by Trainer class) for that batch, optimizer.step() (handled by Trainer Class) for that batch, optimizer.zero_grad() (handled by Trainer Class) for that batch
- Trained handles the above namely things and loss calculation for every batch for trainer is handled by hook/method from LightningModule which is def training_step , from here we the whole process of giving the input to encoder then latent vector then to decoder happens and loss is calculated for the batch.
- The training_step method of LightningModule is applied on every batch which tariner calls 
- The input to training_step is batch, batch_idx and output or return is loss for that batch
- The model was instantiated by nn of torch, dataloaders was used from torch, Lightning module is used for training step (loss calculation), optimizer and trainer from lightning L.Trainer is used to remove boiler plate code of training loops, loss.backward() etc etc.
- When the LitAutoEnocder was instantiated, the input to this class is given the argument accepeted by the def __init__ , here in this case it accepts the encoder and decoder class that is model in the input argument for LitAutoEncoder (L.LightningModule)


 