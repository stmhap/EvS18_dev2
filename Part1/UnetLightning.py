import pytorch_lightning as pl
import torch
from torch import nn
from unet import UNet
from loss import DiceLoss

class UnetLightning(pl.LightningModule):
    def __init__(self, loss_func = 'CE', contract_method = 'MP', expand_method = 'Tr'):
        super().__init__()

        self.model = UNet(3, 3, contract_method = 'MP', expand_method = 'Tr')
        self.loss_func = loss_func     

        self.train_losses = []
        self.save_hyperparameters()

        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

   
    def training_step(self, batch, batch_idx):
        input_imgs, target_masks = batch
        pred_mask_prob =  self.model(input_imgs) 
        target_masks = target_masks*255 #PIL Image open divides all values by 255

        if(self.loss_func == 'CE'): #Cross Entropy loss      
            
            # Reshape the predicted output tensor to [batch_size * height * width, num_classes]
            predicted_output = pred_mask_prob.permute(0, 2, 3, 1).contiguous().view(-1, 3)

            # cross entropy works based on indices, so convert classes 1,2,3 to 0,1,2
            target_categories = target_masks - 1

            # Squeeze the singleton dimension from the target tensor and convert to long tensor
            target_labels = torch.squeeze(target_categories, dim=1).view(-1).long()

            # Create the CrossEntropyLoss criterion
            criterion = torch.nn.CrossEntropyLoss()

            # Calculate the cross-entropy loss
            loss = criterion(predicted_output, target_labels)
            self.train_losses.append(loss.item())

            loss.backward(retain_graph=True)

            return loss
        
        else: #Dice loss

            criterion = DiceLoss()
            loss = criterion(pred_mask_prob, target_masks)
            self.train_losses.append(loss.item())
            
            loss.backward(retain_graph=True)

            return loss
                    
    
    def on_train_epoch_end(self):
        # print loss at the end of every epoch   
        mean_loss = sum(self.train_losses) / len(self.train_losses)
        print(f'Mean training loss at end of epoch {self.trainer.current_epoch} = {mean_loss}')
        
