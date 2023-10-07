import matplotlib.pyplot as plt
#%matplotlib inline
import torchvision.transforms as T
import torch
import torch.nn as nn
from torch.nn import functional as F
import random

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1e-5

        # Reshape the predicted output tensor to [batch_size * height * width, num_classes]
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        pred = F.softmax(pred,dim=1)
        # Reshape the tensor back to [batch_size, 3, height, width]
        pred = pred.view(-1, 400, 600, 3).permute(0, 3, 1, 2)

        target = torch.cat([ (target == i) for i in range(1,4) ], dim=1)

        target = target.view(-1)
        pred = pred.reshape(-1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

       
        dice = (2. * intersection + smooth) / (union + smooth)
      
        return torch.mean(1 - dice)

def show_sample_images(train_dataset):

    idx = random.randint(0, 500)
    fig, arr = plt.subplots(1, 2, figsize=(10, 10))
    arr[0].imshow(train_dataset[idx][0])
    arr[0].set_title('Image '+ str(1))
    arr[1].imshow(train_dataset[idx][1])
    arr[1].set_title('Masked Image '+ str(1))


# define plot losses
def plot_losses(train_losses, test_losses, title):
  #%matplotlib inline
  fig, axs = plt.subplots(1,2,figsize=(9,3))
  axs[0].plot(train_losses)
  axs[0].set_title("Training Loss", fontsize=12).set_position([.5, 0.8])
  axs[1].plot(test_losses)
  axs[1].set_title("Test Loss", fontsize=12).set_position([.5, 0.8])
  fig.suptitle(title, fontsize=16, x=0.5, y=1.10)


def show_image_results(model, test_dataloader):
    model.eval()
    model.to('cpu')
    transform = T.ToPILImage()
    (test_inputs, test_targets) = next(iter(test_dataloader))
    fig, arr = plt.subplots(6, 3, figsize=(15, 15)) # batch size 16

    for index in range(6):
      img = test_inputs[index].unsqueeze(0)
      pred_y = model(img)
      pred_y = nn.Softmax(dim=1)(pred_y)
      pred_mask = pred_y.argmax(dim=1)
      pred_mask = pred_mask.unsqueeze(1).to(torch.float)


      arr[index,0].imshow(transform(test_inputs[index]))
      arr[index,0].set_title('Processed Image')
      arr[index,1].imshow(transform(test_targets[index].float()))
      arr[index,1].set_title('Actual Masked Image ')
      arr[index,2].imshow(pred_mask.squeeze(0).squeeze(0))
      arr[index,2].set_title('Predicted Masked Image ')

def VisualizeResults(test_dataloader, litmodel, batch_size):

    #test_dataloader = datamodule.test_dataloader()
    for batch in test_dataloader:
        images, masks = batch
        break  # Break to get the first batch (a batch of images and labels)

    for index in range(6):
        # Select one image and its corresponding label
        index = random.randint(0, batch_size)
        image = images[index]
        mask = masks[index]*255

        with torch.no_grad():
            input_image = image.to("cuda").unsqueeze(0)
            pred_mask_prob = litmodel.model(input_image)

        # From the probability of predictions for 3 classes, 
        # construct the actual prediction of values - 1,2,3
        pred_mask = torch.argmax(pred_mask_prob, dim=1) + 1
        pred_mask = pred_mask.squeeze().cpu()

        mask = mask.squeeze()
        img_np  = np.transpose(image.numpy(), (1, 2, 0))
        fig, arr = plt.subplots(1, 3, figsize=(15, 15))
        arr[0].imshow(img_np)
        arr[0].set_title('Image ')
        arr[1].imshow(mask)
        arr[1].set_title('Actual Mask')
        arr[2].imshow(pred_mask)
        arr[2].set_title('Predicted Mask')     