import torch
import torchvision
import pytorch_lightning as pl
    
from torchvision import transforms    
    
class OxfordIIITPetsData(pl.LightningDataModule):
    def __init__(self, train_data_path='data\\OxfordPets\\train', 
                 test_data_path='data\\OxfordPets\\test', batch_size=16):
        super().__init__()       
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size

    def prepare_data(self):
        # Create a torchvision transform to resize the image
        target_size = (400,600)
        transform_resize_tensor = transforms.Compose([
        transforms.Resize(target_size),
        # This converts the PIL Image to a PyTorch tensor
        transforms.ToTensor()  
        ])

        # Oxford IIIT Pets Segmentation dataset loaded via torchvision.       
        self.train_data = torchvision.datasets.OxfordIIITPet(root=self.train_data_path, split="trainval", 
                                                             target_types="segmentation", download=True, 
                                                             transform=transform_resize_tensor, 
                                                             target_transform=transform_resize_tensor)
        self.test_data = torchvision.datasets.OxfordIIITPet(root=self.test_data_path, split="test", 
                                                            target_types="segmentation", download=True, 
                                                            transform=transform_resize_tensor, 
                                                            target_transform = transform_resize_tensor)

    def setup(self, stage):
        pass
                 

    def train_dataloader(self):
        return  torch.utils.data.DataLoader(
            self.train_data,
            batch_size= self.batch_size,
            shuffle=True,
            )
    
    def test_dataloader(self):
         return torch.utils.data.DataLoader(
            self.test_data,
            batch_size= self.batch_size,
            shuffle=False,
            )
    
    def val_dataloader(self):
         return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=True,
            )
