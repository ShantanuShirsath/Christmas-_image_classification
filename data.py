from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import os

class ChristmasImages(Dataset):
    
    def __init__(self, path, training=True):
        super().__init__()
        # If training == True, path contains subfolders
        # containing images of the corresponding classes
        # If training == False, path directly contains
        # the test images for testing the classifier
        self.training = training
        self.path = path
        self.Transformations = transforms.Compose([(transforms.Resize((224,224))),
                                                   transforms.ToTensor()])
        if training:
            # Training data
            self.category_name = sorted(os.listdir(self.path))
            self.class_to_label = {categoryname: index for index, categoryname in enumerate(self.category_name)}

            self.file_path = []
            self.label = []

            for label, categoryname in enumerate(self.category_name):
                category_path = os.path.join(self.path, categoryname)
                for file_name in os.listdir(category_path):
                    file_path = os.path.join(category_path, file_name)
                    self.file_path.append(file_path)
                    self.label.append(label)

        else:
            # For testing data
            self.file_path = []
            self.file_name = []
            for file_name in os.listdir(self.path):
                file_path = os.path.join(self.path, file_name)
                self.file_path.append(file_path)
                self.file_name.append(file_name)

        
    def __getitem__(self, index):
        # If self.training == False, output (image, )
        # where image will be used as input for your model
        if self.training:
            img_path = self.file_path[index]
            label = self.label[index]

            # Load Image
            image = Image.open(img_path).convert('RGB')
            image = self.Transformations(image)
            return image, label
        else:
            img_path = self.file_path[index]
            image = Image.open(img_path).convert('RGB')
            image = self.Transformations(image)
            return image
        
        raise NotImplementedError
    
    def __len__(self):
        return len(self.file_path)
    
        
