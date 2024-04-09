import torch
import torch.nn as nn
import torchvision.models as models

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        
        # Load the pre-trained ResNet-18 model
        self.model = models.densenet121(pretrained = True)
        
        self.model.classifier = nn.Linear(in_features=1024, out_features = 8, bias = True)
        #nn.init.xavier_uniform_(self.model.classifier.weight)
        
         
    def forward(self, x):
        
        return self.model(x)
    
    def save_model(self):
        # Saving the model's weights
        torch.save(self.state_dict(), 'model.pkl')


