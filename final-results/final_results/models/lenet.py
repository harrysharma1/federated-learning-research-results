import torch.nn as nn 

class LeNet(nn.Module):
    """
    Classical LeNet Neural Network as used in DLG example.
    
    Args:
        nn (nn.Module): Using inheritance from parent to create and extend Neural Network.
    """
    
    def __init__(self, activation_function=nn.Sigmoid):
        super(LeNet, self).__init__()
        act = activation_function
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(768, 100)
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
class ModelHelper():
    """
    Helper Class for functions to use with Neural Networks
    """
    
    def __init__(self):
        pass
    
    def weight_init(self, m):
        if hasattr(m, 'weight'):
            m.weight.data.uniform_(-0.5,0.5)
        if hasattr(m, 'bias'):
            m.bias.data.uniform_(-0.5, 0.5)