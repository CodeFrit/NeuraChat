import torch.nn as nn

class FFModel(nn.Module):
    def __init__(self, num_inputs,num_class,hidden_size=10):
        super(FFModel,self).__init__()
        self.l1 = nn.Linear(num_inputs,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size-2)
        self.l2h = nn.Linear(hidden_size-2,hidden_size-4)
        self.l3 = nn.Linear(hidden_size-4,num_class)
        self.act = nn.ReLU()
        self.acth = nn.LeakyReLU()

    def forward(self,x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.acth(self.l2h(x))
        x = self.l3(x)
        return x #Using CEL