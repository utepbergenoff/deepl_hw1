import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes,output_size, activation = 'relu'):
        super().__init__()
        sizes = [input_size] + hidden_sizes + [output_size] 
        self.layers = nn.ModuleList()

        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                
        return x

if __name__ == "__main__":
    m = MLP(784, [128, 64], 10)
    dummy = torch.randn(32, 1, 28, 28)  # batch of 32 fake MNIST images
    out = m(dummy)
    print(out.shape)