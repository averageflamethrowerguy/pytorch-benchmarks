import torch
import torch.nn as nn

output_dim = 1
scan_period = 7
depth_multiplier =6

class ResChunk(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=scan_period, padding=tuple([scan_period // 2]))
    self.act1 = nn.LeakyReLU()

  def forward(self, X):
    residual = X
    X = self.conv1(X)
    X = self.act1(X)
    return X




class MLP(nn.Module):
  def __init__(self, NUM_FEATURES, LOOKBACK_DISTANCE, output_dim, number_conv_steps):
    super(MLP, self).__init__()

    self.upscale = nn.Sequential(
      nn.Conv1d(in_channels=NUM_FEATURES, out_channels=depth_multiplier*NUM_FEATURES, kernel_size=(1)),
      nn.Tanh()
    )
    self.convolutions = nn.ModuleList()
    for i in range(number_conv_steps):
      self.convolutions.append(ResChunk(in_channels=depth_multiplier*NUM_FEATURES, out_channels=depth_multiplier*NUM_FEATURES))

    self.downscale = nn.Sequential(
      nn.Conv1d(in_channels=depth_multiplier*NUM_FEATURES, out_channels=1, kernel_size=(1)),
      nn.Tanh()
    )
    
    self.output = nn.Sequential(
      nn.Linear(LOOKBACK_DISTANCE, output_dim),
      nn.Tanh()
    )

  def forward(self, X):
    X = self.upscale(X)
    for operation in self.convolutions:
      X = operation(X)
    X = self.downscale(X)
    X = self.output(X)
    return X


