import torch


class Dataset(torch.utils.data.Dataset):
  def __init__(self, yvals, data_tensor, LOOKBACK_DISTANCE):
    self.yvals = yvals
    self.data_tensor = data_tensor
    self.LOOKBACK_DISTANCE = LOOKBACK_DISTANCE

  def __len__(self):
    return len(self.yvals)

  def __getitem__(self, index):
    X = torch.transpose(self.data_tensor[index: index + self.LOOKBACK_DISTANCE], 0, 1)
#    X = X.reshape([len(X), len(X[0]), 1])
    y = self.yvals[index]

    return X, y
