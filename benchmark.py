import torch
import torch.nn as nn
import datetime
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import Dataset
import MLP

writer = SummaryWriter()

BATCH_SIZE=8092
max_epochs = 10000
number_conv_steps =32
LOOKBACK_DISTANCE=256
PREDICTION_RANGE=1

torch.backends.cudnn.benchmark = True

# read the CSV
df = pd.read_csv('../Intraday-Stock-Data/a-f/A_2010_2019.txt', parse_dates=False, na_values=0)
df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
print("Dataframe length: " + str(len(df)))

# function to get the number of minutes from the open
def get_minutes_from_open(time):
    hour_min_sec = time.split(' ')[1]
    minutes = int(hour_min_sec.split(':')[1]) + 60*int(hour_min_sec.split(':')[0]) - 570
    return minutes / 195 - 1

df['time'] = df['time'].apply(get_minutes_from_open)
df.drop(df[df['time'] > 1].index, inplace=True)
df.drop(df[df['time'] < -1].index, inplace=True)
df.fillna(0)
print("Dataframe length adjusted: " + str(len(df)))

df['1_min_increase'] = df.open.pct_change()
df['1_min_volume_increase'] = df.volume.pct_change()

# delete the first 1 row
df = df.iloc[1:]
df.fillna(0)

df = df.drop(labels=['open', 'high', 'low', 'close', 'volume'], axis=1)

NUM_FEATURES=len(df.columns)

print('Number of features: ' + str(NUM_FEATURES))

yval_tensor = torch.tensor(df['1_min_increase'][LOOKBACK_DISTANCE + PREDICTION_RANGE:].values)
print('Length of yvals: ' + str(len(yval_tensor)))

# Splits into train and test data
train_set_size = int(len(yval_tensor) * 0.8)
yval_train = yval_tensor[:train_set_size]
yval_test = yval_tensor[train_set_size:]

xval_tensor = torch.tensor(df.values)
xval_tensor[np.isnan(xval_tensor)] = 0  # takes care of 8 nan values that slipped through
xval_train = xval_tensor[:train_set_size + LOOKBACK_DISTANCE]
xval_test = xval_tensor[train_set_size:]

dtype = torch.float16

train_set = Dataset.Dataset(yval_train.cuda().to(torch.float32), xval_train.cuda().to(dtype), LOOKBACK_DISTANCE)
test_set = Dataset.Dataset(yval_test.cuda(), xval_test.cuda().to(dtype), LOOKBACK_DISTANCE)

params = {'batch_size': BATCH_SIZE, 'shuffle': False}

train_generator = torch.utils.data.DataLoader(train_set, **params)
test_generator = torch.utils.data.DataLoader(test_set, **params)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)
        if (m.bias != None):
            m.bias.data.fill_(0.01)
    elif (isinstance(m, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

model = MLP.MLP(
    NUM_FEATURES=NUM_FEATURES, 
    LOOKBACK_DISTANCE=LOOKBACK_DISTANCE, 
    output_dim=1,
    number_conv_steps=number_conv_steps
)

model.cuda().to(dtype)
model.apply(init_weights)

loss_fn = torch.nn.MSELoss(size_average=True).cuda()
LEARNING_RATE = 0.000001
         
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

scaler = GradScaler()

torch.cuda.empty_cache()

iteration = 0
for epoch in range(max_epochs):

    train_loss = 0
    test_loss = 0

    for local_batch, local_labels in train_generator:
        optimizer.zero_grad()
        with autocast():
            local_labels_pred = model(local_batch)
            train_loss = loss_fn(local_labels_pred, local_labels)
        del local_batch, local_labels, local_labels_pred
        # Backward pass (FP 32 or native FP 16)
        # train_loss.backward()
        # optimizer.step()        

        # Backward pass (FP 16 autocast)
        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #Validation ...
        if (iteration % 10 == 0):
            with torch.set_grad_enabled(False):
                for local_batch, local_labels in test_generator:
                     local_batch, local_labels = local_batch, local_labels
                     local_labels_pred = model(local_batch)
                     test_loss = loss_fn(local_labels_pred, local_labels)
                     del local_batch, local_labels, local_labels_pred
                     break
                writer.add_scalar('Loss/train', train_loss, iteration)
                writer.add_scalar('Loss/test', test_loss, iteration)
                print (iteration)
        iteration += 1

torch.cuda.empty_cache
