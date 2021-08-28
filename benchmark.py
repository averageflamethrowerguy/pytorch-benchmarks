import torch
import torch.nn as nn
import datetime
import sys
sys.path.append('../')

from utilities import run
from utilities import Dataset
from utilities import data_loader
from models import MLP

BATCH_SIZE=8184
max_epochs = 5000
number_conv_steps =32
LOOKBACK_DISTANCE=256
PREDICTION_RANGE=1
RUN_TIME=300000
SAVE_FILE_AT='./finished_models/32_layer_resnet.pt'

train_generator, test_generator, NUM_FEATURES = data_loader.load_csv(
     '../Intraday-Stock-Data/a-f/A_2010_2019.txt', 
     LOOKBACK_DISTANCE, 
     PREDICTION_RANGE,
     BATCH_SIZE,
     torch.float32
)

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

model.cuda().to(torch.float32)
model.apply(init_weights)

loss_fn = torch.nn.MSELoss(size_average=True).cuda()
LEARNING_RATE = 0.000001
         
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

run.run(model, optimizer, loss_fn, max_epochs, train_generator, test_generator, writer, RUN_TIME, WILL_CHECK_TIMINGS=True, USE_AUTOCAST=True, SAVE_FILE_AT=SAVE_FILE_AT)
