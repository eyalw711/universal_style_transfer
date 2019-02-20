import os
import argparse
import random
import datetime
from PIL import Image

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from seq_models import Decoder, Encoder
from ref_models_factory import Encoder as R_Encoder
from ref_models_factory import Decoder as R_Decoder


LR = 1e-3                 # learning rate
BETA1 = 0.5               # beta1 for Adam arguments
BETA2 = 0.9               # beta2 for Adam arguments
PRINT_INTERVAL = 5
SAMPLE_INTERVAL = 5
SAVE_INTERVAL = 50


torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda")

def models_loss_experiment():
  print('Start model loss experiment')
  tforms = transforms.Compose([transforms.Resize((1024,1024)), transforms.ToTensor()])
  dataset = torchvision.datasets.ImageFolder('data', transform=tforms)
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
  crit = nn.MSELoss()
  
  encoders = [Encoder(j).to(device) for j in range(1,6)]
  decoders = [Decoder(j).to(device) for j in range(1,6)]
  loss = [0 for j in range(1,6)]
  floss = [0 for j in range(1,6)]
  
  ref_encoders = [R_Encoder(j).to(device) for j in range(1,6)]
  ref_decoders = [R_Decoder(j).to(device) for j in range(1,6)]
  ref_loss = [0 for j in range(1,6)]
  ref_floss = [0 for j in range(1,6)]
  
  num_batches = 0
  
  with torch.no_grad():
    for (x,_) in data_loader:
      x = x.to(device)
      num_batches += 1
      xhats = [d(e(x)) for (e,d) in zip(encoders, ref_decoders)]
      ref_xhats = [d(e(x)) for (e,d) in zip(ref_encoders, ref_decoders)]
      
      for i in range(5):
        e = encoders[i]
        loss[i] += crit(xhats[i], x).item()        # pixel loss
        floss[i] += crit(e(xhats[i]), e(x)).item() # feature loss
          
        ref_e = ref_encoders[i]
        ref_loss[i] += crit(ref_xhats[i], x).item()                # pixel loss
        ref_floss[i] += crit(ref_e(ref_xhats[i]), ref_e(x)).item() # feature loss

      print('Batchs done: {}'.format(num_batches))
        
      if num_batches == 1000:
        break
  
    loss = [z/num_batches for z in loss]
    floss = [z/num_batches for z in floss]
    ref_loss = [z/num_batches for z in ref_loss]
    ref_floss = [z/num_batches for z in ref_floss]
    
    print('loss: {}'.format(loss))
    print('floss: {}'.format(floss))
    print('ref_loss: {}'.format(ref_loss))
    print('ref_floss: {}'.format(ref_floss))
    
    
def train_decoder(narch, load, nepochs, lmbda, winit_method):
  LAMBDA = lmbda
  tforms = transforms.Compose([transforms.Resize((1024,1024)), transforms.ToTensor()])
  dataset = torchvision.datasets.ImageFolder('data', transform=tforms)
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
  print('dataloader ready')
  
  crit = nn.MSELoss()#reduction='sum')
  
  encoder = Encoder(narch).to(device)
  for param in encoder.parameters():
    param.requires_grad = False
      
  decoder = Decoder(narch, weight_init_method=winit_method).to(device)
  opt = torch.optim.Adam(decoder.parameters(), lr=LR, betas=(BETA1, BETA2))
  
  if load:
    decoder.load_state_dict(torch.load('models/ust_decoder_{}.pth'.format(narch)))
    opt.load_state_dict(torch.load('models/ust_decoder_{}_opt.pth'.format(narch)))
    print('loaded models and optimizer')
    
  print('models and optimizer ready')
  print('decoder arch is: {}'.format(decoder))
  
  try:
    tmploss = 0
    
    for ep in range(nepochs):
      for i, (x,_) in enumerate(data_loader):
        x = x.float().to(device)
        f = encoder(x)
        
        opt.zero_grad()
        xhat = decoder(f)
        assert xhat.size() == x.size(), "xhat.size()={}, x.size()={}".format(xhat.size(), x.size())
        
        pixloss = crit(xhat, x)
        featloss = LAMBDA*crit(encoder(xhat), f)
        
        loss =  pixloss + featloss
        loss.backward()
        opt.step()
        
        tmploss += loss.item()
        if i % PRINT_INTERVAL == 0:
          mn = tmploss/PRINT_INTERVAL
          print('pixloss:{} featloss:{}'.format(pixloss, featloss))
          print('====> E: {}| batch: {}/{} | loss = {}'.format(ep, i, len(data_loader), mn))
          tmploss = 0
          
        if i % SAMPLE_INTERVAL == 0:
          img_file_path = "results/dec_{}_train.png".format(narch)
          batch = torch.cat((x[0].unsqueeze(0), xhat[0].unsqueeze(0)), 0)
          save_image(batch.data, img_file_path, nrow=2, normalize=True)
          
        if i % SAVE_INTERVAL == SAVE_INTERVAL-1:
          model_path = "models/ust_decoder_{}.pth".format(narch)
          torch.save(decoder.state_dict(), model_path)
          
          opt_path = "models/ust_decoder_{}_opt.pth".format(narch)
          torch.save(opt.state_dict(), opt_path)
  
  except KeyboardInterrupt:
    pass
  
  model_path = "models/ust_decoder_{}.pth".format(narch)
  torch.save(decoder.state_dict(), model_path)
  
  opt_path = "models/ust_decoder_{}_opt.pth".format(narch)
  torch.save(opt.state_dict(), opt_path)
  print('training complete')

def main():
  tm = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
  
  parser = argparse.ArgumentParser(description='Models Module, for generating models and training')
  parser.add_argument('--train', type=int, default=False, help='Flag to train the a decoder')
  parser.add_argument('--load', default=False, action='store_true', help='flag to continue training saved models')
  parser.add_argument('--epochs', default=1, type=int, help='number of training epochs')
  parser.add_argument('--lmbda', default=1.0, type=float, help='lambda weight for loss function')
  parser.add_argument('--pretrained', default=False, action='store_true', help='use trained blocks')
  parser.add_argument('--loss-exper', default=False, action='store_true', help='run model loss experiment')
  args = parser.parse_args()
  
  if args.train:
    print("Training architecture {}...".format(args.train))
    train_decoder(args.train, args.load, args.epochs, args.lmbda, 'blocks' if args.pretrained else 'random')
    
  if args.loss_exper:
    models_loss_experiment()
    
if __name__ == "__main__":
    main()