import os
import argparse
import random
import datetime
import pickle
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

VER = 0.1

LR = 1e-3                 # learning rate
BETA1 = 0.5               # beta1 for Adam arguments
BETA2 = 0.9               # beta2 for Adam arguments
PRINT_INTERVAL = 5
SAMPLE_INTERVAL = 5
SAVE_INTERVAL = 50

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda")

class Session:
  def __init__(self, narch, lmbda, name=None):
    self.ver = 0.1
    self.epochs = 0
    self.arch = narch
    self.lmbda = lmbda
    
    if name is None:
      self.name = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    else:
      self.name = name


def Encoder(n):
  vgg = torchvision.models.vgg19(pretrained=True)
  # 'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
  #layers = [4,9,18,27,36]
  layers = [2, 7, 12, 21, 30]
  i = layers[n-1]
  
  layer_list = list(vgg.features.children())[:i]
  for i, layer in enumerate(layer_list):
    if isinstance(layer, nn.ReLU):
      layer_list[i] = nn.ReLU()
  model = nn.Sequential(*layer_list)
  return model


def decoder_block(n):
  assert 1 <= n <= 5 and isinstance(n, int), "Wrong n={}".format(n)
  
  in_chns = [64, 128, 256, 512, 512]
  out_chns = [3, 64, 128, 256, 512]
  num_convs = [1, 2, 2, 4, 4]
  layer_confs = list(zip(in_chns, out_chns, num_convs))
  
  layers = []
  inc, outc, numc = layer_confs[n-1]
  
  for i in range(numc):
    if i==1:
      if n < 4:
        layers.append(nn.UpsamplingNearest2d(scale_factor=2))
      else:
        layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
      
    if i==0:
      layers.append(nn.Conv2d(inc, outc, (3,3), padding=(1,1)))
    else:
      layers.append(nn.Conv2d(outc, outc, (3,3), padding=(1,1)))
    
    layers.append(nn.ReLU())
    
  if n==1: #exception where nn.ReLU shouldn't be last
    layers = layers[:-1]
  
  block = nn.Sequential(*layers)
  return block


class Decoder(nn.Module):
  RDY_DECODER_BLOCK_PATH = 'models/decoder_block_{}.pth'
  
  def __init__(self, n, init_weight=True):
    super(Decoder, self).__init__()
    blocks = [decoder_block(j) for j in reversed(range(1,n+1))]
    if init_weight:
      for decblock in blocks:
        initialize_weights(decblock)
    else:
      for i,decblock in enumerate(blocks):
        decblock.load_state_dict(torch.load(Decoder.RDY_DECODER_BLOCK_PATH.format(n-i)))
    self.model = nn.Sequential(*blocks)
    
  def forward(self, x):
    #for decblock in self.blocks:
    #  print("forward x_in size=", x.size())
    #  x = decblock(x)
    #print("x_out size =", x.size())
    #return x
    return self.model(x)
    
  #def __repr__(self):
  #  return '\n'.join([decblock.__repr__() for decblock in self.blocks])


def initialize_weights(module):
  ''' from official pytorch vgg code'''
  for m in module.modules():
    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)


def train_decoder(narch, session, nepochs, lmbda, norm):
  load = False
  
  #open session
  if session is None:
    sn = Session(narch, lmbda) # new session with default name
    os.makedirs(os.path.join('models', sn.name))
    print("New session: " + sn.name)
  else:
    try:
      sn = pickle.load(open("sessions/" + session + ".pkl", "rb")) # existing session
      load = True
    except (OSError, IOError) as e:
      sn = Session(narch, lmbda, name=session) # new session with non-default name
      os.makedirs(os.path.join('models', sn.name), exist_ok=True)
      print("New session: " + sn.name)

  assert sn.ver == VER, "Wrong session version! session is {} and script is {}".format(sn.ver, VER)
  
  narch = sn.arch
  
  print("<<< Session: {}, arch: {} >>>".format(sn.name, sn.arch))
  
  RDY_DECODER_BLOCK_PATH = 'models/decoder_block_{}.pth'
  DECODER_BLOCK_PATH = 'models/' + sn.name + '/decoder_block_{}.pth'
  DECODER_BLOCK_OPT_PATH = 'models/' + sn.name + '/decoder_block_{}_opt.pth'
  
  LAMBDA = sn.lmbda
  print("training with session lambda: {}".format(sn.lmbda))
  
  tforms = transforms.Compose([transforms.Resize((768,768)), transforms.ToTensor()])
  dataset = torchvision.datasets.ImageFolder('data', transform=tforms)
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
  print('dataloader ready')
  
  crit = nn.MSELoss()
  
  #freeze encoder
  encoder = Encoder(narch).to(device)
  for param in encoder.parameters():
    param.requires_grad = False

  dec = Decoder(narch-1, init_weight=False).to(device) if narch>1 else None
  print("Using decoder: {}".format(dec))
  if dec:
    #for i,block in enumerate(dec.blocks):
    #  dec.blocks = block.to(device)
    for param in dec.model.parameters(): # freeze decoder
      param.requires_grad = False
    #for block in dec.blocks:       # freeze decoder blocks
    #  for param in block.parameters():
    #    param.requires_grad = False

  b = decoder_block(narch).to(device)
  if load:
    b.load_state_dict(torch.load(DECODER_BLOCK_PATH.format(narch)))
    print('loaded block %d' % narch)
  else:
    initialize_weights(b)
    
  opt = torch.optim.Adam(b.parameters(), lr=LR, betas=(BETA1, BETA2))
  if load:
    opt.load_state_dict(torch.load(DECODER_BLOCK_OPT_PATH.format(narch)))
    print('loaded optimizer')
    
  print("Training block: {}".format(b))
  
  # v2 working
  #b2 = decoder_block(2).to(device)
  #initialize_weights(b2)
  #
  #b1 = decoder_block(1).to(device)
  #b1.load_state_dict(torch.load(RDY_DECODER_BLOCK_PATH.format(1)))
  #for param in b1.parameters():
  #    param.requires_grad = False
  #
  #opt = torch.optim.Adam(b2.parameters(), lr=LR, betas=(BETA1, BETA2))
  
  # v1 not working
  #decoder = Decoder(narch).to(device)
  #opt = torch.optim.Adam(decoder.blocks[0].parameters(), lr=LR, betas=(BETA1, BETA2))
  #
  ##load+freeze trained decoder blocks
  #for i, block in enumerate(decoder.blocks[::-1]): 
  #  #blocks in decoder are in descending order, going over them in ascending order
  #  btype = i + 1
  #  if btype == narch:
  #    if load:
  #      block.load_state_dict(torch.load(DECODER_BLOCK_PATH.format(btype)))
  #      opt.load_state_dict(torch.load(DECODER_BLOCK_OPT_PATH.format(btype)))
  #      print('loaded block in training and optimizer')
  #    continue
  #  
  #  block.load_state_dict(torch.load(RDY_DECODER_BLOCK_PATH.format(btype)))
  #  for param in block.parameters():
  #    param.requires_grad = False
    
  print('loaded and freezed blocks not in training')
  print('models and optimizer ready')
  
  try:
    tmploss = 0
    
    for ep in range(nepochs):
      for i, (x,_) in enumerate(data_loader):
        x = x.float().to(device)
        f = encoder(x)
        
        opt.zero_grad()
        
        #xhat = decoder(f) # ver1
        
        #f1 = b2(f)        # ver2
        #xhat = b1(f1)
        
        f1 = b(f)
        if dec:
          xhat = dec(f1)
        else:
          xhat = f1
        
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
          print('=======> E: {} | batch: {}/{} | loss = {}'.format(ep, i, len(data_loader), mn))
          tmploss = 0
          
        if i % SAMPLE_INTERVAL == 0:
          img_file_path = "results/{}_dec_{}_train.png".format(sn.name, narch)
          batch = torch.cat((x[0].unsqueeze(0), xhat[0].unsqueeze(0)), 0)
          save_image(batch.data, img_file_path, nrow=2, normalize=norm)
          
        if i % SAVE_INTERVAL == SAVE_INTERVAL-1:
          model_path = DECODER_BLOCK_PATH.format(narch)
          torch.save(b.state_dict(), model_path)
          
          opt_path = DECODER_BLOCK_OPT_PATH.format(narch)
          torch.save(opt.state_dict(), opt_path)
          
          pickle.dump(sn, open("sessions/" + session + ".pkl", "wb"))
      
      # EOE
      sn.epochs += 1
  
  except KeyboardInterrupt:
    pass
  
  model_path = DECODER_BLOCK_PATH.format(narch)
  torch.save(b.state_dict(), model_path)
  
  opt_path = DECODER_BLOCK_OPT_PATH.format(narch)
  torch.save(opt.state_dict(), opt_path)
  
  pickle.dump(sn, open("sessions/" + session + ".pkl", "wb"))
  
  print('training complete')


def main():
  tm = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
  
  parser = argparse.ArgumentParser(description='Models Module, for generating models and training')
  parser.add_argument('--train', type=int, default=False, help='Flag to train the a decoder')
  #parser.add_argument('--load', default=False, action='store_true', help='flag to continue training saved models') # to overwrite (not load) delete previous session
  parser.add_argument('--epochs', default=1, type=int, help='number of training epochs')
  parser.add_argument('--lmbda', default=1.0, type=float, help='lambda weight for loss function')
  parser.add_argument('--session', default=None, type=str, help='session name')
  parser.add_argument('--test', type=int, default=False, help='test architecture')
  parser.add_argument('--norm', default=False, action='store_true', help='normalize picture')
  args = parser.parse_args()
  
  if args.train:
    print("Training architecture {}...".format(args.train))
    train_decoder(args.train, args.session, args.epochs, args.lmbda, args.norm)
  
  if args.test:
    narch = args.test
    RDY_DECODER_BLOCK_PATH = 'models/decoder_block_{}.pth'
    encoder = Encoder(narch).to(device)
    decoder = Decoder(narch, init_weight=False).to(device)
    #load+freeze trained decoder blocks
    #for i, block in enumerate(decoder.blocks[::-1]): 
    #  #blocks in decoder are in descending order, going over them in ascending order
    #  btype = i + 1
    #  block.load_state_dict(torch.load(RDY_DECODER_BLOCK_PATH.format(btype)))
    #  print('loaded model %d' % btype)
      
    with torch.no_grad():
      tform = torchvision.transforms.Resize((1024,1024)) #todo
      x = transforms.functional.to_tensor(tform(Image.open('in1.jpg'))).to(device)
      f = encoder(x.unsqueeze(0))
      xhat = decoder(f).squeeze(0)
      img_file_path = "results/dec_{}_test.png".format(narch)
      batch = torch.cat((x.unsqueeze(0), xhat.unsqueeze(0)), 0)
      save_image(batch.data, img_file_path, nrow=2, normalize=args.norm)

if __name__ == "__main__":
    main()
