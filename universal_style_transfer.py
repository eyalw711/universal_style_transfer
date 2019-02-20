import os, sys
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
import torch.optim as optim
import torch.utils.data

import torchvision.datasets
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as transforms

from utils import wct, Range, merge_function, boost

from seq_models import Encoder, Decoder
from ref_models_factory import Encoder as R_Encoder
from ref_models_factory import Decoder as R_Decoder

def style_transfer(args):
  """returns result without batch dimension"""
  merge = args.style_img_pair is not None
  if merge:
    merge = args.merge_method
    style_paths = args.style_img_pair.split(',')
  else:
    style_paths = [args.style_img]
    
  synth = args.texture
  
  if not args.no_cuda and torch.cuda.is_available():
    device = torch.device('cuda:0')
  else:
    device = torch.device('cpu')
  
  if args.single_level:
    ser = [4] #[5]
  elif merge == 2: #channel-merge
    ser = [5,4,3,2] # layer 1 ruins the result in default architecture
  else:
    ser = [5,4,3,2,1] # architecture pipeline
  
  if args.arch:
    ser = args.arch
  
  chnls_per_model = [64, 128, 256, 512, 512] # for 1 2 3 4 5 respectively
  divs_by_model = [(4,2), (2,2), (2,1), False, False] # for 1 2 3 4 5 respectively
  
  if args.ref_models:
    encoders = [R_Encoder(n).to(device) for n in ser]
    decoders = [R_Decoder(n).to(device) for n in ser]
  else:
    encoders = [Encoder(n).to(device) for n in ser]
    decoders = [Decoder(n, weight_init_method='full_net').to(device) for n in ser]
    
  for model in encoders+decoders:
    model.eval()
    
  tform = torchvision.transforms.Resize((1024,1024)) #todo
  
  style_imgs = [transforms.functional.to_tensor(tform(Image.open(style_path))) for style_path in style_paths]
  
  if args.texture:
    content_img = torch.zeros_like(style_imgs[0]).uniform_()
  else:
    content_img = transforms.functional.to_tensor(tform(Image.open(args.content_img)))
  
  # transfer tensors to device
  for i, im in enumerate(style_imgs):
    style_imgs[i] = im.to(device).unsqueeze(0)
  content_img = content_img.to(device).unsqueeze(0)
  
  ################
  #   Pipeline   #
  ################
  
  for level in range(len(ser)):
    print('pipeline level {}'.format(level+1))
    chns = chnls_per_model[ser[level]-1]
    
    #encoder,decoder expect batch dimension, wct doesn't
    cf = encoders[level](content_img).data.to(device).squeeze(0)
    if not merge:
      sf = encoders[level](style_imgs[0]).data.to(device).squeeze(0)
    else:
      sf = merge_function(merge, style_imgs, args.beta, encoders, level, device) # merge returns squeezed
    
    csf = wct(args.alpha, cf, sf)
    
    if merge == 1: #original merge takes special care
      sf2 = merge_function(merge, reversed(style_imgs), args.beta, encoders, level, device)
      csf2 = wct(args.alpha, cf, sf2)
      csf = args.beta*csf + (1-args.beta)*csf2
    
    if args.boost:
      level_rois = [divs_by_model[i-1] for i in ser]
      level_roi = level_rois[level]
      if level_roi:
        csf = boost(csf.squeeze(0), cf, sf, level_rois[level], device) # returns with batch dim
      
    content_img = decoders[level](csf)
  
  return content_img

def main():
  tm = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
  
  parser = argparse.ArgumentParser(description='Universal Style Transfer')
  parser.add_argument('--no-cuda', default=False, action='store_true', help='Dont use CUDA GPU capabilities')
  parser.add_argument('--single-level', default=False, action='store_true', help='flag to transfer style using a single level pipeline')
  parser.add_argument('--texture', default=False, action='store_true', help='flag to synthesize texture (without content image)')
  parser.add_argument('--boost', default=False, action='store_true', help='flag to enable style transfer boosting')
  
  parser.add_argument('--content-img', help='path to the content image')
  parser.add_argument('--style-img', help='path to the style image')
  parser.add_argument('--style-img-pair', help='paths to two style images seperated by ","')
  parser.add_argument('--out-dir', default='results', help='directory outputs are saved in')
  parser.add_argument('--filename', default=None, help='output filename')
  
  parser.add_argument('--merge-method', type=int, default=4, help='method of merging two styles. select 1 for "original", 2 for "channel-merge", 3 for "level-merge" and 4 for "feature-average"', choices=[1,2,3,4])

  parser.add_argument('--alpha', type=float, default=0.5, choices=[Range(0.0, 1.0)], help='original content features and transformed features interpolation parameter')
  parser.add_argument('--beta', type=float, default=0.5, choices=[Range(0.0, 1.0)], help='weighing the style images in the style-img-pair parameter')
  
  parser.add_argument('--ref-models', default=False, action='store_true', help='use reference models')
  parser.add_argument('--arch', help='custom architecture')
  args = parser.parse_args()
  
  ########################
  #      Check args      #
  ########################
  
  assert (args.style_img is not None) + (args.style_img_pair is not None) == 1, "Must choose either one style or style-pair"
  
  ## verify valid paths
  # content
  if args.content_img is not None:
    assert os.path.isfile(args.content_img), "Must enter valid content image path" 
    
  # style
  if args.style_img is not None: # 1 style
    assert os.path.isfile(args.style_img), "Must enter valid style image path"
  else:                          # 2 styles
    style_paths = args.style_img_pair.split(',')
    assert len(style_paths) == 2, "Must enter exactly two style image paths"
    for i, pth in enumerate(style_paths):
      assert os.path.isfile(pth), "{} style path is invalid".format("1st" if i==0 else "2nd")
    assert args.boost is False, "Currently boosting is not supported with two styles"
    assert args.single_level is False, "Cannot support single level pipeline architecture with two style"
  
  ## out result path
  os.makedirs(args.out_dir, exist_ok=True)
  if args.filename is not None:
    assert args.filename.endswith('.png') or args.filename.endswith('.jpg')
  else:
    args.filename = tm + '.png'

  img_file_path = os.path.join(args.out_dir, args.filename) #TODO out_dir might have wrong seperator
  
  ## content or texture
  assert (args.content_img is not None) + args.texture == 1, "Must choose either a content image OR texture"
  
  ## custom architecture
  if args.arch:
    assert args.arch.isdigit(), "arch must be a string of digits from 1 to 5"
    args.arch = [int(c) for c in args.arch]
    assert all(0<j<6 for j in args.arch), "arch must be a string of digits from 1 to 5"
  

    
  ########################
  #     Run Program      #
  ########################
  
  with torch.no_grad():
    result = style_transfer(args)
    result = result.unsqueeze(0)
    
  save_image(result.data, img_file_path, nrow=1, normalize=False)
  print("Style Transfer Complete. result saved to {}".format(img_file_path))
  
  
if __name__ == "__main__":
    main()