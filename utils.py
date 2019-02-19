import torch
import numpy as np
import random
import collections


class Range(object):
  def __init__(self, start, end):
    self.start = start
    self.end = end
  def __eq__(self, other):
    return self.start <= other <= self.end
  def __repr__(self):
    return "[{}, {}]".format(self.start, self.end)


def merge_function(merge, style_imgs, beta, encoders, level, device):
  """gets style imgs with batch dimension, returns features with NO batch dimension"""
  enc = encoders[level]
  
  if merge == 1: # original-merge twice coloring-svd
    raise NotImplementedError

  elif merge == 2: # channel-merge: shuffle channels only 1 coloring-svd
    d = 4
    sfs = [enc(style_img).data.to(device).squeeze(0) for style_img in style_imgs]
    chns = sfs[0].size(0)
    
    sf1 = torch.split(sfs[0], chns//d, dim=0)
    sf2 = torch.split(sfs[1], chns//d, dim=0)
    sf = torch.cat([sf1[i] if i%2==0 else sf2[i] for i in range(d)], 0)
    
  elif merge == 3: # level-merge: each level use different sf - only 1 coloring-svd
    sf = encoders[level](style_imgs[level%2]).data.to(device).squeeze(0)
      
  elif merge == 4: # Interpolated-Style Merge: weighted average sf before wct
    # Difference is that
    # original plan is to: whiten cf, color with sf1 (svd), color with sf2 (svd), merge
    # new plan is to:      whiten cf, color with beta*sf1+(1-beta)*sf2 (only 1 coloring-svd)
    sfs = [enc(style_img).data.to(device).squeeze(0) for style_img in style_imgs]
    print(sfs[0].size(), sfs[1].size())
    sf = beta*sfs[0] + (1-beta)*sfs[1]
  
  else:
    raise ValueError
  
  return sf


def raised_cosine_kern(h,w):
  beta = 0.25
  T = 1.0
  L = (1.0 + beta)/(2*T)
  Ls = (1.0 - beta)/(2*T)

  x = np.linspace(-L, L, h)
  y = np.linspace(-L, L, w)
  xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

  kern = np.ones_like(xv)

  binx1 = np.abs(xv) > Ls
  binx2 = np.abs(yv) > Ls

  kern[binx1] = kern[binx1]*0.5*(1.0 + np.cos((np.pi*T/beta)*(np.abs(xv[binx1]) - Ls)))
  kern[binx2] = kern[binx2]*0.5*(1.0 + np.cos((np.pi*T/beta)*(np.abs(yv[binx2]) - Ls)))

  return torch.from_numpy(kern).view(1, h, w).float()

ROIdef = collections.namedtuple('ROIdef', ['tl', 'br'])
Point = collections.namedtuple('Point', ['h', 'w'])

def gen_roi_defs(h, w, numROIs, hr, wr):
  maxh_tl = h - hr + 1
  maxw_tl = w - wr + 1
  
  tlhs = random.choices(range(maxh_tl), k=numROIs)
  tlws = random.choices(range(maxw_tl), k=numROIs)
  
  roidefs = [ROIdef(Point(tlhs[j], tlws[j]), Point(tlhs[j]+hr, tlws[j]+wr)) for j in range(numROIs)]
  return roidefs


def boost(csf, cf, sf, level_roi, device):
  """gets tensors without batch dimension, returns with"""
  # take ROI channels from csf and apply it orig_csf
  c, h, w = cf.size()
  numrois = level_roi[0]*level_roi[1] #TEMP
  
  hr, wr = h//level_roi[0], w//level_roi[1]
  roidefs = gen_roi_defs(h, w, numrois, hr, wr)
  print(roidefs)
  
  croi = torch.cat([cf[:, roidef.tl.h:roidef.br.h, roidef.tl.w:roidef.br.w] for roidef in roidefs], 0)
  sroi = torch.cat([sf[:, roidef.tl.h:roidef.br.h, roidef.tl.w:roidef.br.w] for roidef in roidefs], 0)
  
  csroi = wct(0.75, croi, sroi)
  csroi = csroi.squeeze(0)
       
  # put ROIs back in place
  csroi = [csroi[i*c:(i+1)*c, :, :] for i in range(numrois)]
  for i in range(numrois):
    roidef = roidefs[i]
    avging_filter = 0.8*raised_cosine_kern(hr, wr).expand(c,hr,wr).to(device)
    csf[:, roidef.tl.h:roidef.br.h, roidef.tl.w:roidef.br.w] = \
      (1.0 - avging_filter)*csf[:, roidef.tl.h:roidef.br.h, roidef.tl.w:roidef.br.w] + avging_filter*csroi[i]
  return csf.unsqueeze(0)


def wct(alpha, content_f, style_f):
  """expects to get features with NO batch dimension"""
  USE_ALL_SINGULARS = False
  content_f = content_f.double()
  style_f = style_f.double()
  sv_threshold = 0.00001

  # SVD computation
  def SVD(Image_f, color):
    Image_f = Image_f.double()
    channels, width, height = Image_f.size(0), Image_f.size(1), Image_f.size(2)
    Image_f_mat = Image_f.view(channels, -1)  # c x (h x w)
    # compute mean and subtract it from feature mat
    Image_f_mean = torch.mean(Image_f_mat, 1) # perform mean for each row
    Image_f_mean = Image_f_mean.unsqueeze(1).expand_as(Image_f_mat) # add dim and replicate mean on rows
    Image_f_mat = Image_f_mat - Image_f_mean # subtract mean element-wise
    
    Image_f_cov_mat = torch.mm(Image_f_mat, Image_f_mat.t()).div((width * height) - 1)  # construct covariance matrix
    u, s, v = torch.svd(Image_f_cov_mat, some=False) # singular value decomposition

    if (USE_ALL_SINGULARS):
      len_singulars = len(s)
    else:
    # take singular values greater than value 
      len_singulars = len(s[s >= sv_threshold])
    if color:
      s = (s[0:len_singulars]).pow(0.5)
    else:
      s = (s[0:len_singulars]).pow(-0.5)
    step_1 = torch.mm(v[:, 0:len_singulars], torch.diag(s))
    step_2 = torch.mm(step_1, (v[:, 0:len_singulars].t()))
    if color:
      return step_2, Image_f_mat, Image_f_mean
    else:
      return step_2, Image_f_mat

  # whitening
  step_2, content_f_mat_zeromean = SVD(content_f, color=False)
  whitened = torch.mm(step_2, content_f_mat_zeromean)
  # coloring
  step_2, style_f_mat_zeromean, style_f_mean = SVD(style_f, color=True)
  colored = torch.mm(step_2, whitened) 
  content_style_features = colored + style_f_mean.resize_as_(colored)
  content_style_features = content_style_features.view_as(content_f)
  colored_content_style_features = alpha * content_style_features + (1.0 - alpha) * content_f
  return colored_content_style_features.float().unsqueeze(0)