import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import models
from models import register
from utils import make_coord

@register('lte-fast')
class LTEP(nn.Module):

    def __init__(self, encoder_spec, num_layer=3, hidden_dim=256, out_dim=3):
        super().__init__()
        self.encoder = models.make(encoder_spec)
                
        # Fourier prediction
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1) # coefficient
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1) # frequency
        self.phase = nn.Linear(2, hidden_dim//2, bias=False) # phase
                
        layers = []
        for i in range(num_layer):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(hidden_dim, out_dim, 1))
        self.layers = nn.Sequential(*layers)
        
    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat
    
    def query_rgb(self, coord, cell):      
        feat = self.feat
        coef = self.coef(feat) # coefficient
        freq = self.freq(feat) # frequency
        
        # prepare meta-data (coordinate)
        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        
        # local ensemble loop
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # coefficient & frequency prediction
                coef_ = F.grid_sample(coef, coord_.flip(-1), mode='nearest', align_corners=False)
                freq_ = F.grid_sample(freq, coord_.flip(-1), mode='nearest', align_corners=False)

                # rel coord
                rel_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - rel_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]
                
                # cell
                rel_cell = cell.clone()
                rel_cell[:, 0] *= feat.shape[-2]
                rel_cell[:, 1] *= feat.shape[-1]
                
                # basis
                freq_ = torch.stack(torch.split(freq_, freq.shape[1]//2, dim=1), dim=2)
                freq_ = torch.mul(freq_, rel_coord.unsqueeze(1))
                freq_ = torch.sum(freq_, dim=2)
                freq_ += self.phase(rel_cell).unsqueeze(-1).unsqueeze(-1)
                freq_ = torch.cat((torch.cos(np.pi*freq_), torch.sin(np.pi*freq_)), dim=1)

                # apply coefficeint to basis
                coef_ = torch.mul(coef_, freq_)

                # shared MLP
                coef_ = self.layers(coef_)
                preds.append(coef_)
        
                # area
                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)
        
        # apply local ensemble
        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(1)
        return ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
                                padding_mode='border', align_corners=False)

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)