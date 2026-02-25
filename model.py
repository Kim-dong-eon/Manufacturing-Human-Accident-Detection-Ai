import torch
from torch import nn

from models import cnnlstm

def generate_model(opt, device):
    assert opt.model in [
        'cnnlstm'
    ]

    if opt.model == 'cnnlstm':
        # backbone_name 옵션을 명시적으로 전달
        model = cnnlstm.CNNLSTM(num_classes=opt.n_classes, backbone_name=opt.backbone_name.lower())
    return model.to(device)
