import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2, backbone_name='resnet18'):
        super(CNNLSTM, self).__init__()
        self.backbone_name = backbone_name.lower()
        backbones = {
            'resnet18': (models.resnet18, 512),
            'resnet101': (models.resnet101, 2048),
            'efficientnet_b0': (models.efficientnet_b0, 1280),
        }
        if self.backbone_name not in backbones:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        backbone_fn, feat_dim = backbones[self.backbone_name]
        self.resnet = backbone_fn(pretrained=True)
        # EfficientNet classifier 제거 + 특성 변환 FC 정의
        if 'efficientnet' in self.backbone_name:
            self.resnet.classifier = nn.Identity()
            self.fc_transform = nn.Linear(feat_dim, 300)
        else:
            self.resnet.fc = nn.Sequential(nn.Linear(feat_dim, 300))
            self.fc_transform = nn.Identity()
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                feat = self.resnet(x_3d[:, t, :, :, :])
            feat = self.fc_transform(feat)  # EfficientNet은 여기서 강제 변환
            out, hidden = self.lstm(feat.unsqueeze(0), hidden)
        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x
