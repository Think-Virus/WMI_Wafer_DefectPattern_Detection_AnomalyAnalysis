# wmi_triage/model.py
import torch.nn as nn
from torchvision import models


class ResNet18TVEmbed(nn.Module):
    """
    ckpt 키와 동일한 형태 유지:
    conv1, bn1, layer1..4, avgpool, fc
    forward는 (logits, emb) 반환
    """

    def __init__(self, num_classes: int, in_chans: int = 3):
        super().__init__()
        m = models.resnet18(weights=None)

        # polar4면 conv1 입력 채널 4로 변경
        if in_chans != 3:
            m.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # head 변경
        m.fc = nn.Linear(512, num_classes)

        # ✅ 레이어를 "직접" 들고 있어서 state_dict 키가 ckpt와 동일해짐
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.avgpool = m.avgpool
        self.fc = m.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        emb = x.flatten(1)  # (N,512)
        logits = self.fc(emb)  # (N,C)
        return logits, emb
