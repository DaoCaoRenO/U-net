import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=21):
        super(UNet, self).__init__()
        self.d1 = double_conv(in_channels, 64)
        self.d2 = double_conv(64, 128)
        self.d3 = double_conv(128, 256)
        self.d4 = double_conv(256, 512)
        self.b  = double_conv(512, 1024)
        
        self.pool = nn.MaxPool2d(2)
        
        self.u6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.c6 = double_conv(1024, 512)
        self.u7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.c7 = double_conv(512, 256)
        self.u8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c8 = double_conv(256, 128)
        self.u9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c9 = double_conv(128, 64)
        
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        c1 = self.d1(x)
        p1 = self.pool(c1)
        c2 = self.d2(p1)
        p2 = self.pool(c2)
        c3 = self.d3(p2)
        p3 = self.pool(c3)
        c4 = self.d4(p3)
        p4 = self.pool(c4)
        c5 = self.b(p4)
        
        u6 = self.u6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6(u6)
        u7 = self.u7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7(u7)
        u8 = self.u8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8(u8)
        u9 = self.u9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9(u9)
        
        out = self.out(c9)
        return F.log_softmax(out, dim=1)