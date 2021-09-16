import torch
import torch.nn as nn
from torch.nn import Conv2d as Conv2D
import torch.nn.init as init
import torch.nn.functional as F
import numpy
from torch.nn import Upsample

class Up(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Up, self).__init__()
        self.conv = nn.Sequential(
            Conv2D(channel_in, 4*channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(4*channel_out),
            nn.ReLU(inplace=True),
            Conv2D(4*channel_out, 2*channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(2*channel_out),
            nn.ReLU(inplace=True),            
            Conv2D(2*channel_out, channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True),
            Conv2D(channel_out, channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)
        )
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        
    def forward(self, x1, x2):
        x1 = self.conv(x1)
        difference_in_X = x1.size()[2] - x2.size()[2]
        difference_in_Y = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (difference_in_X // 2, int(difference_in_X / 2),
                        difference_in_Y // 2, int(difference_in_Y / 2)))
        x = torch.cat([x2, x1], dim=1)                
        x= self.upsample(x)
        return x

class Up_final(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Up_final, self).__init__()
        self.conv = nn.Sequential(
            Conv2D(channel_in, 4*channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(4*channel_out),
            nn.ReLU(inplace=True),
            Conv2D(4*channel_out, 2*channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(2*channel_out),
            nn.ReLU(inplace=True),            
            Conv2D(2*channel_out, channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True),
            Conv2D(channel_out, channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self, x1):
        x = self.conv(x1)             
        return x

class Down(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            Conv2D(channel_in, channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True),
            Conv2D(channel_out, channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)           
        )
    
    def forward(self, x):
        x = F.max_pool2d(x,2)
        x = self.conv(x)
        return x

class Down_final(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Down_final, self).__init__()
        self.conv = nn.Sequential(
            Conv2D(channel_in, channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True),
            Conv2D(channel_out, channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)           
        )
    
    def forward(self,x1,x2,x3,x4):
        x = torch.cat([x2, x1], dim=1) 
        x = torch.cat([x3, x], dim=1)  
        x = torch.cat([x4, x], dim=1) 
        x = self.conv(x)
        return x
        





class et(nn.Module):
    def __init__(self, channel_in=1, classes=3):
        super(et, self).__init__()
        self.input_conv = nn.Sequential(
            Conv2D(channel_in, 8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            Conv2D(8, 8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),     
            Conv2D(8, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            Conv2D(16, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)                   
        )
        
        self.conv_final = nn.Sequential(
            Conv2D(256*3, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Conv2D(256, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)                
        )
        
        
        self.down0 = Down(16, 32)        
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128,256)
        self.down4 = Down_final(1024,2048)
      
        
        self.up1 = Up(2048, 256)
        self.up2 = Up(1024, 128)
        self.up3 = Up(512, 64)
        self.up4 = Up(256, 32)
        self.up5 = Up_final(128,16)        
        self.output_conv = nn.Conv2d(16, classes, kernel_size = 1)
        
    def forward(self,im):

        im0 = im[:,:,0:512,:]
        im1 = im[:,:,512:1024,:]
        dem = im[:,:,1024:1536,:]


        im0_0 = self.input_conv(im0)
        im1_0 = self.input_conv(im1)
        dem_0 = self.input_conv(dem)
        
        im0_1 = self.down0(im0_0)
        im1_1 = self.down0(im1_0)
        dem_1 = self.down0(dem_0)	
        cat1 = torch.cat([im0_1, im1_1], dim=1)
        cat_1 = torch.cat([cat1, dem_1], dim=1)  #32*3,256,256
 
        return cat_1
        #return F.sigmoid(output)
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight, gain=numpy.sqrt(2.0))
        init.constant(m.bias, 0.1)
    
