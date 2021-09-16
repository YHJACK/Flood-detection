import torch
import torch.nn as nn
from torch.nn import Conv2d as Conv2D
import torch.nn.init as init
import torch.nn.functional as F
import numpy
from torch.nn import Upsample

class sSE(nn.Module):
    def __init__(self, in_channels):
        super(sSE, self).__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super(cSE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels //4, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//4, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class SCSE(nn.Module):
    def __init__(self, in_channels):
        super(SCSE, self).__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse































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
        

class UNet(nn.Module):
    def __init__(self, channel_in=1, classes=3):
        super(UNet, self).__init__()
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

        self.sce16 = SCSE(16)
        self.sce32 = SCSE(32)
        self.sce64 = SCSE(64)
        self.sce128 = SCSE(128)
        self.sce256 = SCSE(256)
        self.sce512 = SCSE(512)
        self.sce1024 = SCSE(1024)
        self.sce2048 = SCSE(2048)
        
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
        
        im0_1 = self.sce32(im0_1)
        im1_1 = self.sce32(im1_1)
        dem_1 = self.sce32(dem_1)        
        	
        cat1 = torch.cat([im0_1, im1_1], dim=1)
        cat_1 = torch.cat([cat1, dem_1], dim=1)  #32*3,256,256
         
        im0_2 = self.down1(im0_1)
        im1_2 = self.down1(im1_1)
        dem_2 = self.down1(dem_1)
        
        im0_2 = self.sce64(im0_2)
        im1_2 = self.sce64(im1_2)
        dem_2 = self.sce64(dem_2)
                
        cat2 = torch.cat([im0_2, im1_2], dim=1)
        cat_2 = torch.cat([cat2, dem_2], dim=1)  #64*3,128,128       
                      
        im0_3 = self.down2(im0_2)
        im1_3 = self.down2(im1_2)
        dem_3 = self.down2(dem_2)
        
        im0_3 = self.sce128(im0_3)
        im1_3 = self.sce128(im1_3)
        dem_3 = self.sce128(dem_3)        
              
        cat3 = torch.cat([im0_3, im1_3], dim=1)
        cat_3 = torch.cat([cat3, dem_3], dim=1)  #128*3,64,64              
                   
        im0_4 = self.down3(im0_3)
        im1_4 = self.down3(im1_3)
        dem_4 = self.down3(dem_3)
        
        im0_4 = self.sce256(im0_4)
        im1_4 = self.sce256(im1_4)
        dem_4 = self.sce256(dem_4)         
        
        
        cat4 = torch.cat([im0_4, im1_4], dim=1)
        cat_4 = torch.cat([cat4, dem_4], dim=1)  #256*3,32,32   

        x = self.conv_final(cat_4)  #256,32,32      
        x = self.down4(im0_4,im1_4,dem_4,x) #2048 32 32
        #x = self.sce2048(x)
        
        x = self.up1(x, cat_4)
        x = self.sce1024(x)
        x = self.up2(x, cat_3)
        x = self.sce512(x)
        x = self.up3(x, cat_2)
        x = self.sce256(x)
        x = self.up4(x, cat_1)
        x = self.sce128(x)
        x = self.up5(x)
        #x = self.sce16(x)
        output = self.output_conv(x)
        return output
       # return F.sigmoid(output)
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight, gain=numpy.sqrt(2.0))
        init.constant(m.bias, 0.1)
    
