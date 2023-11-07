import torch.nn as nn
import torch
import torch.nn.functional as F

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Unit of ASPP R2U-Net with Attention Gates.
    """
    def __init__(self, in_channel=512, depth=256):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
 
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
    
        return net

class reshape_layer(nn.Module):
    """
    Reshaping the input by performing convolution and batch normalisation.
    """
    def __init__(self, in_channels, out_channels):
        super(reshape_layer, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.seq(x)  
    
class default_layer(nn.Module):
    """
    Default CNN like layer with convolution, maxpooling and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(default_layer, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)  
    
class up_layer(nn.Module):
    """
    Decoding Layer of the Decoding Unit.
    """
    def __init__(self,in_channels,out_channels):
        super(up_layer,self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.layer = default_layer(in_channels, out_channels)
        
    def forward(self,x):
        return self.layer(self.upsample(x))

class recurrent_layer(nn.Module):
    """
    Default layers modified according to the recurrent block of R2U-Net.
    """
    def __init__(self, channels):
        super(recurrent_layer, self).__init__()
        self.layer = default_layer(channels, channels)

    def forward(self, x):
        return self.layer(x + self.layer(x))  
    
class recurrent_block(nn.Module):
    """
    Recurrent Residual Block of the R2U-Net. 
    """
    def __init__(self,  in_channels, out_channels):
        super(recurrent_block, self).__init__()
        self.r2c = nn.Sequential(
            recurrent_layer(out_channels),
            recurrent_layer(out_channels)
        )
        self.conv1x1 = nn.Conv2d(in_channels,out_channels,1)
        
    def forward(self, x):
        x = self.conv1x1(x)
        return x + self.r2c(x)
    
class down_block(nn.Module):
    """
    Encoding Unit of the R2U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super(down_block, self).__init__()
        self.r2c = recurrent_block(in_channels,out_channels)
        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.r2c(x)
        return self.maxpool(x), x
    
class up_block(nn.Module):
    """
    Decoding Unit of the R2U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super(up_block, self).__init__()
        self.up_layer = up_layer(in_channels, out_channels)
        self.r2c = recurrent_block(in_channels,out_channels)
        self.attention = attention_block(in_channels // 2, out_channels // 2 )

    def forward(self, x, attachment):
        x = self.up_layer(x)
        attachment = self.attention(x, attachment)
        x = torch.cat((attachment, x),dim=1)
        x = self.r2c( x)
        return x
    
class attention_block(nn.Module):
    """
    Attention Gates Unit of the ASPP R2U-Net with Attention Gate.
    """
    def __init__(self, in_channels, out_channels):
        super(attention_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.attachment_reshape = reshape_layer(in_channels, out_channels)
        self.x_reshape = reshape_layer(in_channels, out_channels)
        self.output_reshape = reshape_layer(out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, attachment):
        x_reshaped = self.x_reshape(x)
        attachment_reshaped = self.attachment_reshape(attachment)
        relu = self.relu( x_reshaped + attachment_reshaped )
        return attachment * self.sigmoid(self.output_reshape( relu ))
    
class R2U_Net_Optimized(nn.Module):
    """
    Class denoting Atrous Spatial Pyramid Pooling R2U-Net with Attention Gates.
    """
    def __init__(self, classes):
        super(R2U_Net_Optimized, self).__init__()
        
        self.aspp = ASPP(3, 32)
        
        self.down_rcl1 = down_block(32, 64)
        self.down_rcl2 = down_block(64, 128)
        self.down_rcl3 = down_block(128, 256)
        self.down_rcl4 = down_block(256, 512)
        
        self.down_rcl5 = recurrent_block(512, 1024)
        
        self.up_rcl1 = up_block(1024, 512)
        self.up_rcl2 = up_block(512, 256)
        self.up_rcl3 = up_block(256, 128)
        self.up_rcl4 = up_block(128, 64)
        self.conv1x1 = nn.Conv2d(64,classes,1)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.aspp(x)
        
        x, a = self.down_rcl1(x)
        x, b = self.down_rcl2(x)
        x, c = self.down_rcl3(x)
        x, d = self.down_rcl4(x)
        x = self.down_rcl5(x)
        
        x = self.up_rcl1(x, d)
        x = self.up_rcl2(x, c)
        x = self.up_rcl3(x, b)
        x = self.up_rcl4(x, a)
        x = self.conv1x1(x)
        
        return x
        