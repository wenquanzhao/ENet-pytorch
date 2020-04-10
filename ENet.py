import torch
import torch.nn as nn

class initialBlock(nn.Module):
    ''' 
    Structe of ENet initial block
                    input
                   /     \
                  /       \                
                 /         \        
     3*3, stride 2        MaxPooling
    (normalization)\       /
                    \     /
                 Concat-PReLU
                    
    (1) maxpooling is performed with non-overlap 2*2 window
    (2) conv3x3 = nn.Conv2d(3, 13, 3, 2, 1)
    input channels are 3, output channels are 13(fliters)
    kernel size is 3, stride is 2, padding is 1
    (3) normalization on each channel
    (4) if the input image(x) is 512 * 512 * 3, output is 16 feature maps after concatenation
    '''
    def __init__(self, in_channels = 3, out_channels = 13, debug = False):
        super(initialBlock, self).__init__()
        # in_channels = 3, out_channels =13, 
        # kernel_size = 3, stride = 2, padding = 1, bias=False
        self.debug = debug
        self.conv3x3 = nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size = 3,
                                 stride = 2,
                                 padding = 1,
                                 bias = False)
        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                      stride = 2,
                                      padding = 0)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU(16)
    def forward(self, x):
        if self.debug:
            # batch, channels, height, width, i.e. (N,C,H,W)
            x = torch.ones(1, 3, 512, 512)
        
        main = self.conv3x3(images)   # 1, 13, 256, 256
        main = self.batchnorm(main)   # normalization on each channel-
                                      # all pixels are used to compute Mean and Var
        side = self.maxpool(images)   # 1, 3, 256, 256 
        
        x = torch.cat((main, side), dim=1) # torch.Size([1, 16, 256, 256])
        x = self.activation(x)
        # PReLU(x) = max(0, x) + a * min(0,x)
        # Called with nn.PReLU(nChannels)
        # vector a is used for each input channel.
        return x



class ENet(nn.Module):
    # For the regularizer, we use Spatial Dropout, with p = 0.01
    # before bottlenect 2.0, and p = 0.1 afterwards
    def __init__(self, C):
        super(ENet, self).__init__()
        # C - number of classes
        self.C = C
        # The finitial bottlenectk
        self.init = initialBlock()
    def forward(self, x):
        # The initial block
        x = self.init(x)
        return x


if __name__=="__main__":
    print("Hello ENet for semantic segmentation")
    C = 3
    ENet = ENet(C)
    images = torch.rand(6, 3, 512, 512)
    output = ENet(images)
    print(output.shape)