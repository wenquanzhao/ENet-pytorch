import torch
import torch.nn as nn

class initialBlock(nn.Module):
    ''' 
    #Structe of ENet initial block
    #                input
    #               /     \
    #              /       \                
    #             /         \        
    # 3*3, stride 2        MaxPooling
    #(normalization)\       /
    #                \     /
    #             Concat-PReLU
    #                
    #(1) maxpooling is performed with non-overlap 2*2 window
    #(2) conv3x3 = nn.Conv2d(3, 13, 3, 2, 1)
    #input channels are 3, output channels are 13(fliters)
    #kernel size is 3, stride is 2, padding is 1
    #(3) normalization on each channel
    #(4) if the input x is 512 * 512 * 3, 
    # output is 16 feature maps after concatenation
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
        
        main = self.conv3x3(x)   # 1, 13, 256, 256
        main = self.batchnorm(main)   # normalization on each channel-
                                      # all pixels are used to compute Mean and Var
        side = self.maxpool(x)   # 1, 3, 256, 256 
        
        x = torch.cat((main, side), dim=1) # torch.Size([1, 16, 256, 256])
        x = self.activation(x)
        # PReLU(x) = max(0, x) + a * min(0,x)
        # Called with nn.PReLU(nChannels)
        # vector a is used for each input channel.
        return x

class bottleNeck(nn.Module):
    '''
    # Regular|Dilated|Deconvolution bottleneck:
    #
    #     Bottleneck Input
    #        /        \
    #       /          \
    # maxpooling2d   conv2d-1x1(named conv1)
    #      |             | PReLU
    #      |         conv2d-3x3(named conv2)
    #      |             | PReLU
    #      |         conv2d-1x1(named conv3)
    #      |             |
    #  Padding2d     Regularizer
    #       \           /  
    #        \         /
    #      Summing + PReLU
    # Params: 
    #  sampling_flag(bool) - if True: down sampling, if False: no sampling
    #  ratio - ratio between input and output channels
    #  p - dropout ratio
    #  default:
    #  activation: the activation function is PReLU(x) = max(0, x) + a* min(0,x)
    '''
    def __init__(self, in_channels, out_channels, dilation, sampling_flag = False, ratio = 4, p = 0.1):
       super(bottleNeck, self).__init__()
       self.in_channels = in_channels
       self.out_channels = out_channels
       self.dilation = dilation
       self.sampling_flag = sampling_flag
       
       # caluculating the number of reduced channels
       # self.reduced_channels = int()
       if sampling_flag == True:
           print('== down sampling ==')
           self.stride = 2
           self.reduced_channels = int(in_channels // ratio)
       if sampling_flag == False:
           print('== no down sampling ==')
           self.stride = 1
           self.reduced_channels = int(out_channels // ratio)
           
       self.activation = nn.PReLU()
       
       # non-overlap 2*2 window
       self.maxpool = nn.MaxPool2d(2, return_indices = True)
       # Randomly zero out entire channels, input[i, j],
       # the j-th channel of the i-th sample in the batched input
       self.dropout = nn.Dropout2d(p = p)
       
       ''' The following sentences are in Adam Paszke's paper
       If the bottleneck is downsampling,
       (1) a max pooling layer is addded to the main branch
       (2) the first 1 x 1 projection is replaced with
       a 2 x 2 convolution with stride 2 in both dimensions.
       '''
       self.conv1 = nn.Conv2d(in_channels = self.in_channels,
                              out_channels = self.reduced_channels,
                              kernel_size = 1,
                              stride = 1,
                              padding = 0,
                              dilation = 1,
                              bias = False) # mentioned in the paper, no bias
       self.conv2 = nn.Conv2d(in_channels = self.reduced_channels,
                              out_channels = self.reduced_channels,
                              kernel_size = 3,
                              stride = self.stride,
                              padding = self.dilation, # attention
                              dilation = self.dilation,
                              bias = True)
       
       self.conv3 = nn.Conv2d(in_channels = self.reduced_channels,
                              out_channels = self.out_channels,
                              kernel_size = 1,
                              stride = 1,
                              padding = 0,
                              dilation = 1,
                              bias = False)
       # normalization needs the number of channels
       self.batchNorm1 = nn.BatchNorm2d(self.reduced_channels)
       self.batchNorm2 = nn.BatchNorm2d(self.out_channels)
    def forward(self, x):
        batch_size =  x.shape[0]
        #print(x.shape)
        x_main = x
    
        # main branch
        if self.sampling_flag:
            x_main, indices = self.maxpool(x_main)
            
        if self.in_channels != self.out_channels:
            delta_channels = self.out_channels - self.in_channels
            #padding and concatenating for matching the channels axis 
            # of the extension and main branches
            extras = torch.zeros(batch_size,
                                 delta_channels,
                                 x_main.shape[2],
                                 x_main.shape[3])
            if torch.cuda.is_available():
                extras = extras.cuda()
            x_main = torch.cat([x_main, extras], dim = 1)
        
        # extension
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.batchNorm1(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        x = self.dropout(x)
        # print('the shape of x', x.shape)
        # summing and prelu
        x = x + x_main
        x = self.activation(x)
        
        if self.sampling_flag:
            return x, indices
        else:
            return x

class bottleNeck_as(nn.Module):
    '''
    # Asymetric bottleneck:
    #
    #     Bottleneck Input
    #        /        \
    #       /          \
    #      |         conv2d-1x1
    #      |             | PReLU
    #      |         conv2d-1x5
    #      |             |
    #      |         conv2d-5x1
    #      |             | PReLU
    #      |         conv2d-1x1
    #      |             |
    #  Padding2d     Regularizer
    #       \           /  
    #        \         /
    #      Summing + PReLU
    #
    # Params:    
    #  ratio - ratio between input and output channels
    '''
    def __init__(self, in_channels, out_channels, ratio=4):
        super(bottleNeck_as, self).__init__()
        self.in_channels = in_channels
        self.reduced_channels = int(in_channels / ratio)
        self.out_channels = out_channels
        
        self.dropout = nn.Dropout2d(p = 0.1)
        
        self.conv1 = nn.Conv2d(in_channels = self.in_channels,
                               out_channels = self.reduced_channels,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               bias = False)
        self.activation = nn.PReLU()
        self.conv21 = nn.Conv2d(in_channels = self.reduced_channels,
                                out_channels = self.reduced_channels,
                                kernel_size = (1,5),
                                stride = 1,
                                padding = (0, 2),
                                bias = False)
        self.conv22 = nn.Conv2d(in_channels = self.reduced_channels,
                                out_channels = self.reduced_channels,
                                kernel_size = (5,1),
                                stride = 1,
                                padding = (2, 0),
                                bias  = False)
        self.conv3 = nn.Conv2d(in_channels = self.reduced_channels,
                               out_channels = self.out_channels,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               bias = False)
        self.batchNorm1 = nn.BatchNorm2d(self.reduced_channels)
        self.batchNorm2 = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_main = x
        # main branch
        if self.in_channels != self.out_channels:
            delta_channels = self.out_channels - self.in_channels
            #padding and concatenating in order to match the channels axis of the side and main branches
            extras = torch.zeros((batch_size, delta_channels, x_main.shape[2], x_main.shape[3]))
            if torch.cuda.is_available():
                extras = extras.cuda()
            x_main = torch.cat((x_main, extras), dim = 1)
        # extension
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.activation(x)
        
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.batchNorm1(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        
        x = self.dropout(x)
        x = self.batchNorm2(x)
        
        # summing
        x = x + x_main
        x = self.activation(x)
        
        return x
    
class bottleNeck_up(nn.Module):
    # Upsampling bottleneck:
    #     Bottleneck Input
    #        /        \
    #       /          \
    # conv2d-1x1     convTrans2d-1x1
    #      |             | PReLU
    #      |         convTrans2d-3x3
    #      |             | PReLU
    #      |         convTrans2d-1x1
    #      |             |
    # maxunpool2d    Regularizer
    #       \           /  
    #        \         /
    #      Summing + PReLU
    #  Regularizer: L1, L2, Dropout 
    #  Params: 
    #  ratio - ratio between input and output channels
    #  activation: the activation function is PReLU(x) = max(0, x) + a*min(0,x)
    def __init__(self, in_channels, out_channels, ratio = 4):
        super(bottleNeck_up, self).__init__()
        self.in_channels = in_channels
        self.reduced_channels = int(in_channels / ratio)
        self.out_channels = out_channels
        
        self.activation = nn.PReLU()
        
        self.unpool = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        self.main_conv = nn.Conv2d(in_channels = self.in_channels,
                              out_channels = self.out_channels,
                              kernel_size = 1)
                              
        self.dropout = nn.Dropout2d(p = 0.1)
        
        self.convt1 = nn.ConvTranspose2d(in_channels = self.in_channels,
                                         out_channels = self.reduced_channels,
                                         kernel_size = 1,
                                         padding = 0,
                                         bias = False)
        # This layer used for unsampling
        self.convt2 = nn.ConvTranspose2d(in_channels = self.reduced_channels,
                                         out_channels = self.reduced_channels,
                                         kernel_size = 3,
                                         stride = 2,
                                         padding =1,
                                         output_padding = 1,
                                         bias = False)
        self.convt3 = nn.ConvTranspose2d(in_channels = self.reduced_channels,
                                         out_channels = self.out_channels,
                                         kernel_size = 1,
                                         padding = 0,
                                         bias = False)
        self.batchNorm1 = nn.BatchNorm2d(self.reduced_channels)
        self.batchNorm2 = nn.BatchNorm2d(self.out_channels)
    def forward(self, x, indices):
        x_main = x
        # print("Before operation, the size of x:", x.size())
        # extension
        x = self.convt1(x)
        x = self.batchNorm1(x)
        x = self.activation(x)
        
        x = self.convt2(x)
        x = self.batchNorm1(x)
        x = self.activation(x)
        
        x = self.convt3(x)
        x = self.batchNorm2(x)
        x = self.activation(x)
        
        x = self.dropout(x)
        
        # main branch
        x_main = self.main_conv(x_main)
        # print('Check the size: ',x_main.size(), x.size())
        # input()
        x_main = self.unpool(x_main, indices, output_size = x.size())
        
        # summing
        x = x + x_main
        x = self.activation(x)
        
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
        # Stage 1: 5 bottleneck blocks
        self.b10 = bottleNeck(in_channels = 16,
                              out_channels = 64,
                              dilation = 1,
                              sampling_flag = True,
                              p = 0.01)
        self.b11 = bottleNeck(in_channels = 64,
                              out_channels = 64,
                              dilation = 1,
                              sampling_flag = False,
                              p = 0.01)
        self.b12 = bottleNeck(in_channels = 64,
                              out_channels = 64,
                              dilation = 1,
                              sampling_flag = False,
                              p = 0.01)
        self.b13 = bottleNeck(in_channels = 64,
                              out_channels = 64,
                              dilation = 1,
                              sampling_flag = False,
                              p = 0.01)
        self.b14 = bottleNeck(in_channels = 64,
                              out_channels = 64,
                              dilation = 1,
                              sampling_flag = False,
                              p = 0.01)
        # Stage 2: 9 bottleneck blocks 
        self.b20 = bottleNeck(dilation = 1,
                              in_channels = 64,
                              out_channels = 128,
                              sampling_flag = True)
                              
        self.b21 = bottleNeck(dilation = 1,
                              in_channels = 128,
                              out_channels = 128,
                              sampling_flag = False)
        # dilated 2
        self.b22 = bottleNeck(dilation = 2,
                              in_channels = 128,
                              out_channels = 128,
                              sampling_flag = False)
        self.b23 = bottleNeck_as(in_channels = 128,
                                 out_channels = 128)
        self.b24 = bottleNeck(dilation = 4,
                              in_channels = 128,
                              out_channels = 128,
                              sampling_flag = False)
                              
        self.b25 = bottleNeck(dilation = 1,
                              in_channels = 128,
                              out_channels = 128,
                              sampling_flag = False)
                              
        self.b26 = bottleNeck(dilation = 8,
                              in_channels = 128,
                              out_channels = 128,
                              sampling_flag = False)
        self.b27 = bottleNeck_as(in_channels = 128,
                                 out_channels = 128)
        self.b28 = bottleNeck(dilation = 16,
                              in_channels = 128,
                              out_channels = 128,
                              sampling_flag = False)
        # Stage 3: same a Stage 2, without bottleneck2.0
        self.b31 = bottleNeck(dilation = 1,
                              in_channels = 128,
                              out_channels = 128,
                              sampling_flag = False)
                              
        self.b32 = bottleNeck(dilation = 2,
                              in_channels = 128,
                              out_channels = 128,
                              sampling_flag = False)
        self.b33 = bottleNeck_as(in_channels = 128,
                                 out_channels = 128)
        self.b34 = bottleNeck(dilation = 4,
                              in_channels = 128,
                              out_channels = 128,
                              sampling_flag = False)
                              
        self.b35 = bottleNeck(dilation = 1,
                              in_channels = 128,
                              out_channels = 128,
                              sampling_flag = False)
                              
        self.b36 = bottleNeck(dilation = 8,
                              in_channels = 128,
                              out_channels = 128,
                              sampling_flag = False)
        self.b37 = bottleNeck_as(in_channels = 128,
                                 out_channels = 128)
        self.b38 = bottleNeck(dilation = 16,
                              in_channels = 128,
                              out_channels = 128,
                              sampling_flag = False)
    def forward(self, x):
        # The initial block
        x = self.init(x)
        # Stage 1: 5 bottleneck blocks
        x, ind1 = self.b10(x)
        x = self.b11(x)
        x = self.b12(x)
        x = self.b13(x)
        x = self.b14(x)
        # Stage 2: 9 bottleneck blocks
        x, ind2 = self.b20(x)
        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)
        x = self.b24(x)
        x = self.b25(x)
        x = self.b26(x)
        x = self.b27(x)
        x = self.b28(x)
        
        # Stage 3: same as Stage 2 without bottleneck2.0
        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)
        x = self.b35(x)
        x = self.b36(x)
        x = self.b37(x)
        x = self.b38(x)
        # Stage 4: upsampling bottleneck
        
        return x

if __name__=="__main__":
    print("Hello ENet for semantic segmentation")
    C = 3
    ENet = ENet(C)
    images = torch.rand(6, 3, 512, 512)
    output = ENet(images)
    print(output.shape)