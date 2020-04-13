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
    #  dilation - if True: creating dilation bottleneck
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
        
        return x


if __name__=="__main__":
    print("Hello ENet for semantic segmentation")
    C = 3
    ENet = ENet(C)
    images = torch.rand(6, 3, 512, 512)
    output = ENet(images)
    print(output.shape)