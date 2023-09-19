import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,ngpu,ngf):
        super(Generator,self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf

        self.input_channel = 256
        self.out_channel = 256
        self.output_size = torch.Size([26,26])
        #self.out_ch = 26

        self.transpose = nn.Sequential(
            nn.ConvTranspose2d(self.input_channel , self.ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 1, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, self.out_channel, 1,1,1, bias = False),
            nn.SiLU(True)
        )

        self.pre_conv1d_512 = nn.Conv2d(512, 256,1)
        self.pre_conv1d_128 = nn.Conv2d(128, 256,1)

        self.post_conv1d_to_512 = nn.Conv2d(256, 512, 1)
        self.post_conv1d_to_128 = nn.Conv2d(256, 128, 1)

    #def pre_conv1d(self,input,channel_in):
    #    return nn.Conv2d(channel_in, self.out_channel,1)(input)
    
    #def post_conv1d(self,input, channel_out):
    #    return nn.Conv2d(self.input_channel, channel_out,1)(input)

    def forward(self,input : torch.Tensor):
        
        #tolgo il batch size
        input_size = list(input.size())
        del input_size[0]
        input_size = torch.Size(input_size)
        output :torch.Size() = None
        reduce = False
        up = False
        #dimensioni(dimezzate) delle feature map dove devo fare le conv 1x1
        #dim1 = torch.Size([40,40,128])
        #dim2 = torch.Size([10,10,512])
        

        if input_size == torch.Size([128,28,28]):
            #self.input_channel = list(input.size())[1]
            #input = self.pre_conv1d(input)
            #in_ch = list(input.size())[1]
            input = self.pre_conv1d_128(input)
            self.output_size = torch.Size([52,52])
            #self.out_channel = 52
            reduce = True

        if input_size == torch.Size([512,7,7]):
            #self.input_channel = list(input.size())[1]
            #input = self.pre_conv1d(input)
            #in_ch = list(input.size())[1]
            input = self.pre_conv1d_512(input)
            self.output_size = torch.Size([13,13])
            #self.out_channel = 20
            up = True
        
        #print(self.output_size)
        #print(input.size()," ciao")
        #print(input.size())
        output = self.transpose(input)
        #print(output.size()," ciao2")

        if(reduce):
            output = self.post_conv1d_to_128(output)
        if(up):
            output = self.post_conv1d_to_512(output)
        #print(output.size())
        #print(self.output_size)
        return nn.Upsample(size = self.output_size)(output)
    

ndf = 64
input = torch.randn(0,128,28,28)
#input : (128,28,28)
#         (256,14,14)
#         (512,7,7)

#target : (128,52,52)
#         (256,26,26)
#         (512,13,13)

model = Generator(1,ndf)



output = model(input)

#print(output.size())