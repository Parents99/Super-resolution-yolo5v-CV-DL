import torch
import torch.nn as nn

ngpu = 1
ndf = 64

class Discriminator(nn.Module):
    def __init__(self,ngpu,ndf):
        super(Discriminator,self).__init__()
        self.ndf = ndf
        self.ngpu = ngpu

        self.model = nn.Sequential(
            nn.Conv2d(26, ndf, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf , ndf * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, ndf * 16, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 16, 1, 1, 1, 0, bias=False),
            #nn.AdaptiveAvgPool2d([1,1]),
            nn.Sigmoid()
        )

        self.avgPool = nn.AdaptiveAvgPool2d([1,1])

        self.pre_conv_1d_52 = nn.Conv2d(52,26,1)
        self.pre_conv_1d_13 = nn.Conv2d(13,26,1)

    def forward(self,input):
        input_channel = list(input.size())[1]
        if(input_channel == 52):
            input = self.pre_conv_1d_52(input)
        if(input_channel == 13):
            input = self.pre_conv_1d_13(input)

        out = self.model(input)
        print(out.size())

        return self.avgPool(out)

    
    
input = torch.randn(1,13,13,512)

model = Discriminator(ngpu,ndf)

output = model(input)

print(output)
