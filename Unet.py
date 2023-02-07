import torch

class SimpleUnet(torch.nn.Module):
    r"""
    """

    def __init__(self):
        super().__init__()

        self.pool2d = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()

        encoder_sizes = {
                'encoder_init' : (1,1,3),
                'encoder01' : (1,5,3),
                'encoder02' : (5,10,3),
                'encoder03' : (10,20,3),
                'encoder04' : (20,40,3)
                }
        
        base_size = (40,40,3)

        decoder_sizes = {
                'decoder_outp' : (1,1,3),
                'decoder01' : (10,1,3),
                'decoder02' : (20,5,3),
                'decoder03' : (40,10,3),
                'decoder04' : (80,20,3)
                }

        self.encoder_init = torch.nn.Conv2d(*encoder_sizes['encoder_init'], padding=1)
        self.encoder01 = torch.nn.Conv2d(*encoder_sizes['encoder01'], padding=1)
        self.encoder02 = torch.nn.Conv2d(*encoder_sizes['encoder02'], padding=1)
        self.encoder03 = torch.nn.Conv2d(*encoder_sizes['encoder03'], padding=1)
        self.encoder04 = torch.nn.Conv2d(*encoder_sizes['encoder04'], padding=1)
        self.base = torch.nn.Conv2d(*base_size, padding=1)
        self.decoder04 = torch.nn.ConvTranspose2d(*decoder_sizes['decoder04'], stride=2, padding=1, output_padding=1)
        self.decoder03 = torch.nn.ConvTranspose2d(*decoder_sizes['decoder03'], stride=2, padding=1, output_padding=1)
        self.decoder02 = torch.nn.ConvTranspose2d(*decoder_sizes['decoder02'], stride=2, padding=1, output_padding=1)
        self.decoder01 = torch.nn.ConvTranspose2d(*decoder_sizes['decoder01'], stride=2, padding=1, output_padding=1)
        self.decoder_outp = torch.nn.Conv2d(*decoder_sizes['decoder_outp'], padding=1)

    def forward(self, x):
        r"""
        Inputs:
            :x: (B,C,H,W)
        """

        relu = self.relu
        pool = self.pool2d

        out_down_0 = relu(self.encoder_init(x))
        out_down_1 = relu(pool(self.encoder01(out_down_0)))
        out_down_2 = relu(pool(self.encoder02(out_down_1)))
        out_down_3 = relu(pool(self.encoder03(out_down_2)))
        out_down_4 = relu(pool(self.encoder04(out_down_3)))       

        out_base = relu(self.base(out_down_4))
        
        out_up_4 = relu(self.decoder04(torch.cat([ out_base,out_down_4 ], dim=1)))
        out_up_3 = relu(self.decoder03(torch.cat([ out_up_4,out_down_3 ], dim=1)))
        out_up_2 = relu(self.decoder02(torch.cat([ out_up_3,out_down_2 ], dim=1)))
        out_up_1 = relu(self.decoder01(torch.cat([ out_up_2,out_down_1 ], dim=1)))
        out_final = self.decoder_outp(out_up_1)

        return out_final
