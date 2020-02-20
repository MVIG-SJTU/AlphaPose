from torch import nn


def padding_same_conv2d(input_size, in_c, out_c, kernel_size=4, stride=1):
    output_size = input_size // stride
    padding_num = stride * (output_size - 1) - input_size + kernel_size
    if padding_num % 2 == 0:
        return nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding_num // 2, bias=False))
    else:
        return nn.Sequential(
            nn.ConstantPad2d((padding_num // 2, padding_num // 2 + 1, padding_num // 2, padding_num // 2 + 1), 0),
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        )

class resBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=4, stride=1, input_size=None):
        super().__init__()
        assert kernel_size == 4
        self.shortcut = lambda x: x
        if in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)

        main_layers = [
            nn.Conv2d(in_c, out_c // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c // 2, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True),
        ]

        main_layers.extend([
            *padding_same_conv2d(input_size, out_c // 2, out_c // 2, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_c // 2, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True)])

        main_layers.extend(
            padding_same_conv2d(input_size, out_c // 2, out_c, kernel_size=1, stride=1)
        )
        self.main = nn.Sequential(*main_layers)
        self.activate = nn.Sequential(
            nn.BatchNorm2d(out_c, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        shortcut_x = self.shortcut(x)
        main_x = self.main(x)
        x = self.activate(shortcut_x + main_x)
        return x


class upBlock(nn.Module):
    def __init__(self, in_c, out_c, conv_num=2):
        super().__init__()
        additional_conv = []
        layer_length = 4

        for i in range(1, conv_num+1):
            additional_conv += [
                nn.ConstantPad2d((2, 1, 2, 1), 0),
                nn.ConvTranspose2d(out_c, out_c, kernel_size=4, stride=1, padding=3, bias=False),
                nn.BatchNorm2d(out_c, eps=0.001, momentum=0.001),
                nn.ReLU(inplace=True)
            ]
        self.main = nn.Sequential(
            # nn.ConstantPad2d((0, 1, 0, 1), 0),
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True),
            *additional_conv
            )

    def forward(self, x):
        x = self.main(x)
        return x

class PRNet(nn.Module):
    def __init__(self, in_channel, out_channel=3):
        super().__init__()
        size = 16
        self.input_conv = nn.Sequential( #*[
            *padding_same_conv2d(256, in_channel, size, kernel_size=4, stride=1),  # 256x256x16
            nn.BatchNorm2d(size, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True)
            # ]
        ) 
        self.down_conv_1 = resBlock(size, size * 2, kernel_size=4, stride=2, input_size=256)  # 128x128x32
        self.down_conv_2 = resBlock(size * 2, size * 2, kernel_size=4, stride=1, input_size=128)  # 128x128x32
        self.down_conv_3 = resBlock(size * 2, size * 4, kernel_size=4, stride=2, input_size=128)  # 64x64x64
        self.down_conv_4 = resBlock(size * 4, size * 4, kernel_size=4, stride=1, input_size=64)  # 64x64x64
        self.down_conv_5 = resBlock(size * 4, size * 8, kernel_size=4, stride=2, input_size=64)  # 32x32x128
        self.down_conv_6 = resBlock(size * 8, size * 8, kernel_size=4, stride=1, input_size=32)  # 32x32x128
        self.down_conv_7 = resBlock(size * 8, size * 16, kernel_size=4, stride=2, input_size=32)  # 16x16x256
        self.down_conv_8 = resBlock(size * 16, size * 16, kernel_size=4, stride=1, input_size=16)  # 16x16x256
        self.down_conv_9 = resBlock(size * 16, size * 32, kernel_size=4, stride=2, input_size=16)  # 8x8x512
        self.down_conv_10 = resBlock(size * 32, size * 32, kernel_size=4, stride=1, input_size=8)  # 8x8x512

        self.center_conv = nn.Sequential(
            nn.ConstantPad2d((2, 1, 2, 1), 0),
            nn.ConvTranspose2d(size * 32, size * 32, kernel_size=4, stride=1, padding=3, bias=False),  # 8x8x512
            nn.BatchNorm2d(size * 32, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True)
        )

        self.up_conv_5 = upBlock(size * 32, size * 16)  # 16x16x256
        self.up_conv_4 = upBlock(size * 16, size * 8)  # 32x32x128
        self.up_conv_3 = upBlock(size * 8, size * 4)  # 64x64x64

        self.up_conv_2 = upBlock(size * 4, size * 2, 1)  # 128x128x32
        self.up_conv_1 = upBlock(size * 2, size, 1)  # 256x256x16

        self.output_conv = nn.Sequential(
            nn.ConstantPad2d((2, 1, 2, 1), 0),
            nn.ConvTranspose2d(size, 3, kernel_size=4, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True),

            nn.ConstantPad2d((2, 1, 2, 1), 0),
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True),

            nn.ConstantPad2d((2, 1, 2, 1), 0),
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.001),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.down_conv_1(x)
        x = self.down_conv_2(x)
        x = self.down_conv_3(x)
        x = self.down_conv_4(x)
        x = self.down_conv_5(x)
        x = self.down_conv_6(x)
        x = self.down_conv_7(x)
        x = self.down_conv_8(x)
        x = self.down_conv_9(x)
        x = self.down_conv_10(x)

        x = self.center_conv(x)

        x = self.up_conv_5(x)
        x = self.up_conv_4(x)
        x = self.up_conv_3(x)
        x = self.up_conv_2(x)
        x = self.up_conv_1(x)
        x = self.output_conv(x)
        return x
