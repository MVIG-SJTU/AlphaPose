from mxnet.gluon.block import HybridBlock


class PixelShuffle(HybridBlock):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def hybrid_forward(self, F, x):
        f1, f2 = self.upscale_factor, self.upscale_factor
        # (N, f1*f2*C, H, W)
        x = F.reshape(x, (0, -4, -1, f1 * f2, 0, 0))  # (N, C, f1*f2, H, W)
        x = F.reshape(x, (0, 0, -4, f1, f2, 0, 0))    # (N, C, f1, f2, H, W)
        x = F.transpose(x, (0, 1, 4, 2, 5, 3))        # (N, C, H, f1, W, f2)
        x = F.reshape(x, (0, 0, -3, -3))              # (N, C, H*f1, W*f2)
        return x
