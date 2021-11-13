from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import PIL
import numpy as np

class RealESRGANUpscaler:
    def __init__(self):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path="RealESRGAN_x4plus_anime_6B.pth",
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False
        )

    def upscale(self, pil_image):
        output, _ = self.upsampler.enhance(np.array(pil_image), outscale=4)
        pil_image = PIL.Image.fromarray(output, mode='RGBA')

        return pil_image

upscaler = RealESRGANUpscaler()
image = PIL.Image.open("test.png")
pil_image = upscaler.upscale(image)
pil_image.show()