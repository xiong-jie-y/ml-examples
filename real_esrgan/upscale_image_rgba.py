import onnxruntime
import torch
from PIL import Image
import numpy as np
import cv2
import time
from basicsr.archs.rrdbnet_arch import RRDBNet
import click

class SimpleRealESRGAN:
    def __init__(self, onnx=False):
        self.onnx = onnx
        if onnx:
            self.session = onnxruntime.InferenceSession(
                "RealESRGAN_x4plus_anime_6B.onnx", providers=["CUDAExecutionProvider"])
            self.device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            loadnet = torch.load("RealESRGAN_x4plus_anime_6B.pth")
            if 'params_ema' in loadnet:
                keyname = 'params_ema'
            else:
                keyname = 'params'
            model.load_state_dict(loadnet[keyname], strict=True)
            model.eval()
            self.model = model.to(device)
            self.device = device

    def upscale_image(self, np_image_rgb):
        np_image_rgb = cv2.cvtColor(np_image_rgb, cv2.COLOR_RGB2BGR)
        image_rgb_tensor = torch.tensor(np_image_rgb.astype(np.float32)).to(self.device)
        image_rgb_tensor /= 255
        image_rgb_tensor = image_rgb_tensor.permute(2, 0, 1)
        image_rgb_tensor = image_rgb_tensor.unsqueeze(0)

        if self.onnx:
            output_img = torch.tensor(self.session.run([], {"image.1":  np_image_rgb.cpu().numpy()})[0])
        else:
            output_img = self.model(image_rgb_tensor)

        output_img = output_img.data.squeeze().float().clamp_(0, 1)
        output_img = output_img.permute((1, 2, 0))
        output = (output_img * 255.0).round().cpu().numpy().astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output

    def upscale_rgba_image(self, np_image_rgba):
        s = time.time()
        upscaled_rgb = self.upscale_image(np_image_rgba[:, :, 0:3])
        upscaled_alpha = np.expand_dims(
            cv2.cvtColor(
            self.upscale_image(
                cv2.cvtColor(np_image_rgba[:, :, 3], cv2.COLOR_GRAY2RGB)), cv2.COLOR_RGB2GRAY), axis=2)
        output = np.concatenate((upscaled_rgb, upscaled_alpha), axis=2)
        pil_image = Image.fromarray(output, mode="RGBA")
        print(time.time() - s)  
        return pil_image

@click.command()
@click.option("--onnx", is_flag=True)
def main(onnx):
    upscaler = SimpleRealESRGAN(onnx)
    np_image = np.array(Image.open("test.png"))
    pil_image = upscaler.upscale_rgba_image(np_image)
    pil_image.show()

if __name__ == "__main__":
    main()