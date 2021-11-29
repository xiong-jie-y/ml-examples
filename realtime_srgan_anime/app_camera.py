import torch
import numpy as np
import gradio
from huggingface_hub import hf_hub_download
from basicsr.archs.srresnet_arch import MSRResNet

class SimpleRealUpscaler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MSRResNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=6, upscale=4)
        path = hf_hub_download("xiongjie/realtime-SRGAN-for-anime", filename="SRGAN_x4plus_anime.pth")
        loadnet = torch.load(path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(self.device)

    def upscale(self, np_image_rgb):
        image_rgb_tensor = torch.tensor(np_image_rgb[:,:,::-1].astype(np.float32)).to(self.device)
        image_rgb_tensor /= 255
        image_rgb_tensor = image_rgb_tensor.permute(2, 0, 1)
        image_rgb_tensor = image_rgb_tensor.unsqueeze(0)
        output_img = self.model(image_rgb_tensor)
        output_img = output_img.data.squeeze().float().clamp_(0, 1)
        output_img = output_img.permute((1, 2, 0))
        output = (output_img * 255.0).round().cpu().numpy().astype(np.uint8)
        return output[:, :, ::-1]


import numpy as np
import cv2 as cv


cap = cv.VideoCapture(0)
upscaler = SimpleRealUpscaler()

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    print(frame.shape)
    frame = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    frame = upscaler.upscale(frame)

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
