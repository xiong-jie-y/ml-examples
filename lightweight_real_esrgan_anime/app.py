import gradio
from huggingface_hub import hf_hub_download
import onnxruntime
from PIL import Image
import numpy as np

path = hf_hub_download("xiongjie/lightweight-real-ESRGAN-anime", filename="RealESRGAN_x4plus_anime_4B32F.onnx")
session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])

def upscale(np_image_rgb):
    # From RGB to BGR
    np_image_bgr = np_image_rgb[:, :, ::-1]
    np_image_bgr = np_image_bgr.astype(np.float32)
    np_image_bgr /= 255
    np_image_bgr = np.transpose(np_image_bgr, (2, 0, 1))
    np_image_bgr = np.expand_dims(np_image_bgr, axis=0)
    output_img = session.run([], {"image.1":  np_image_bgr})[0]
    output_img = np.squeeze(output_img, axis=0).astype(np.float32).clip(0, 1)
    output_img = np.transpose(output_img, (1, 2, 0))
    output = (output_img * 255.0).astype(np.uint8)
    # From BGR to RGB
    output = output[:, :, ::-1]
    
    return output

css = ".output_image {height: 100% !important; width: 100% !important;}"
inputs = gradio.inputs.Image()
outputs = gradio.outputs.Image()
gradio.Interface(fn=upscale, inputs=inputs, outputs=outputs, css=css).launch()