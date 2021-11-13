place test.py in this folder.

```
pip install -r requirements.txt

# Download pth model for anime
gdown https://drive.google.com/uc?id=1cExySdxIOh0mw7XK_P_LZiijDKMx9m-p

# Download onnx model for anime
gdown https://drive.google.com/file/d/1mm7xflWKdCKvCtWDGwh8TUrGItNs-V3I/view?usp=sharing

# There's also onnx model for wide range of images.
# https://github.com/PINTO0309/PINTO_model_zoo/tree/main/133_Real-ESRGAN

# No dependency to realsrgan package.
python upscale_image_rgba.py

# shorter code
python upscale_image.py
```