import io
import base64
from PIL import Image

def convert_image_to_base64(image_input: Image.Image | bytes | bytearray) -> str:
    """将图片转换为base64格式"""
    if isinstance(image_input, Image.Image):
        if image_input.mode != 'RGB':
            image_input = image_input.convert('RGB')
        img_byte_arr = io.BytesIO()

        image_input.save(img_byte_arr, format='PNG')
        img_binary_data = img_byte_arr.getvalue()
        
    elif isinstance(image_input, (bytes, bytearray)):
        img_binary_data = image_input
    else:
        raise ValueError('数据类型不支持')

    base64_encoded_data = base64.b64encode(img_binary_data).decode('utf-8')
    return f"data:image/png;base64,{base64_encoded_data}"

def convert_base64_to_image(base64_str: str) -> Image.Image:
    """将base64格式的字符串转换为PIL图像对象"""
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image