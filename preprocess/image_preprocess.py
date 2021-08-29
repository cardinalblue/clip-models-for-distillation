import os
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image

image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

image_transform = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize(image_mean, image_std)
])

def png_to_jpg(png):
    jpg = Image.new('RGB', png.size, (255, 255, 255))
    jpg.paste(png, mask=png.split()[3])
    return jpg

def imagename_to_image(imagename, convert_to_rgb=True):
    extension = os.path.splitext(imagename)[1]
    if extension.lower() in ['.png', '.l']:
        img = Image.open(imagename).convert('RGBA')
        if convert_to_rgb:
            img = png_to_jpg(img)
    else:
        img = Image.open(imagename).convert('RGB')
    return img

def image_to_tensor(image):
    return image_transform(image).unsqueeze(0)

if __name__ == '__main__':
    imagename = 'cat.jpg'
    image = imagename_to_image(imagename)
    tensor = image_to_tensor(image)
    print(tensor.size())