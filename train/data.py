

import torch
from PIL import Image, ImageFont, ImageDraw
import random

class ChineseFontDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 ttf_path, 
                 font_size=24, 
                 image_size=(32, 32), 
                 brightness_range=(220, 255),
                 transform=None):
        self.ttf_path = ttf_path
        self.font_size = font_size
        self.image_size = image_size
        self.brightness_range = brightness_range
        self.transform = transform
        
        self.font = ImageFont.truetype(font=self.ttf_path, size=self.font_size)
        self.code_range = [0x4e00, 0x9fbf - 25]
    
    def text_to_image(self, text):
        img = Image.new(size=self.image_size, mode='L')
        draw = ImageDraw.Draw(img)
        pos = (random.randint(0, self.image_size[0] - self.font_size), 
               random.randint(0, self.image_size[1] - self.font_size))
        brightness = random.randint(*self.brightness_range)
        draw.text(pos, text, brightness, font=self.font)
        return img
    
    def __getitem__(self, idx):
        text = chr(idx + self.code_range[0])
        img = self.text_to_image(text)
        
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return self.code_range[1] - self.code_range[0]

