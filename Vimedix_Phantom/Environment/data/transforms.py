import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

class CustomCropTransform:   
    def __init__(self, crop_params):
        self.crop_params = crop_params
    def __call__(self, img):
        return transforms.functional.crop(img, *self.crop_params)

class UltrasoundEnhanceTransform:
    def __init__(self, clahe_clip=2.0, denoise_strength=5):
        self.clahe_clip = clahe_clip
        self.denoise_strength = denoise_strength
        
    def __call__(self, image):
        np_image = np.array(image)    
        if len(np_image.shape) == 2 or np_image.shape[2] == 1:  # Grayscale
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
            enhanced = clahe.apply(np_image.astype(np.uint8))
        else:  # RGB
            enhanced = np.zeros_like(np_image)
            for i in range(3):
                clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
                enhanced[:, :, i] = clahe.apply(np_image[:, :, i].astype(np.uint8))   
        # Apply speckle noise reduction
        denoised = cv2.medianBlur(enhanced, self.denoise_strength)
        # Convert back to PIL image
        return Image.fromarray(denoised)


def get_ultrasound_transform(img_size=128, 
                           crop_params=None,
                           clahe_clip=3.0, 
                           denoise_strength=3,
                           brightness=0.1, 
                           contrast=0.2,
                           noise_std=0.03,
                           use_horizontal_flip=False,
                           use_affine=False):

    transform_list = []
    
    if crop_params is not None:
        transform_list.append(CustomCropTransform(crop_params))
    
    transform_list.extend([
        UltrasoundEnhanceTransform(clahe_clip=clahe_clip, denoise_strength=denoise_strength),
        transforms.Resize((img_size, img_size)),
    ])
    
    if use_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.3))
    
    if use_affine:
        transform_list.append(
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))
        )
    
    # Color adjustments
    transform_list.append(
        transforms.ColorJitter(brightness=brightness, contrast=contrast)
    )
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
    ])
    
    if noise_std > 0:
        transform_list.append(
            transforms.Lambda(lambda x: x + torch.randn_like(x) * noise_std)
        )
    
    # Normalize to [-1, 1] range
    transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    
    return transforms.Compose(transform_list)


def get_basic_transform(img_size=128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5,), (0.5,))
    ])


def get_test_transform(img_size=128):
    return get_ultrasound_transform(
        img_size         = img_size,
        clahe_clip       = 3.0,     
        denoise_strength = 3,          
        brightness       = 0.0,       
        contrast         = 0.0,        
        noise_std        = 0.0,      
        use_horizontal_flip = False,
        use_affine          = False,
    )


def get_default_data_transform(img_size=128):
    return get_ultrasound_transform(
        img_size=img_size,
        clahe_clip=3.0,
        denoise_strength=3,
        brightness=0.1,
        contrast=0.2,
        noise_std=0.03,
        use_horizontal_flip=False,
        use_affine=False
    )

data_transform = get_default_data_transform()
