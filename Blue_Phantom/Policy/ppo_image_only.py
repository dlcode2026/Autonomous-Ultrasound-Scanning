import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import torch.nn.init as init
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet18
from gym import Env
from gym import spaces
import torch.nn.functional as F
import glob
from datetime import datetime
from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, LogSoftmax
from torch import flatten
from torch.distributions import MultivariateNormal,Categorical
import random
import pandas as pd
from collections import deque
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #mps 

#-----------------------------------------------------------------------------
# Classification Model
#-----------------------------------------------------------------------------
class ImageClassifier:
    def __init__(self, model_path, num_classes=6, target_class_idx=None, class_names=None):
        self.device = device
        self.num_classes = num_classes
        self.target_class_idx = target_class_idx
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        self.model = resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Classification model loaded from: {model_path}")
            print(f"Model has {num_classes} classes: {self.class_names}")
        except Exception as e:
            print(f"Error loading classification model: {e}")
            raise

        self.preprocess = transforms.Compose([
          transforms.Resize((128, 128)),
          transforms.ToTensor(),
          transforms.Grayscale(num_output_channels=1),
          transforms.Normalize((0.5,), (0.5,))
      ])

    def classify_image(self, image):
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Grayscale
                    image = Image.fromarray(image.astype(np.uint8))
                else:
                    image = Image.fromarray(image.astype(np.uint8)).convert('L')

            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            predicted_class_idx = predicted.item()
            confidence_score = confidence.item()
            class_probabilities = probabilities.squeeze().cpu().numpy()

            return predicted_class_idx, confidence_score, class_probabilities

        except Exception as e:
            print(f"Error in image classification: {e}")
            return 0, 0.0, np.zeros(self.num_classes)

    def is_target_class(self, image, confidence_threshold=0.95):
        if self.target_class_idx is None:
            raise ValueError("Target class index not set")
        predicted_class_idx, confidence, _ = self.classify_image(image)
        is_target = (predicted_class_idx == self.target_class_idx) and (confidence >= confidence_threshold)
        return is_target, confidence

    def get_class_name(self, class_idx):
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return f"Unknown_Class_{class_idx}"

class ImageGradeRegressor:
    def __init__(self, model_path):
        self.device = device
        self.model = resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Regression model loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading regression model: {e}")
            raise

        self.preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict_grade(self, image):
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Grayscale
                    image = Image.fromarray(image.astype(np.uint8))
                else:
                    image = Image.fromarray(image.astype(np.uint8)).convert('L')

            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                grade_output = self.model(image_tensor)
                predicted_grade = grade_output.item()

            predicted_grade = max(0.0, min(3.0, predicted_grade))
            return predicted_grade

        except Exception as e:
            print(f"Error in grade prediction: {e}")
            return 1.0

#-----------------------------------------------------------------------------
# VAE-GAN Model
#-----------------------------------------------------------------------------
class Encoder(torch.nn.Module):
    def __init__(self, img_channels=1, latent_dim=100, label_dim=3):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv1 = torch.nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1)  
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)    
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)   
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) 
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) 
        self.conv6 = torch.nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)  

        self.fc_mu = torch.nn.Linear(512 * 2 * 2 + label_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(512 * 2 * 2 + label_dim, latent_dim)

        self.label_encoder = torch.nn.Sequential(
            torch.nn.Linear(label_dim, 64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(0.2)
        )

    def forward(self, x, labels):
        x = torch.nn.functional.leaky_relu(self.conv1(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.conv2(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.conv3(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.conv4(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.conv5(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.conv6(x), 0.2)
        x = x.view(x.size(0), -1)

        label_features = self.label_encoder(labels)
        combined = torch.cat([x, labels], dim=1)

        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z


class Generator(torch.nn.Module):
    def __init__(self, latent_dim=100, label_dim=12, img_channels=1):
        super(Generator, self).__init__()

        self.fc = torch.nn.Linear(latent_dim + label_dim, 512 * 2 * 2)

        self.deconv1 = torch.nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)  
        self.deconv2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  
        self.deconv3 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  
        self.deconv4 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   
        self.deconv5 = torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    
        self.deconv6 = torch.nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1)  

        self.bn1 = torch.nn.BatchNorm2d(512)
        self.bn2 = torch.nn.BatchNorm2d(256)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.bn5 = torch.nn.BatchNorm2d(32)

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        x = self.fc(x)
        x = x.view(-1, 512, 2, 2)

        x = torch.nn.functional.relu(self.bn1(self.deconv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.deconv2(x)))
        x = torch.nn.functional.relu(self.bn3(self.deconv3(x)))
        x = torch.nn.functional.relu(self.bn4(self.deconv4(x)))
        x = torch.nn.functional.relu(self.bn5(self.deconv5(x)))
        x = torch.tanh(self.deconv6(x))

        return x


class DiscriminatorWithFeatures(torch.nn.Module):
    def __init__(self, img_channels=1, label_dim=12):
        super(DiscriminatorWithFeatures, self).__init__()

        self.conv1 = torch.nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1)  
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)            
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)           
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)         
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)         
        self.conv6 = torch.nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)      

        self.label_embedding = torch.nn.Sequential(
            torch.nn.Linear(label_dim, 512),
            torch.nn.LeakyReLU(0.2)
        )

        self.fc = torch.nn.Linear(512 * 2 * 2 + 512, 1)

    def forward(self, x, labels, return_features=False):
        x1 = torch.nn.functional.leaky_relu(self.conv1(x), 0.2)
        x2 = torch.nn.functional.leaky_relu(self.conv2(x1), 0.2)
        x3 = torch.nn.functional.leaky_relu(self.conv3(x2), 0.2)
        x4 = torch.nn.functional.leaky_relu(self.conv4(x3), 0.2)
        x5 = torch.nn.functional.leaky_relu(self.conv5(x4), 0.2)
        x6 = torch.nn.functional.leaky_relu(self.conv6(x5), 0.2)

        x_flat = x6.view(x6.size(0), -1)

        label_embedding = self.label_embedding(labels)
        combined = torch.cat([x_flat, label_embedding], dim=1)
        validity = self.fc(combined)

        if return_features:
            return validity, [x1, x2, x3, x4, x5, x6]
        return validity


class ModifiedVAEGAN(torch.nn.Module):
    def __init__(self, latent_dim=100, label_dim=12, img_channels=1):
        super(ModifiedVAEGAN, self).__init__()

        self.latent_dim = latent_dim
        self.label_dim = label_dim

        self.encoder = Encoder(img_channels, latent_dim, label_dim)
        self.generator = Generator(latent_dim, label_dim, img_channels)
        self.discriminator = DiscriminatorWithFeatures(img_channels, label_dim)

    def encode(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.encoder.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z, labels):
        return self.generator(z, labels)

    def forward(self, x, labels):
        z, mu, logvar = self.encode(x, labels)
        x_recon = self.decode(z, labels)
        return x_recon, mu, logvar

    def sample(self, n_samples, labels):
        z = torch.randn(n_samples, self.latent_dim).to(labels.device)
        return self.generator(z, labels)


class UltrasoundImageGenerator:
    def __init__(self, model_path, norm_params_path, device='cuda'):
        self.device = torch.device(device)
        self.model = ModifiedVAEGAN(latent_dim=100, label_dim=12, img_channels=1).to(self.device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict']

            cleaned_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("._orig_mod", "")  
                cleaned_state_dict[new_key] = value

            self.model.load_state_dict(cleaned_state_dict)

            self.model.eval()
            print(f"VAE-GAN model loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        try:
            self.norm_params = dict(np.load(norm_params_path, allow_pickle=True))
            print(f"Normalization parameters loaded from: {norm_params_path}")
        except Exception as e:
            print(f"Error loading normalization parameters: {e}")
            raise

        self.fixed_z = torch.randn(1, self.model.latent_dim).to(self.device)
        self.model.eval()
        print(f"Ultrasound image generator initialized")

        self.fixed_z = torch.randn(1, self.model.latent_dim).to(self.device)
        self.model.eval()
        print(f"Ultrasound image generator initialized")

    def min_max_normalize(self, value, min_val, max_val, eps=1e-8):
        return (value - min_val) / (max_val - min_val + eps)

    def normalize_label(self, raw_label):
        normalized_values = np.zeros_like(raw_label, dtype=np.float32)
        feature_groups = self.norm_params['feature_groups'].item()

        for group_name, indices in feature_groups.items():
            for i in indices:
                if i >= len(raw_label):
                    continue

                if group_name == 'forces':
                    value = (raw_label[i] - self.norm_params['median'][i]) / (self.norm_params['iqr'][i] + 1e-8)
                    # Scale to [0, 1] range
                    normalized_values[i] = (value + 2) / 4

                elif group_name == 'position':
                    normalized_values[i] = self.min_max_normalize(
                        raw_label[i],
                        self.norm_params['min'][i],
                        self.norm_params['max'][i]
                    )

                elif group_name == 'rotation':
                    normalized_values[i] = 2 * self.min_max_normalize(
                        raw_label[i],
                        self.norm_params['min'][i],
                        self.norm_params['max'][i]
                    ) - 1

        return torch.tensor(normalized_values, dtype=torch.float32)

    def generate_image(self, x, y, z, rx, ry, rz):
        with torch.no_grad():
            label_values = np.zeros(12, dtype=np.float32)
            
            label_values[0] = 2.58
            label_values[1] = 1.29
            label_values[2] = -11.0

            label_values[3] = -0.98
            label_values[4] = 0.89
            label_values[5] = 0.26

            label_values[6] = x
            label_values[7] = y
            label_values[8] = z

            label_values[9] = rx
            label_values[10] = ry
            label_values[11] = rz

            try:
                normalized_label = self.normalize_label(label_values)
                normalized_label = normalized_label.to(self.device).unsqueeze(0) 
            except Exception as e:
                print(f"Error normalizing label: {e}")
                normalized_label = torch.zeros((1, 12), dtype=torch.float32, device=self.device)

            try:
                generated_img = self.model.generator(self.fixed_z, normalized_label)
            except Exception as e:
                print(f"Error generating image: {e}")
                return np.zeros((128, 128), dtype=np.uint8)

            try:
                if isinstance(generated_img, torch.Tensor):
                    img_np = generated_img.squeeze().cpu().numpy()
                else:
                    img_np = generated_img

                # Denormalize from [-1, 1] to [0, 1]
                img_np = (img_np + 1) / 2.0

                # Convert to uint8 range [0, 255]
                img_np = (img_np * 255).astype(np.uint8)

                return img_np

            except Exception as e:
                print(f"Error in image post-processing: {e}")
                # Return a blank image as fallback
                return np.zeros((128, 128), dtype=np.uint8)

# Model paths
MODEL_PATH = 'vaegan.pth'
NORM_PARAMS_PATH = 'normalization_params.npz'
CLASSIFICATION_MODEL_PATH = 'classification.pth'
GRADER_MODEL_PATH = 'regression.pth'

# Step sizes for movement
us_step_x = 0.0270
us_step_y = 0.0065
us_step_z = 0.0045
us_step_rx = 0.314
us_step_ry = 0.1210
us_step_rz = 0.8620

image_generator = UltrasoundImageGenerator(MODEL_PATH, NORM_PARAMS_PATH, device=str(device))
classifier = ImageClassifier(
    model_path=CLASSIFICATION_MODEL_PATH,
    num_classes=6,
    target_class_idx=3,
    class_names=['A4C', 'PSMV', 'Random', 'SC', 'PLAX', 'PSAV']
)
        
#-----------------------------------------------------------------------------
# IMAGE ONLY BASED ENVIRONMENT
#-----------------------------------------------------------------------------

class ImageOnlyEnv(Env):
    def __init__(self, classifier, grader_model_path=None, use_rotation=True):
        super().__init__()
        self.classifier = classifier
        self.use_rotation = use_rotation
        
        self.steps_in_episode = 0
        self.action_history = []
        self.previous_sc_probability = 0.0
        self.previous_grade = 0.0
        
        self.initial_position = (0.55, 0.01, 0.08)
        self.initial_rotation = (2.19, -0.19, -0.80)
        self.reset_state()
        
        self.grader = None
        if grader_model_path:
            self.grader = ImageGradeRegressor(grader_model_path)
        
        # Action space: 12 actions (6 translation + 6 rotation)
        self.action_space = spaces.Discrete(12 if use_rotation else 6)
        
        # Observation space: Image only (1, 128, 128)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1, 128, 128), dtype=np.float32)
        
        self.action_names = [
            "move_x_negative", "move_x_positive",
            "move_y_negative", "move_y_positive", 
            "move_z_negative", "move_z_positive",
            "rotate_rx_negative", "rotate_rx_positive",
            "rotate_ry_negative", "rotate_ry_positive",
            "rotate_rz_negative", "rotate_rz_positive"
        ]
    
    def reset_state(self):
        self.current_position = self.initial_position
        self.current_rotation = self.initial_rotation
        self.steps_in_episode = 0
        self.action_history = []
        self.previous_sc_probability = 0.0
        self.previous_grade = 0.0
    
    def _preprocess_image(self, image):
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Grayscale
                    pil_image = Image.fromarray(image.astype(np.uint8))
                else:
                    pil_image = Image.fromarray(image.astype(np.uint8)).convert('L')
            else:
                pil_image = image
            
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
            
            image_tensor = transform(pil_image)  # Shape: (1, 128, 128)
            return image_tensor
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return torch.zeros((1, 128, 128), dtype=torch.float32)
    
    def _get_state(self, position, rotation, image):
        processed_image = self._preprocess_image(image)
        return processed_image.numpy()
    
    def step(self, action):
        self.steps_in_episode += 1
        
        action_name = self.action_names[action] if action < len(self.action_names) else f"unknown_action_{action}"
        self.action_history.append({
            'step': self.steps_in_episode,
            'action_id': action,
            'action_name': action_name,
            'position_before': self.current_position,
            'rotation_before': self.current_rotation
        })
        
        new_position, new_rotation = self._apply_action(action)
        
        new_image = image_generator.generate_image(*new_position, *new_rotation)
        
        observation = self._get_state(new_position, new_rotation, new_image)
        
        reward, done = self._calculate_reward(new_image, new_position, new_rotation)
        
        self.current_position = new_position
      
        self.current_rotation = new_rotation

        self.action_history[-1]['position_after'] = new_position
        self.action_history[-1]['rotation_after'] = new_rotation
        
        info = {
            "position": new_position,
            "rotation": new_rotation,
            "steps": self.steps_in_episode,
            "last_action": action_name,
            "action_history": self.action_history.copy(),
            "state_type": "image_only"
        }
        
        return observation, reward, done, info
        
    def _apply_action(self, action):
        x, y, z = self.current_position
        rx, ry, rz = self.current_rotation
  
        original_x, original_y, original_z = x, y, z
        original_rx, original_ry, original_rz = rx, ry, rz
        
        if action == 0:    # move_x_negative
            x = x - us_step_x
        elif action == 1:  # move_x_positive
            x = x + us_step_x
        elif action == 2:  # move_y_negative
            y = y - us_step_y
        elif action == 3:  # move_y_positive
            y = y + us_step_y
        elif action == 4:  # move_z_negative
            z = z - us_step_z
        elif action == 5:  # move_z_positive
            z = z + us_step_z
        
        elif self.use_rotation and action == 6:   # rotate_rx_negative
            rx = rx - us_step_rx
        elif self.use_rotation and action == 7:   # rotate_rx_positive
            rx = rx + us_step_rx
        elif self.use_rotation and action == 8:   # rotate_ry_negative
            ry = ry - us_step_ry
        elif self.use_rotation and action == 9:   # rotate_ry_positive
            ry = ry + us_step_ry
        elif self.use_rotation and action == 10:  # rotate_rz_negative
            rz = rz - us_step_rz
        elif self.use_rotation and action == 11:  # rotate_rz_positive
            rz = rz + us_step_rz
        
        # BOUNDARY CHECKING
        if (x < 0.4 or x > 0.58 or 
            y < -0.05 or y > 0.08 or 
            z < 0.04 or z > 0.13 or
            rx < -3.14 or rx > 3.14 or
            ry < -0.56 or ry > 0.65 or
            rz < -2.18 or rz > 2.13):
            
            # If out of bounds, revert to original position (no movement)
            print(f"Boundary hit. Action {action} blocked. Staying at current position.")
            x, y, z = original_x, original_y, original_z
            rx, ry, rz = original_rx, original_ry, original_rz
        
        # Apply clamping
        x = max(0.4, min(x, 0.58))
        y = max(-0.05, min(y, 0.08))
        z = max(0.04, min(z, 0.13))
        rx = max(-3.14, min(rx, 3.14))
        ry = max(-0.56, min(ry, 0.65))
        rz = max(-2.18, min(rz, 2.13))
        
        new_position = (round(x, 10), round(y, 10), round(z, 10))
        new_rotation = (round(rx, 10), round(ry, 10), round(rz, 10))
        
        return new_position, new_rotation

        
    def _calculate_reward(self, current_image, position, rotation):
        _, _, class_probs = self.classifier.classify_image(current_image)
        current_sc_probability = class_probs[self.classifier.target_class_idx]
        
        SCALE_FACTOR = 100
      
        current_potential = current_sc_probability * SCALE_FACTOR
        previous_potential = self.previous_sc_probability * SCALE_FACTOR
        shaping_reward = current_potential - previous_potential
        
        base_reward = 0
        grade_shaping_reward = 0
        done = False
        if current_sc_probability >= 0.9:
            base_reward = 20.0
            current_grade = self.grader.predict_grade(current_image)
            grade_shaping_reward =  current_grade - self.previous_grade
            self.previous_grade = current_grade
            if self.grader and current_grade >= 2.0:
                done = True
                base_reward = 50.0
        
        step_penalty = -0.001
        
        total_reward = base_reward + shaping_reward + step_penalty + grade_shaping_reward
        
        self.previous_sc_probability = current_sc_probability
        
        return total_reward, done    
    
    def reset(self):
        print("\n" + "="*80)
        print(f"EPISODE RESET")
        print("="*80)
        
        self.initial_position = (
            random.uniform(0.4, 0.58),    # x between 0.4 and 0.58
            random.uniform(-0.05, 0.08),  # y between -0.05 and 0.08
            random.uniform(0.04, 0.13)    # z between 0.04 and 0.13
        )
        
        self.initial_rotation = (
            random.uniform(-3.14, 3.14), # rx between -3.14 and 3.14
            random.uniform(-0.56, 0.65),  # ry between -0.56 and 0.65
            random.uniform(-2.18, 2.13)    # rz between -2.18 and 2.13
        )
        
        self.reset_state()
        
        initial_image = image_generator.generate_image(*self.initial_position, *self.initial_rotation)
        
        initial_state = self._get_state(self.initial_position, self.initial_rotation, initial_image)
        
        _, _, class_probs = self.classifier.classify_image(initial_image)
        initial_sc = class_probs[self.classifier.target_class_idx]
        self.previous_sc_probability = initial_sc
        
        return initial_state
    
    def get_action_summary(self):
        if not self.action_history:
            return "No actions taken yet"
        
        action_counts = {}
        for action_record in self.action_history:
            action_name = action_record['action_name']
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        summary = f"Episode Summary - {len(self.action_history)} steps:\n"
        for action_name, count in sorted(action_counts.items()):
            summary += f"  {action_name}: {count}x\n"
        
        return summary


#-----------------------------------------------------------------------------
# PPO Policy
#-----------------------------------------------------------------------------

def preprocess_image_state(state):
    if isinstance(state, np.ndarray):
        image_tensor = torch.from_numpy(state).float().unsqueeze(0)
    else:
        image_tensor = state.unsqueeze(0)
        
    image_tensor = image_tensor.to(device)
    
    return image_tensor


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ImageOnlyActor(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
      
        self.cnn_features = nn.Sequential(
  
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        self.cnn_feature_size = 256 * 4 * 4
        
        self.actor_head = nn.Sequential(
            nn.Linear(self.cnn_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, state):
        cnn_features = self.cnn_features(state)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        logits = self.actor_head(cnn_features)
        return F.softmax(logits, dim=-1)


class ImageOnlyCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        self.cnn_feature_size = 256 * 4 * 4
        
        self.critic_head = nn.Sequential(
            nn.Linear(self.cnn_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
    def forward(self, state):
        cnn_features = self.cnn_features(state)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        value = self.critic_head(cnn_features)
        return value


class ImageOnlyActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ImageOnlyActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        if has_continuous_action_space:
            raise NotImplementedError("Continuous actions not implemented")
        else:
            self.actor = ImageOnlyActor(action_dim).to(device)
        
        self.critic = ImageOnlyCritic().to(device)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

def create_image_env(classifier, grader_model_path=None):
    return ImageOnlyEnv(classifier, grader_model_path, use_rotation=True)

#####################################Monitoring########################################

class TrainingMonitor:
    def __init__(self, log_dir, save_freq=100):
        self.log_dir = log_dir
        self.save_freq = save_freq
        
        # Create monitoring directories
        self.plots_dir = os.path.join(log_dir, "plots")
        self.data_dir = os.path.join(log_dir, "monitoring_data")
        self.validation_dir = os.path.join(log_dir, "validation")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.validation_dir, exist_ok=True)
        
        # Training tracking variables
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.max_sc_values = []
        
        # Validation tracking variables
        self.validation_timesteps = []
        self.validation_rewards = []
        self.validation_success_rates = []
        self.validation_episode_lengths = []
        self.validation_max_sc_values = []
        
        # Success tracking
        self.recent_successes = deque(maxlen=100)  # Last 100 episodes        
    
    def debug_validation_status(self):
        if self.validation_timesteps:
            print(f"   Latest validation at timestep: {self.validation_timesteps[-1]}")
            print(f"   Latest validation reward: {self.validation_rewards[-1]:.2f}")
        else:
            print(f"   No validation data available yet")
    
    def log_ppo_update(self, actor_loss, critic_loss, entropy):
        self.actor_losses.append(float(actor_loss))
        self.critic_losses.append(float(critic_loss))
        self.entropies.append(float(entropy))
    
    def log_episode(self, episode_reward, episode_length, success, max_sc):
        self.episode_rewards.append(float(episode_reward))
        self.episode_lengths.append(int(episode_length))
        self.max_sc_values.append(float(max_sc))
        
        self.recent_successes.append(1 if success else 0)
        current_success_rate = sum(self.recent_successes) / len(self.recent_successes)
        self.success_rates.append(float(current_success_rate))
    
    def log_validation(self, timestep, val_rewards, val_success_rate, val_lengths, val_max_sc):
        try:
            self.validation_timesteps.append(timestep)
            self.validation_rewards.append(float(np.mean(val_rewards)))
            self.validation_success_rates.append(float(val_success_rate))
            self.validation_episode_lengths.append(float(np.mean(val_lengths)))
            self.validation_max_sc_values.append(float(np.mean(val_max_sc)))
            
            print(f"   Avg Reward: {np.mean(val_rewards):.2f}")
            print(f"   Success Rate: {val_success_rate:.2%}")
            print(f"   Avg Length: {np.mean(val_lengths):.1f}")
            print(f"   Avg Max SC: {np.mean(val_max_sc):.4f}")
            print(f"   Total validation runs so far: {len(self.validation_timesteps)}")
            
        except Exception as e:
            print(f"Error logging validation data: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_losses(self, update_step):
        if len(self.actor_losses) < 2:
            print(f"Not enough loss data to plot (have {len(self.actor_losses)}, need at least 2)")
            return
        
        try:
            plt.figure(figsize=(12, 8))
            steps = range(len(self.actor_losses))
            
            plt.plot(steps, self.actor_losses, label='Actor Loss', color='blue', alpha=0.7, linewidth=2)
            plt.plot(steps, self.critic_losses, label='Critic Loss', color='red', alpha=0.7, linewidth=2)
            
            plt.title(f'Training Losses - Update Step {update_step}', fontsize=16)
            plt.xlabel('Update Step', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            if len(self.actor_losses) >= 10:
                window = min(10, len(self.actor_losses) // 2)
                actor_ma = pd.Series(self.actor_losses).rolling(window=window).mean()
                critic_ma = pd.Series(self.critic_losses).rolling(window=window).mean()
                plt.plot(steps, actor_ma, color='darkblue', linewidth=3, alpha=0.8, label=f'Actor MA({window})')
                plt.plot(steps, critic_ma, color='darkred', linewidth=3, alpha=0.8, label=f'Critic MA({window})')
                plt.legend(fontsize=12)
            
            plt.tight_layout()
            plot_path = os.path.join(self.plots_dir, f'losses_step_{update_step}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Losses plot saved: {plot_path}")
            
        except Exception as e:
            print(f"Error generating losses plot: {e}")
            plt.close()
    
    def plot_average_reward(self, update_step):
        if len(self.episode_rewards) < 2:  # Reduced minimum requirement
            print(f"Not enough reward data to plot (have {len(self.episode_rewards)}, need at least 2)")
            return
        
        try:
            plt.figure(figsize=(12, 8))
            episodes = range(len(self.episode_rewards))
          
            plt.plot(episodes, self.episode_rewards, alpha=0.3, color='lightblue', label='Episode Rewards')
            
            window = min(20, len(self.episode_rewards) // 2) 
            if window >= 2:
                moving_avg = pd.Series(self.episode_rewards).rolling(window=window).mean()
                plt.plot(episodes, moving_avg, color='blue', linewidth=3, label=f'Moving Average ({window} episodes)')

            if self.validation_timesteps and self.validation_rewards:
                val_episodes = [ts // 200 for ts in self.validation_timesteps]
                plt.scatter(val_episodes, self.validation_rewards, color='red', s=100, 
                           marker='D', label='Validation Rewards', zorder=5, edgecolor='darkred')
            
            plt.title(f'Episode Rewards - Update Step {update_step}', fontsize=16)
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Total Reward', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()

            plot_path = os.path.join(self.plots_dir, f'rewards_step_{update_step}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Rewards plot saved: {plot_path}")
            
        except Exception as e:
            print(f"Error generating rewards plot: {e}")
            plt.close()
    
    def plot_entropy(self, update_step):
        if len(self.entropies) < 2:
            print(f"Not enough entropy data to plot (have {len(self.entropies)}, need at least 2)")
            return
        
        try:
            plt.figure(figsize=(12, 8))
            steps = range(len(self.entropies))
            
            plt.plot(steps, self.entropies, color='orange', alpha=0.7, linewidth=2, label='Entropy')

            if len(self.entropies) >= 10:
                window = min(10, len(self.entropies) // 2) 
                entropy_ma = pd.Series(self.entropies).rolling(window=window).mean()
                plt.plot(steps, entropy_ma, color='darkorange', linewidth=3, alpha=0.8, label=f'MA({window})')
            
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='High Exploration (0.5)')
            plt.axhline(y=0.1, color='blue', linestyle='--', alpha=0.5, label='Low Exploration (0.1)')
            
            plt.title(f'Policy Entropy - Update Step {update_step}', fontsize=16)
            plt.xlabel('Update Step', fontsize=12)
            plt.ylabel('Entropy', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()

            plot_path = os.path.join(self.plots_dir, f'entropy_step_{update_step}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Entropy plot saved: {plot_path}")
            
        except Exception as e:
            print(f"Error generating entropy plot: {e}")
            plt.close()
    
    def force_generate_plots(self, update_step):
        self.plot_losses(update_step)
        self.plot_average_reward(update_step)
        self.plot_entropy(update_step)
        
        if self.validation_timesteps:
            latest_timestep = self.validation_timesteps[-1]
            self.plot_validation_only(latest_timestep)
            self.plot_validation_comparison(update_step)
        
        self.save_monitoring_data(update_step)
    
    def plot_validation_comparison(self, update_step):
        if not self.validation_timesteps:
            print(f"No validation data available for comparison plot")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Training vs Validation Comparison - Update Step {update_step}', fontsize=16)
            
            ax1 = axes[0, 0]
            episodes = range(len(self.episode_rewards))
            if len(self.episode_rewards) >= 20:
                window = min(20, len(self.episode_rewards) // 4)
                training_ma = pd.Series(self.episode_rewards).rolling(window=window).mean()
                ax1.plot(episodes, training_ma, color='blue', linewidth=2, label='Training (MA)')
            
            val_episodes = [ts // 200 for ts in self.validation_timesteps]
            ax1.plot(val_episodes, self.validation_rewards, color='red', linewidth=2, 
                    marker='o', markersize=6, label='Validation')
            
            ax1.set_title('Rewards: Training vs Validation')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Average Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Success rates comparison
            ax2 = axes[0, 1]
            if len(self.success_rates) >= 20:
                success_ma = pd.Series(self.success_rates).rolling(window=window).mean()
                ax2.plot(episodes, success_ma, color='blue', linewidth=2, label='Training (MA)')
            
            ax2.plot(val_episodes, self.validation_success_rates, color='red', linewidth=2,
                    marker='o', markersize=6, label='Validation')
            
            ax2.set_title('Success Rate: Training vs Validation')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Success Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Episode lengths comparison
            ax3 = axes[1, 0]
            if len(self.episode_lengths) >= 20:
                length_ma = pd.Series(self.episode_lengths).rolling(window=window).mean()
                ax3.plot(episodes, length_ma, color='blue', linewidth=2, label='Training (MA)')
            
            ax3.plot(val_episodes, self.validation_episode_lengths, color='red', linewidth=2,
                    marker='o', markersize=6, label='Validation')
            
            ax3.set_title('Episode Length: Training vs Validation')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Average Length')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Max SC values comparison
            ax4 = axes[1, 1]
            if len(self.max_sc_values) >= 20:
                sc_ma = pd.Series(self.max_sc_values).rolling(window=window).mean()
                ax4.plot(episodes, sc_ma, color='blue', linewidth=2, label='Training (MA)')
            
            ax4.plot(val_episodes, self.validation_max_sc_values, color='red', linewidth=2,
                    marker='o', markersize=6, label='Validation')
            
            ax4.set_title('Max SC Value: Training vs Validation')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Average Max SC')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.validation_dir, f'validation_comparison_step_{update_step}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error generating validation comparison plot: {e}")
            import traceback
            traceback.print_exc()
            plt.close()
    
    def plot_validation_only(self, timestep):
        if not self.validation_timesteps:
            print(f"No validation data available")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Validation Results - Timestep {timestep}', fontsize=16)
            
            # Validation rewards over time
            ax1 = axes[0, 0]
            ax1.plot(self.validation_timesteps, self.validation_rewards, 'r-o', linewidth=2, markersize=8)
            ax1.set_title('Validation Rewards Over Time')
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Average Reward')
            ax1.grid(True, alpha=0.3)
            
            # Validation success rates over time
            ax2 = axes[0, 1]
            ax2.plot(self.validation_timesteps, self.validation_success_rates, 'g-o', linewidth=2, markersize=8)
            ax2.set_title('Validation Success Rate Over Time')
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Success Rate')
            ax2.grid(True, alpha=0.3)
            
            # Validation episode lengths
            ax3 = axes[1, 0]
            ax3.plot(self.validation_timesteps, self.validation_episode_lengths, 'b-o', linewidth=2, markersize=8)
            ax3.set_title('Validation Episode Length Over Time')
            ax3.set_xlabel('Timestep')
            ax3.set_ylabel('Average Length')
            ax3.grid(True, alpha=0.3)
            
            # Validation max SC values
            ax4 = axes[1, 1]
            ax4.plot(self.validation_timesteps, self.validation_max_sc_values, 'm-o', linewidth=2, markersize=8)
            ax4.set_title('Validation Max SC Over Time')
            ax4.set_xlabel('Timestep')
            ax4.set_ylabel('Average Max SC')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.validation_dir, f'validation_only_timestep_{timestep}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error generating validation-only plot: {e}")
            import traceback
            traceback.print_exc()
            plt.close()
    
    def save_monitoring_data(self, update_step):
        try:
            training_data = {
                'update_step': int(update_step),
                'actor_losses': self.actor_losses,
                'critic_losses': self.critic_losses,
                'entropies': self.entropies,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'success_rates': self.success_rates,
                'max_sc_values': self.max_sc_values
            }
            
            json_path = os.path.join(self.data_dir, f'training_data_step_{update_step}.json')
            with open(json_path, 'w') as f:
                json.dump(training_data, f, indent=2)
            
            if self.validation_timesteps:
                validation_data = {
                    'validation_timesteps': self.validation_timesteps,
                    'validation_rewards': self.validation_rewards,
                    'validation_success_rates': self.validation_success_rates,
                    'validation_episode_lengths': self.validation_episode_lengths,
                    'validation_max_sc_values': self.validation_max_sc_values
                }
                
                val_json_path = os.path.join(self.validation_dir, f'validation_data_step_{update_step}.json')
                with open(val_json_path, 'w') as f:
                    json.dump(validation_data, f, indent=2)
            
            print(f"Monitoring data saved to {json_path}")
            
        except Exception as e:
            print(f"Error saving monitoring data: {e}")
    
    def generate_summary_report(self, update_step):
        if len(self.episode_rewards) == 0:
            return
        
        try:
            recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
            recent_success_rate = sum(self.recent_successes) / len(self.recent_successes) if self.recent_successes else 0
            
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            max_reward = np.max(recent_rewards)
            
            avg_entropy = np.mean(self.entropies[-100:]) if len(self.entropies) >= 100 else np.mean(self.entropies)
            
            val_summary = ""
            if self.validation_rewards:
                latest_val_reward = self.validation_rewards[-1]
                latest_val_success = self.validation_success_rates[-1]
                val_summary = f"""
            LATEST VALIDATION:
            - Validation Reward: {latest_val_reward:.2f}
            - Validation Success Rate: {latest_val_success:.2%}
            - Training vs Validation Gap: {avg_reward - latest_val_reward:.2f}
                """
            
            # Create summary
            summary = f"""
            ===== TRAINING SUMMARY (Update Step {update_step}) =====
            
            TRAINING PERFORMANCE:
            - Recent Average Reward: {avg_reward:.2f} ± {std_reward:.2f}
            - Recent Max Reward: {max_reward:.2f}
            - Success Rate (last 100): {recent_success_rate:.2%}
            - Total Episodes: {len(self.episode_rewards)}
            
            POLICY METRICS:
            - Average Entropy: {avg_entropy:.4f}
            - Latest Actor Loss: {self.actor_losses[-1]:.4f}
            - Latest Critic Loss: {self.critic_losses[-1]:.4f}
            {val_summary}
            TRAINING STATUS:
            - High Entropy (>0.3): {'' if avg_entropy > 0.3 else ''}
            - Improving Performance: {'' if len(self.episode_rewards) > 50 and np.mean(self.episode_rewards[-25:]) > np.mean(self.episode_rewards[-50:-25]) else ''}
            - Stable Learning: {'' if len(self.actor_losses) > 20 and np.std(self.actor_losses[-20:]) < 0.1 else ''}
            
            =====================================================
            """
            
            print(summary)
            summary_path = os.path.join(self.log_dir, f'summary_step_{update_step}.txt')
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            return summary
            
        except Exception as e:
            print(f"Error generating summary report: {e}")
            return None


def run_validation(ppo_agent, env, num_episodes=100):
    validation_rewards = []
    validation_lengths = []
    validation_successes = []
    validation_max_sc = []
    
    # Save current training mode
    was_training = ppo_agent.policy.training
    ppo_agent.policy.eval()  # Set to evaluation mode
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        max_sc_in_episode = env.previous_sc_probability

        for step in range(200):
            with torch.no_grad():
                # Use current policy for action selection (deterministic)
                state_tensor = preprocess_image_state(state)
                action_probs = ppo_agent.policy.actor(state_tensor)
                
                if ppo_agent.has_continuous_action_space:
                    action = action_probs.mean  # Use mean for deterministic action
                    action = action.cpu().numpy().flatten()
                else:
                    action = torch.argmax(action_probs).item()  # Deterministic discrete action
            
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            max_sc_in_episode = max(max_sc_in_episode, env.previous_sc_probability)
            
            state = next_state
            
            if done:
                break
        
        validation_rewards.append(episode_reward)
        validation_lengths.append(episode_length)
        validation_successes.append(1 if done else 0)
        validation_max_sc.append(max_sc_in_episode)
        
        if (episode + 1) % 20 == 0:
            print(f"   Validation episode {episode + 1}/{num_episodes} completed")
    
    if was_training:
        ppo_agent.policy.train()

    avg_reward = np.mean(validation_rewards)
    success_rate = np.mean(validation_successes)
    avg_length = np.mean(validation_lengths)
    avg_max_sc = np.mean(validation_max_sc)
    
    print(f"Validation completed!")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   Success Rate: {success_rate:.2%}")
    print(f"   Average Length: {avg_length:.1f}")
    print(f"   Average Max SC: {avg_max_sc:.4f}")
    
    return validation_rewards, success_rate, validation_lengths, validation_max_sc

def train_image_with_monitoring():
    print("TRAINING")
    print("="*70)

    image_env = create_image_env(classifier, GRADER_MODEL_PATH)
    
    # Training hyperparameters
    max_ep_len = 200
    update_timestep = 2048
    validation_freq = 10000  # Run validation every 10,000 timesteps
    validation_episodes = 100  # Number of episodes for validation
    
    # PPO hyperparameters  
    K_epochs = 5     
    eps_clip = 0.2
    lr_actor = 0.00002  
    lr_critic = 0.0001 
    max_training_timesteps = int(15e6)
    gamma = 0.95 
    
    print_freq = max_ep_len * 1
    save_model_freq = int(2e5)

    # Model dimensions
    state_dim = None
    action_dim = 12
    
    # Setup monitoring
    log_dir = "drl_results"
    os.makedirs(log_dir, exist_ok=True)
    
    monitor = TrainingMonitor(log_dir, save_freq=10)  # Generate plots every 10 updates
    
    print(f"Monitoring Configuration:")
    print(f"   Log directory: {log_dir}")
    print(f"   Validation frequency: every {validation_freq} timesteps")
    print(f"   Validation episodes: {validation_episodes}")
    print(f"   Plot generation frequency: every {monitor.save_freq} updates")
    print(f"   Update timestep: every {update_timestep} timesteps")
    print("="*70)
    
    # Initialize PPO agent
    image_ppo_agent = MonitoredImagePPO(
        state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
        False, monitor=monitor
    )
    
    # Checkpointing
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"ppo_image_only_model.pth")
    
    # Training variables
    start_time = datetime.now()
    print_running_reward = 0
    print_running_episodes = 0
    time_step = 0
    i_episode = 0
    last_validation_timestep = 0
    
    detailed_log_file = os.path.join(log_dir, f"detailed_training_log_{int(time.time())}.csv")
    
    # Start training
    with open(detailed_log_file, "w") as f:
        f.write("episode,timestep,total_reward,episode_length,success,max_sc_reached,actor_loss,critic_loss,entropy\n")
        
        while time_step <= max_training_timesteps:
            state = image_env.reset()
            current_ep_reward = 0
            episode_success = False
            max_sc_in_episode = image_env.previous_sc_probability
            
            for t in range(1, max_ep_len + 1):
                # Select action
                action = image_ppo_agent.select_action(state)
                next_state, reward, done, info = image_env.step(action)
                
                # Track max SC reached
                current_sc = image_env.previous_sc_probability
                max_sc_in_episode = max(max_sc_in_episode, current_sc)
                
                # Store experience
                image_ppo_agent.buffer.rewards.append(reward)
                image_ppo_agent.buffer.is_terminals.append(done)
                
                time_step += 1
                current_ep_reward += reward
                state = next_state
                
                # Run validation
                if time_step - last_validation_timestep >= validation_freq:
                    print(f"\n Running validation at timestep {time_step}")
                    val_rewards, val_success_rate, val_lengths, val_max_sc = run_validation(
                        image_ppo_agent, image_env, validation_episodes
                    )
                    monitor.log_validation(time_step, val_rewards, val_success_rate, val_lengths, val_max_sc)
                    
                    print(f"Generating validation plots for timestep {time_step}...")
                    monitor.debug_validation_status() 
                    monitor.plot_validation_only(time_step)
                    if len(monitor.episode_rewards) > 0:
                        monitor.plot_validation_comparison(image_ppo_agent.update_count)
                    
                    last_validation_timestep = time_step
                    print(f"Validation plots generated for timestep {time_step}")
                    print()  # Add spacing after validation
                
                # Update PPO agent
                if time_step % update_timestep == 0:
                    print(f"Updating policy at timestep {time_step} (Update #{image_ppo_agent.update_count})")
                    update_metrics = image_ppo_agent.update()
                    
                    print(f"Saving monitoring data for update {image_ppo_agent.update_count}")
                    monitor.save_monitoring_data(image_ppo_agent.update_count)
                    
                    if image_ppo_agent.update_count <= 5:
                        monitor.force_generate_plots(image_ppo_agent.update_count)
                    
                    elif image_ppo_agent.update_count % monitor.save_freq == 0:
                        monitor.plot_losses(image_ppo_agent.update_count)
                        monitor.plot_average_reward(image_ppo_agent.update_count)
                        monitor.plot_entropy(image_ppo_agent.update_count)
                        
                        if monitor.validation_timesteps:
                            monitor.plot_validation_comparison(image_ppo_agent.update_count)
                        else:
                            print(f"No validation data available yet for comparison")
                        
                        print(f"All plots generated for update {image_ppo_agent.update_count}")
                        
                        if image_ppo_agent.update_count % (monitor.save_freq * 5) == 0:
                            monitor.generate_summary_report(image_ppo_agent.update_count)
                    
                    elif image_ppo_agent.update_count % 25 == 0:
                        monitor.generate_summary_report(image_ppo_agent.update_count)
                    
                    print(f"   Actor Loss: {update_metrics['actor_loss']:.4f}")
                    print(f"   Critic Loss: {update_metrics['critic_loss']:.4f}")
                    print(f"   Entropy: {update_metrics['entropy']:.4f}")
                
                if time_step % print_freq == 0:
                    avg_reward = print_running_reward / max(1, print_running_episodes)
                    recent_entropy = monitor.entropies[-1] if monitor.entropies else 0
                    recent_success_rate = sum(monitor.recent_successes) / len(monitor.recent_successes) if monitor.recent_successes else 0
                    
                    print(f"Episode: {i_episode:6d} | Timestep: {time_step:8d} | "
                          f"Avg Reward: {avg_reward:8.2f} | Max SC: {max_sc_in_episode:.4f} | "
                          f"Entropy: {recent_entropy:.4f} | Success Rate: {recent_success_rate:.2%}")
                    
                    print_running_reward = 0
                    print_running_episodes = 0
                
                if time_step % save_model_freq == 0:
                    print(f"Saving model at timestep {time_step}")
                    image_ppo_agent.save(checkpoint_path)
                
                if done:
                    episode_success = True
                    print(f"Episode {i_episode} SUCCESS at step {t}! Final SC: {current_sc:.4f}")
                    break
            
            monitor.log_episode(current_ep_reward, t, episode_success, max_sc_in_episode)
            
            latest_actor_loss = monitor.actor_losses[-1] if monitor.actor_losses else 0
            latest_critic_loss = monitor.critic_losses[-1] if monitor.critic_losses else 0  
            latest_entropy = monitor.entropies[-1] if monitor.entropies else 0
            
            with open(detailed_log_file, "a") as f:
                f.write(f"{i_episode},{time_step},{current_ep_reward},{t},{episode_success},"
                       f"{max_sc_in_episode:.6f},{latest_actor_loss:.6f},{latest_critic_loss:.6f},{latest_entropy:.6f}\n")
            
            print_running_reward += current_ep_reward
            print_running_episodes += 1
            i_episode += 1
            
            if i_episode % 10 == 0:
                print(f"\nEpisode {i_episode} Summary:")
                print(f"   Total reward: {current_ep_reward:.2f}")
                print(f"   Steps taken: {t}")
                print(f"   Max SC reached: {max_sc_in_episode:.4f}")
                print(f"   Success: {'Yes' if episode_success else 'No'}")
                if monitor.entropies:
                    print(f"   Current entropy: {monitor.entropies[-1]:.4f}")
                if len(monitor.recent_successes) > 0:
                    recent_success_rate = sum(monitor.recent_successes) / len(monitor.recent_successes)
                    print(f"   Recent success rate: {recent_success_rate:.2%}")
                print(image_env.get_action_summary())
    
    # Final validation
    val_rewards, val_success_rate, val_lengths, val_max_sc = run_validation(
        image_ppo_agent, image_env, validation_episodes
    )
    monitor.log_validation(time_step, val_rewards, val_success_rate, val_lengths, val_max_sc)
    
    # Generate validation plots
    monitor.plot_validation_only(time_step)

    monitor.plot_losses(image_ppo_agent.update_count)
    monitor.plot_average_reward(image_ppo_agent.update_count)
    monitor.plot_entropy(image_ppo_agent.update_count)
    monitor.plot_validation_comparison(image_ppo_agent.update_count)
    monitor.save_monitoring_data(image_ppo_agent.update_count)
    monitor.generate_summary_report(image_ppo_agent.update_count)

    image_ppo_agent.save(checkpoint_path)
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print(f"   Training time: {training_time}")
    print(f"   Total episodes: {i_episode}")
    print(f"   Total updates: {image_ppo_agent.update_count}")
    print(f"   Validation runs: {len(monitor.validation_timesteps)}")
    print(f"   Model saved to: {checkpoint_path}")
    print(f"   Logs and plots saved to: {log_dir}")
    print("="*70)

class MonitoredImagePPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
                 has_continuous_action_space, action_std_init=0.6, monitor=None):
        
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.monitor = monitor

        self.buffer = RolloutBuffer()

        self.policy = ImageOnlyActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ImageOnlyActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.update_count = 0
        self.time_step = 0  # ADD THIS
        self.max_training_timesteps = int(15e6)  # ADD THIS

    def select_action(self, state):
        self.time_step += 1
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = preprocess_image_state(state)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = preprocess_image_state(state)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            return action.item()

    def update(self):
        self.update_count += 1
        
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.cat(self.buffer.states, dim=0).to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        advantages = rewards.detach() - old_state_values.detach()

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for epoch in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
 
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * self.MseLoss(state_values, rewards)
            #entropy_loss = -0.01 * dist_entropy.mean()
            entropy_coef = max(0.05, 0.1 * (1 - self.time_step / self.max_training_timesteps))
            entropy_loss = -entropy_coef * dist_entropy.mean()
            
            loss = actor_loss + critic_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += dist_entropy.mean().item()

        avg_actor_loss = total_actor_loss / self.K_epochs
        avg_critic_loss = total_critic_loss / self.K_epochs
        avg_entropy = total_entropy / self.K_epochs

        if self.monitor:
            self.monitor.log_ppo_update(avg_actor_loss, avg_critic_loss, avg_entropy)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer.clear()
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss, 
            'entropy': avg_entropy
        }

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


###################################################### MAIN EXECUTION ######################################################
if __name__ == '__main__':
    print("="*70)   
    train_image_with_monitoring()
print("\n" + "="*70)
print("="*70)
