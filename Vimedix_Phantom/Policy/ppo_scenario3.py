import warnings
warnings.filterwarnings("ignore")

import sys
import os
import json
import math
import random
import time
from collections import deque
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision.models import resnet18
from gym import Env, spaces
from residual_gan import ConditionalGAN


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

NORM_MIN = np.array([-0.144989, -0.487173,  0.198017,
                     -0.0400,   -0.5600,   -0.3100,   -0.5000  ], dtype=np.float32)
NORM_MAX = np.array([ 0.010570, -0.272387,  0.274926,
                      0.8900,    0.0200,    0.3200,    1.0000   ], dtype=np.float32)

WS_MIN = np.array([-0.144989, -0.46,  0.198017], dtype=np.float32)
WS_MAX = np.array([ 0.01,     -0.35,  0.22    ], dtype=np.float32)

norm_params = {"min": NORM_MIN, "max": NORM_MAX}

print(f" GAN norm range  x:[{NORM_MIN[0]:.3f},{NORM_MAX[0]:.3f}]  "
      f"y:[{NORM_MIN[1]:.3f},{NORM_MAX[1]:.3f}]  "
      f"z:[{NORM_MIN[2]:.3f},{NORM_MAX[2]:.3f}]")
print(f" Agent workspace x:[{WS_MIN[0]:.3f},{WS_MAX[0]:.3f}]  "
      f"y:[{WS_MIN[1]:.3f},{WS_MAX[1]:.3f}]  "
      f"z:[{WS_MIN[2]:.3f},{WS_MAX[2]:.3f}]")


us_step_x  = 0.0150
us_step_y  = 0.0065
us_step_z  = 0.0045
us_step_rx = math.radians(5) 
us_step_ry = math.radians(5)
us_step_rz = math.radians(5) 

Z_DIM     = 100
LABEL_DIM = 7

SC_CLASS_IDX       = 5   # SC2
RANDOM_CLASS_IDX   = 3   # Random

CLASS_NAMES          = ["5CH", "PLAX", "PSAX", "Random", "SC1", "SC2"]
NUM_CLASSES          = 6
SC_SUCCESS_THRESHOLD = 0.8

GAN_CHECKPOINT            = "conditional_gan_residual_epoch_100.pth"
CLASSIFICATION_MODEL_PATH = "classifier_best.pth"

IMG_C, IMG_H, IMG_W = 1, 128, 128

QX_RANGE = (-0.04,   1.0)
QY_RANGE = (-0.6,   1.0)
QZ_RANGE = (-0.6,   0.4)
QW_RANGE = (-0.6,   1.0)

STRAIGHT_QUAT = (-0.03352953333753081,
                 -0.015565514277348152,
                  0.0048032260838987775,
                  0.9993049655528767)

np.random.seed(42)

def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float32)


def _axis_angle_to_quat(axis: str, angle: float) -> np.ndarray:
    half = angle / 2.0
    s, c = math.sin(half), math.cos(half)
    if axis == 'x': return np.array([s, 0, 0, c], dtype=np.float32)
    if axis == 'y': return np.array([0, s, 0, c], dtype=np.float32)
    if axis == 'z': return np.array([0, 0, s, c], dtype=np.float32)


def _random_bounded_quat() -> np.ndarray:
    q = np.array([
        random.uniform(*QX_RANGE),
        random.uniform(*QY_RANGE),
        random.uniform(*QZ_RANGE),
        random.uniform(*QW_RANGE),
    ], dtype=np.float32)
    return q / np.linalg.norm(q)

def _random_unit_quat() -> np.ndarray:
    q = np.random.randn(4).astype(np.float32)
    return q / np.linalg.norm(q)


def domain_aware_normalize(values, norm_params):
    """Normalise xyz to [-1,1]; keep quaternion components raw."""
    mn, mx = norm_params["min"], norm_params["max"]
    out = np.zeros(7, dtype=np.float32)
    out[:3] = 2 * (values[:3] - mn[:3]) / (mx[:3] - mn[:3] + 1e-8) - 1
    out[3:] = values[3:]
    return out


class ImageClassifier:
    def __init__(self, model_path, num_classes=NUM_CLASSES,
                 target_class_idx=SC_CLASS_IDX, class_names=None):
        self.device           = device
        self.num_classes      = num_classes
        self.target_class_idx = target_class_idx
        self.class_names      = class_names or CLASS_NAMES

        self.model = resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        print(f" Classifier loaded  (SC={SC_CLASS_IDX}, "
              f"Random={RANDOM_CLASS_IDX}, VGAN_neg={VGAN_NEG_CLASS_IDX})")

    def classify_image(self, image):
        try:
            img_t = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(img_t)
                probs  = F.softmax(logits, dim=1)
                conf, pred = torch.max(probs, 1)
            return pred.item(), conf.item(), probs.squeeze().cpu().numpy()
        except Exception as e:
            print(f"  Classifier error: {e}")
            return 0, 0.0, np.zeros(self.num_classes)


class UltrasoundImageGenerator:
    def __init__(self, checkpoint_path, z_dim=Z_DIM, label_dim=LABEL_DIM):
        self.device = device
        self.z_dim  = z_dim

        self.model = ConditionalGAN(z_dim=z_dim, label_dim=label_dim, img_channels=1).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f" GAN loaded  (epoch {ckpt.get('epoch','?')})")

        self.fixed_z = torch.randn(1, z_dim, device=device)

    def generate_tensor(self, normalised_pose: np.ndarray) -> torch.Tensor:
        label = torch.tensor(normalised_pose, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            gen = self.model.generator(self.fixed_z, label)
        return gen.squeeze(0).cpu()   # (1, H, W)

    def generate_numpy(self, normalised_pose: np.ndarray) -> np.ndarray:
        t = self.generate_tensor(normalised_pose)
        return ((t.squeeze().numpy() + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)

image_generator = UltrasoundImageGenerator(GAN_CHECKPOINT)
classifier      = ImageClassifier(CLASSIFICATION_MODEL_PATH)

def preprocess_state(image: np.ndarray) -> torch.Tensor:
    image_t = torch.from_numpy(image).float()   # (1, H, W)
    return image_t.unsqueeze(0).to(device)       # (1, 1, H, W)


class ImageStateEnv(Env):

    def __init__(self, use_rotation=True):
        super().__init__()
        self.use_rotation = use_rotation

        self.steps_in_episode        = 0
        self.action_history          = []
        self.previous_sc_probability = 0.0

        self.initial_position = (
            float(np.mean([WS_MIN[0], WS_MAX[0]])),
            float(np.mean([WS_MIN[1], WS_MAX[1]])),
            float(np.mean([WS_MIN[2], WS_MAX[2]])),
        )
        self.initial_rotation = STRAIGHT_QUAT
        self._reset_internal()

        self.action_space      = spaces.Discrete(12 if use_rotation else 6)
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(IMG_C, IMG_H, IMG_W),
                                            dtype=np.float32)

        self.action_names = [
            "move_x_neg", "move_x_pos",
            "move_y_neg", "move_y_pos",
            "move_z_neg", "move_z_pos",
            "rot_x_neg",  "rot_x_pos",
            "rot_y_neg",  "rot_y_pos",
            "rot_z_neg",  "rot_z_pos",
        ]

        print(" ImageStateEnv (Case 3) Initialized")
        print(f"   Init quat (straight): qx={STRAIGHT_QUAT[0]:.4f}  "
              f"qy={STRAIGHT_QUAT[1]:.4f}  "
              f"qz={STRAIGHT_QUAT[2]:.4f}  "
              f"qw={STRAIGHT_QUAT[3]:.4f}")

    def _reset_internal(self):
        self.current_position        = self.initial_position
        self.current_rotation        = self.initial_rotation
        self.steps_in_episode        = 0
        self.action_history          = []
        self.previous_sc_probability = 0.0

    def _get_norm_pose(self) -> np.ndarray:
        pose = np.array([*self.current_position, *self.current_rotation], dtype=np.float32)
        return domain_aware_normalize(pose, norm_params)

    def reset(self):
        self.initial_position = (
            random.uniform(float(WS_MIN[0]), float(WS_MAX[0])),
            random.uniform(float(WS_MIN[1]), float(WS_MAX[1])),
            random.uniform(float(WS_MIN[2]), float(WS_MAX[2])),
        )
        q = _random_bounded_quat()  # _random_unit_quat() #
        self.initial_rotation = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        self._reset_internal()

        norm_pose   = self._get_norm_pose()
        gen_tensor  = image_generator.generate_tensor(norm_pose)
        _, _, probs = classifier.classify_image(gen_tensor)

        self.previous_sc_probability = float(probs[SC_CLASS_IDX])
        image = gen_tensor.numpy()   # (1, 128, 128)

        print(f"🔄 RESET | pos=({self.initial_position[0]:.3f},"
              f"{self.initial_position[1]:.3f},{self.initial_position[2]:.3f}) | "
              f"SC={self.previous_sc_probability:.4f}")
        return image

    def step(self, action):
        self.steps_in_episode += 1
        action_name = (self.action_names[action]
                       if action < len(self.action_names) else f"unknown_{action}")

        self.action_history.append({
            "step":            self.steps_in_episode,
            "action_id":       action,
            "action_name":     action_name,
            "position_before": self.current_position,
            "rotation_before": self.current_rotation,
        })

        new_position, new_rotation = self._apply_action(action)
        new_pose  = np.array([*new_position, *new_rotation], dtype=np.float32)
        norm_pose = domain_aware_normalize(new_pose, norm_params)

        gen_tensor  = image_generator.generate_tensor(norm_pose)
        _, _, probs = classifier.classify_image(gen_tensor)

        reward, done = self._calculate_reward(probs)

        self.current_position = new_position
        self.current_rotation = new_rotation
        self.action_history[-1]["position_after"] = new_position
        self.action_history[-1]["rotation_after"] = new_rotation

        image = gen_tensor.numpy()   # (1, 128, 128)

        info = {
            "position":       new_position,
            "rotation":       new_rotation,
            "steps":          self.steps_in_episode,
            "last_action":    action_name,
            "sc_probability": self.previous_sc_probability,
        }
        return image, reward, done, info

    def _apply_action(self, action):
        x, y, z        = self.current_position
        qx, qy, qz, qw = self.current_rotation

        original_x, original_y, original_z = x, y, z

        if   action == 0: x -= us_step_x
        elif action == 1: x += us_step_x
        elif action == 2: y -= us_step_y
        elif action == 3: y += us_step_y
        elif action == 4: z -= us_step_z
        elif action == 5: z += us_step_z
        elif self.use_rotation and action == 6:
            q = _quat_multiply([qx,qy,qz,qw], _axis_angle_to_quat('x', -us_step_rx))
            qx,qy,qz,qw = q / np.linalg.norm(q)
        elif self.use_rotation and action == 7:
            q = _quat_multiply([qx,qy,qz,qw], _axis_angle_to_quat('x', +us_step_rx))
            qx,qy,qz,qw = q / np.linalg.norm(q)
        elif self.use_rotation and action == 8:
            q = _quat_multiply([qx,qy,qz,qw], _axis_angle_to_quat('y', -us_step_ry))
            qx,qy,qz,qw = q / np.linalg.norm(q)
        elif self.use_rotation and action == 9:
            q = _quat_multiply([qx,qy,qz,qw], _axis_angle_to_quat('y', +us_step_ry))
            qx,qy,qz,qw = q / np.linalg.norm(q)
        elif self.use_rotation and action == 10:
            q = _quat_multiply([qx,qy,qz,qw], _axis_angle_to_quat('z', -us_step_rz))
            qx,qy,qz,qw = q / np.linalg.norm(q)
        elif self.use_rotation and action == 11:
            q = _quat_multiply([qx,qy,qz,qw], _axis_angle_to_quat('z', +us_step_rz))
            qx,qy,qz,qw = q / np.linalg.norm(q)

        
        if (x < WS_MIN[0] or x > WS_MAX[0] or
            y < WS_MIN[1] or y > WS_MAX[1] or
            z < WS_MIN[2] or z > WS_MAX[2]):

            # If out of bounds, revert to original position (no movement)
            print(f" Boundary hit! Action {action} blocked. Staying at current position.")
            x, y, z = original_x, original_y, original_z

        x = float(np.clip(x, WS_MIN[0], WS_MAX[0]))
        y = float(np.clip(y, WS_MIN[1], WS_MAX[1]))
        z = float(np.clip(z, WS_MIN[2], WS_MAX[2]))

        return (round(x,10), round(y,10), round(z,10)), \
               (round(qx,10), round(qy,10), round(qz,10), round(qw,10))

    def _calculate_reward(self, probs: np.ndarray):
        SCALE_FACTOR = 100.0

        current_sc_prob = float(probs[SC_CLASS_IDX])
        shaping_reward  = (current_sc_prob * SCALE_FACTOR
                           - self.previous_sc_probability * SCALE_FACTOR)

        base_reward = 0.0
        done        = False

        if current_sc_prob >= SC_SUCCESS_THRESHOLD:
            base_reward = 100.0
            done        = True
            print(f" SUCCESS  SC={current_sc_prob:.4f}  "
                  f"pos=({self.current_position[0]:.4f},"
                  f"{self.current_position[1]:.4f},"
                  f"{self.current_position[2]:.4f})")

        step_penalty    = -0.001
        #direction_bonus = 1.0 if shaping_reward > 0 else -1.0
        #sc_reward       = current_sc_prob #* SCALE_FACTOR


        #total_reward    = base_reward + step_penalty + direction_bonus + sc_reward
        total_reward    = base_reward + step_penalty + shaping_reward 

        self.previous_sc_probability = current_sc_prob
        return total_reward, done

    def get_action_summary(self):
        if not self.action_history:
            return "No actions yet"
        counts = {}
        for r in self.action_history:
            counts[r["action_name"]] = counts.get(r["action_name"], 0) + 1
        s = f"Episode {self.steps_in_episode} steps:\n"
        for name, count in sorted(counts.items()):
            s += f"  {name}: {count}x\n"
        return s

class RolloutBuffer:
    def __init__(self):
        self.images       = []
        self.actions      = []
        self.logprobs     = []
        self.rewards      = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        for lst in [self.images, self.actions, self.logprobs,
                    self.rewards, self.state_values, self.is_terminals]:
            del lst[:]

CNN_FEAT_DIM = 256

class ImageActor(nn.Module):
    def __init__(self, action_dim=12):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,   32,  kernel_size=8, stride=4, padding=2),
            nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32,  64,  kernel_size=4, stride=2, padding=1),
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,  128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.BatchNorm2d(256),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, CNN_FEAT_DIM), nn.ReLU(),
            nn.Linear(CNN_FEAT_DIM, action_dim),
        )

    def forward(self, image):
        feat = self.cnn(image).view(image.size(0), -1)
        return F.softmax(self.actor_head(feat), dim=-1)


class ImageCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,   32,  kernel_size=8, stride=4, padding=2),
            nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32,  64,  kernel_size=4, stride=2, padding=1),
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,  128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.BatchNorm2d(256),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, CNN_FEAT_DIM), nn.ReLU(),
            nn.Linear(CNN_FEAT_DIM, 1),
        )

    def forward(self, image):
        feat = self.cnn(image).view(image.size(0), -1)
        return self.critic_head(feat)


class ImageActorCritic(nn.Module):
    def __init__(self, action_dim=12):
        super().__init__()
        self.actor  = ImageActor(action_dim=action_dim).to(device)
        self.critic = ImageCritic().to(device)

    def set_action_std(self, _): pass
    def forward(self): raise NotImplementedError

    def act(self, image):
        probs  = self.actor(image)
        dist   = Categorical(probs)
        action = dist.sample()
        return (action.detach(),
                dist.log_prob(action).detach(),
                self.critic(image).detach())

    def evaluate(self, image, action):
        probs = self.actor(image)
        dist  = Categorical(probs)
        return (dist.log_prob(action),
                self.critic(image),
                dist.entropy())

class ImagePPO:
    def __init__(self, action_dim, lr_actor, lr_critic, gamma,
                 K_epochs, eps_clip, monitor=None):

        self.gamma    = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.monitor  = monitor
        self.buffer   = RolloutBuffer()

        self.policy = ImageActorCritic(action_dim=action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {"params": self.policy.actor.parameters(),  "lr": lr_actor},
            {"params": self.policy.critic.parameters(), "lr": lr_critic},
        ])
        self.policy_old = ImageActorCritic(action_dim=action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss      = nn.MSELoss()
        self.update_count = 0

        n_actor  = sum(p.numel() for p in self.policy.actor.parameters())
        n_critic = sum(p.numel() for p in self.policy.critic.parameters())
        print(f" ImagePPO | actor params: {n_actor:,} | critic params: {n_critic:,}")

    def select_action(self, image: np.ndarray) -> int:
        image_t = preprocess_state(image)
        with torch.no_grad():
            action, logprob, val = self.policy_old.act(image_t)
        self.buffer.images.append(image_t)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        self.buffer.state_values.append(val)
        return action.item()

    def update(self, time_step=0, max_training_timesteps=15e6):
        self.update_count += 1

        rewards    = []
        discounted = 0.0
        for r, terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if terminal:
                discounted = 0.0
            discounted = r + self.gamma * discounted
            rewards.insert(0, discounted)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_images   = torch.cat(self.buffer.images, dim=0).to(device)
        old_actions  = torch.squeeze(torch.stack(self.buffer.actions,      dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs,     dim=0)).detach().to(device)
        old_values   = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        advantages = rewards.detach() - old_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        tot_a = tot_c = tot_e = 0.0
        for _ in range(self.K_epochs):
            logprobs, vals, entropy = self.policy.evaluate(old_images, old_actions)
            vals   = torch.squeeze(vals)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1  = ratios * advantages
            surr2  = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss =  0.5 * self.MseLoss(vals, rewards)
            ent_loss    = -0.01 * entropy.mean()
            loss        = actor_loss + critic_loss + ent_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            tot_a += actor_loss.item()
            tot_c += critic_loss.item()
            tot_e += entropy.mean().item()

        avg_a = tot_a / self.K_epochs
        avg_c = tot_c / self.K_epochs
        avg_e = tot_e / self.K_epochs

        if self.monitor:
            self.monitor.log_ppo_update(avg_a, avg_c, avg_e)

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        return {"actor_loss": avg_a, "critic_loss": avg_c, "entropy": avg_e}

    def save(self, path): torch.save(self.policy_old.state_dict(), path)

    def load(self, path):
        sd = torch.load(path, map_location=lambda s, l: s)
        self.policy_old.load_state_dict(sd)
        self.policy.load_state_dict(sd)

class TrainingMonitor:
    def __init__(self, log_dir, save_freq=100):
        self.log_dir   = log_dir
        self.save_freq = save_freq

        self.plots_dir      = os.path.join(log_dir, "plots")
        self.data_dir       = os.path.join(log_dir, "monitoring_data")
        self.validation_dir = os.path.join(log_dir, "validation")
        for d in [self.plots_dir, self.data_dir, self.validation_dir]:
            os.makedirs(d, exist_ok=True)

        self.actor_losses    = []; self.critic_losses = []; self.entropies     = []
        self.episode_rewards = []; self.episode_lengths = []
        self.success_rates   = []; self.max_sc_values   = []
        self.training_times  = []

        self.validation_timesteps       = []
        self.validation_rewards         = []
        self.validation_success_rates   = []
        self.validation_episode_lengths = []
        self.validation_max_sc_values   = []

        self.recent_successes = deque(maxlen=100)
        self._start_time      = time.time()
        print(f" TrainingMonitor → {log_dir}")

    def log_ppo_update(self, a, c, e):
        self.actor_losses.append(float(a))
        self.critic_losses.append(float(c))
        self.entropies.append(float(e))

    def log_episode(self, reward, length, success, max_sc):
        self.episode_rewards.append(float(reward))
        self.episode_lengths.append(int(length))
        self.max_sc_values.append(float(max_sc))
        self.recent_successes.append(1 if success else 0)
        self.success_rates.append(sum(self.recent_successes) / len(self.recent_successes))
        self.training_times.append(float(time.time() - self._start_time))

    def log_validation(self, timestep, val_r, val_sr, val_l, val_sc):
        self.validation_timesteps.append(timestep)
        self.validation_rewards.append(float(np.mean(val_r)))
        self.validation_success_rates.append(float(val_sr))
        self.validation_episode_lengths.append(float(np.mean(val_l)))
        self.validation_max_sc_values.append(float(np.mean(val_sc)))
        print(f" Val @ {timestep}: reward={np.mean(val_r):.2f}  "
              f"success={val_sr:.2%}  max_sc={np.mean(val_sc):.4f}")

    def force_generate_plots(self, step):
        self.save_monitoring_data(step)

    def save_monitoring_data(self, step):
        data = {
            "case":            "case3_sc",
            "update_step":     int(step),
            "actor_losses":    self.actor_losses,
            "critic_losses":   self.critic_losses,
            "entropies":       self.entropies,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "success_rates":   self.success_rates,
            "max_sc_values":   self.max_sc_values,
            "training_times":  self.training_times,
            "validation": {
                "timesteps":       self.validation_timesteps,
                "rewards":         self.validation_rewards,
                "success_rates":   self.validation_success_rates,
                "episode_lengths": self.validation_episode_lengths,
                "max_sc_values":   self.validation_max_sc_values,
            },
        }
        path = os.path.join(self.data_dir, f"training_{step}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        with open(os.path.join(self.log_dir, "results_latest.json"), "w") as f:
            json.dump(data, f, indent=2)

    def generate_summary_report(self, step):
        if not self.episode_rewards: return
        recent = self.episode_rewards[-100:]
        sr     = sum(self.recent_successes) / max(1, len(self.recent_successes))
        print(f"\n===== SUMMARY (step {step}) =====")
        print(f"  Avg reward (last 100): {np.mean(recent):.2f} ± {np.std(recent):.2f}")
        print(f"  Success rate:          {sr:.2%}")
        if self.entropies:
            print(f"  Entropy:               {self.entropies[-1]:.4f}")
        print("=" * 40)


def run_validation(agent: ImagePPO, env: ImageStateEnv, num_episodes=20) -> tuple:
    print(f"\n Validation ({num_episodes} episodes) …")
    val_rewards = []; val_lengths = []; val_successes = []; val_max_sc = []

    was_training = agent.policy.training
    agent.policy.eval()

    for ep in range(num_episodes):
        image     = env.reset()
        ep_reward = 0; ep_length = 0; done = False
        max_sc_ep = env.previous_sc_probability

        for _ in range(300):
            image_t = preprocess_state(image)
            with torch.no_grad():
                probs  = agent.policy.actor(image_t)
                action = torch.argmax(probs).item()
            image, reward, done, _ = env.step(action)
            ep_reward += reward; ep_length += 1
            max_sc_ep  = max(max_sc_ep, env.previous_sc_probability)
            if done: break

        val_rewards.append(ep_reward); val_lengths.append(ep_length)
        val_successes.append(1 if done else 0); val_max_sc.append(max_sc_ep)
        if (ep + 1) % 10 == 0:
            print(f"   Val ep {ep+1}/{num_episodes}")

    if was_training:
        agent.policy.train()

    print(f" Val | reward={np.mean(val_rewards):.2f}  "
          f"success={np.mean(val_successes):.2%}  "
          f"max_sc={np.mean(val_max_sc):.4f}")
    return val_rewards, float(np.mean(val_successes)), val_lengths, val_max_sc


def train():
    print(" CASE 3 — Image State, GAN Reward (SC)")
    print("=" * 70)

    env = ImageStateEnv(use_rotation=True)

    max_ep_len             = 300 
    update_timestep        = 4096 
    K_epochs               = 20
    eps_clip               = 0.2
    lr_actor               = 0.0001
    lr_critic              = 0.0005
    gamma                  = 0.99
    max_training_timesteps = int(15e6)
    validation_freq        = 10_000
    validation_episodes    = 20
    action_dim             = 12
    print_freq             = max_ep_len * 1
    save_model_freq        = int(2e5)

    log_dir        = "scenario3_log_dir"
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path     = os.path.join(checkpoint_dir, "case3_sc_image.pth")
    best_model_path      = os.path.join(checkpoint_dir, "best_model.pth")
    training_state_path = os.path.join(checkpoint_dir, "training_state.json")

    monitor = TrainingMonitor(log_dir, save_freq=10)
    agent   = ImagePPO(
        action_dim=action_dim,
        lr_actor=lr_actor, lr_critic=lr_critic,
        gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip,
        monitor=monitor,
    )

    # Resume support
    time_step              = 0
    i_episode              = 0
    last_validation_ts     = 0
    print_running_reward   = 0
    print_running_episodes = 0
    best_val_success       = -1.0

    if os.path.exists(checkpoint_path):
        print(f"\n Resuming from {checkpoint_path}")
        agent.load(checkpoint_path)
        if os.path.exists(training_state_path):
            with open(training_state_path) as f:
                saved = json.load(f)
            time_step          = saved.get("time_step", 0)
            i_episode          = saved.get("i_episode", 0)
            last_validation_ts = saved.get("last_validation_ts", 0)
            agent.update_count = saved.get("update_count", 0)
            best_val_success    = saved.get("best_val_success", -1.0) 
            print(f"   ↳ timestep={time_step}, episode={i_episode}, "
                  f"updates={agent.update_count}")
    else:
        print("\n Starting from scratch")

    start_time   = datetime.now()
    detailed_log = os.path.join(log_dir, f"training_log_{int(time.time())}.csv")
    with open(detailed_log, "a") as f:
        if time_step == 0:
            f.write("episode,timestep,total_reward,ep_length,success,"
                    "max_sc_prob,actor_loss,critic_loss,entropy\n")

    while time_step <= max_training_timesteps:
        image             = env.reset()
        current_ep_reward = 0
        episode_success   = False
        max_sc_episode    = env.previous_sc_probability

        for t in range(1, max_ep_len + 1):
            action = agent.select_action(image)
            image, reward, done, info = env.step(action)

            max_sc_episode = max(max_sc_episode, env.previous_sc_probability)
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step         += 1
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                agent.update(time_step=time_step,
                             max_training_timesteps=max_training_timesteps)
                monitor.save_monitoring_data(agent.update_count)
                if agent.update_count <= 5 or agent.update_count % monitor.save_freq == 0:
                    monitor.force_generate_plots(agent.update_count)
                elif agent.update_count % 25 == 0:
                    monitor.generate_summary_report(agent.update_count)

            if time_step % print_freq == 0:
                avg_r = print_running_reward / max(1, print_running_episodes)
                ent   = monitor.entropies[-1] if monitor.entropies else 0
                sr    = sum(monitor.recent_successes) / max(1, len(monitor.recent_successes))
                print(f"Ep {i_episode:6d} | Step {time_step:8d} | "
                      f"AvgR {avg_r:8.2f} | MaxSC {max_sc_episode:.4f} | "
                      f"Ent {ent:.4f} | SR {sr:.2%}")
                print_running_reward = 0; print_running_episodes = 0

            if time_step % save_model_freq == 0:
                agent.save(checkpoint_path)
                with open(training_state_path, "w") as f:
                    json.dump({
                        "time_step":          time_step,
                        "i_episode":          i_episode,
                        "last_validation_ts": last_validation_ts,
                        "update_count":       agent.update_count,
                        "best_val_success":   best_val_success, 
                    }, f, indent=2)
                print(f" Checkpoint saved @ step {time_step}")

            if done:
                episode_success = True
                print(f" Ep {i_episode} SUCCESS @ step {t}  "
                      f"SC={env.previous_sc_probability:.4f}")
                break

        if time_step - last_validation_ts >= validation_freq:
            val_r, val_sr, val_l, val_sc = run_validation(agent, env, validation_episodes)
            monitor.log_validation(time_step, val_r, val_sr, val_l, val_sc)
            last_validation_ts = time_step

            if val_sr > best_val_success:
                best_val_success = val_sr
                agent.save(best_model_path)
                print(f" New best model (val_sr={val_sr:.2%}) saved @ step {time_step}")

        monitor.log_episode(current_ep_reward, t, episode_success, max_sc_episode)

        al = monitor.actor_losses[-1]  if monitor.actor_losses  else 0
        cl = monitor.critic_losses[-1] if monitor.critic_losses else 0
        en = monitor.entropies[-1]     if monitor.entropies     else 0

        with open(detailed_log, "a") as f:
            f.write(f"{i_episode},{time_step},{current_ep_reward},{t},"
                    f"{episode_success},{max_sc_episode:.6f},"
                    f"{al:.6f},{cl:.6f},{en:.6f}\n")

        print_running_reward   += current_ep_reward
        print_running_episodes += 1
        i_episode              += 1

        if i_episode % 10 == 0:
            sr = sum(monitor.recent_successes) / max(1, len(monitor.recent_successes))
            print(f"\n Ep {i_episode} | reward={current_ep_reward:.2f} | "
                  f"steps={t} | max_sc={max_sc_episode:.4f} | "
                  f"success={'Yes' if episode_success else 'No'} | sr={sr:.2%}")
            print(env.get_action_summary())

    # Final save
    val_r, val_sr, val_l, val_sc = run_validation(agent, env, validation_episodes)
    monitor.log_validation(time_step, val_r, val_sr, val_l, val_sc)
    monitor.save_monitoring_data(agent.update_count)
    agent.save(checkpoint_path)
    with open(training_state_path, "w") as f:
        json.dump({
            "time_step":          time_step,
            "i_episode":          i_episode,
            "last_validation_ts": last_validation_ts,
            "update_count":       agent.update_count,
            "best_val_success":   best_val_success,
        }, f, indent=2)

    elapsed = datetime.now() - start_time
    print(f"\n DONE | time={elapsed} | episodes={i_episode} | model={checkpoint_path}")


if __name__ == "__main__":
    print("=" * 70)
    train()
    print("=" * 70)
