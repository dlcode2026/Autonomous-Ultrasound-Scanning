import warnings
warnings.filterwarnings("ignore")

import torch
import os
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import torchvision.transforms as transforms
from torchvision.models import resnet18
from gym import Env, spaces
import torch.nn.functional as F
from datetime import datetime
from torch.distributions import Categorical
import random
import math
import pandas as pd
from collections import deque
import json
import sys
from residual_gan import ConditionalGAN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

NORM_MIN = np.array([-0.144989, -0.487173,  0.198017,
                     -0.706294, -0.705661, -0.340682, -0.440026], dtype=np.float32)
NORM_MAX = np.array([ 0.010570, -0.272387,  0.274926,
                      0.996069,  0.999752,  0.388672,  0.377515],  # ← new dataset
                    dtype=np.float32)

WS_MIN = np.array([-0.144989, -0.487173,  0.198017], dtype=np.float32)
WS_MAX = np.array([ 0.010570, -0.272387,  0.22], dtype=np.float32)  # 0.274926

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
us_step_rx = 0.1 
us_step_ry = 0.1 
us_step_rz = 0.1 

Z_DIM     = 100
LABEL_DIM = 7

SC_CLASS_IDX         = 5
CLASS_NAMES          = ["5CH", "PLAX", "PSAX", "Random", "SC1", "SC2"]
NUM_CLASSES          = 6
SC_SUCCESS_THRESHOLD = 0.96

GAN_CHECKPOINT            = "conditional_gan_residual_epoch_100.pth"
CLASSIFICATION_MODEL_PATH = "classifier_best.pth"

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

def _random_unit_quat() -> np.ndarray:
    q = np.random.randn(4).astype(np.float32)
    return q / np.linalg.norm(q)


def domain_aware_normalize(values, norm_params):

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
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                     stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        print(f" Classifier loaded  (SC index={target_class_idx})")

        self.preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def classify_image(self, image):
        try:
            img_t = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(img_t)
                probs  = F.softmax(logits, dim=1)
                conf, pred = torch.max(probs, 1)

            return pred.item(), conf.item(), probs.squeeze().cpu().numpy()

        except Exception as e:
            print(f" Classifier error: {e}")
            return 0, 0.0, np.zeros(self.num_classes)


class UltrasoundImageGenerator:
    def __init__(self, checkpoint_path, z_dim=Z_DIM, label_dim=LABEL_DIM):
        self.device = device
        self.z_dim  = z_dim

        self.model = ConditionalGAN(z_dim=z_dim, label_dim=label_dim,
                                    img_channels=1).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f" GAN loaded  (epoch {ckpt.get('epoch','?')})")

        self.fixed_z = torch.randn(1, z_dim, device=device)
        print(self.fixed_z)

    def generate_tensor(self, normalised_pose: np.ndarray) -> torch.Tensor:
        label = torch.tensor(normalised_pose, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            gen = self.model.generator(self.fixed_z, label)
        return gen.squeeze(0).cpu()

    def generate_numpy(self, normalised_pose: np.ndarray) -> np.ndarray:
        t = self.generate_tensor(normalised_pose)
        return ((t.squeeze().numpy() + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)


image_generator = UltrasoundImageGenerator(GAN_CHECKPOINT)
classifier      = ImageClassifier(CLASSIFICATION_MODEL_PATH)

def preprocess_parameter_state(state):
    tensor = torch.from_numpy(state).float() \
             if isinstance(state, np.ndarray) else state.float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)

class ImageOnlyEnv(Env):

    def __init__(self, use_rotation=True):
        super().__init__()
        self.use_rotation = use_rotation

        self.steps_in_episode        = 0
        self.action_history          = []
        self.previous_sc_probability = 0.0

        # Default start — centre of workspace
        self.initial_position = (
            float(np.mean([WS_MIN[0], WS_MAX[0]])),
            float(np.mean([WS_MIN[1], WS_MAX[1]])),
            float(np.mean([WS_MIN[2], WS_MAX[2]])),
        )
        self.initial_rotation = (0.0, 0.0, -0.7071, 0.7071)
        self.reset_state()

        self.action_space      = spaces.Discrete(12 if use_rotation else 6)
        self.observation_space = spaces.Box(low=-2.0, high=2.0,
                                            shape=(7,), dtype=np.float32)

        self.action_names = [
            "move_x_neg", "move_x_pos",
            "move_y_neg", "move_y_pos",
            "move_z_neg", "move_z_pos",
            "rot_x_neg",  "rot_x_pos",
            "rot_y_neg",  "rot_y_pos",
            "rot_z_neg",  "rot_z_pos",
        ]

        print(" Environment Initialized")
        print(f"   Workspace  x:[{WS_MIN[0]:.3f},{WS_MAX[0]:.3f}]  "
              f"y:[{WS_MIN[1]:.3f},{WS_MAX[1]:.3f}]  "
              f"z:[{WS_MIN[2]:.3f},{WS_MAX[2]:.3f}]")
        print(f"   GAN norm   x:[{NORM_MIN[0]:.3f},{NORM_MAX[0]:.3f}]  "
              f"y:[{NORM_MIN[1]:.3f},{NORM_MAX[1]:.3f}]  "
              f"z:[{NORM_MIN[2]:.3f},{NORM_MAX[2]:.3f}]")
        print(f"   Done when SC prob >= {SC_SUCCESS_THRESHOLD}")

    def _load_target_images(self, image_dir):
        images = []
        for fname in sorted(os.listdir(image_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img = np.array(
                    Image.open(os.path.join(image_dir, fname))
                         .convert("L").resize((128, 128)),
                    dtype=np.uint8
                )
                images.append(img)
        if not images:
            raise RuntimeError(f"No target images in {image_dir}")
        print(f"   Target pool: {len(images)} images")
        return images

    def reset_state(self):
        self.current_position        = self.initial_position
        self.current_rotation        = self.initial_rotation
        self.steps_in_episode        = 0
        self.action_history          = []
        self.previous_sc_probability = 0.0

    def _get_state(self) -> np.ndarray:
        pose = np.array([*self.current_position, *self.current_rotation],
                        dtype=np.float32)
        return domain_aware_normalize(pose, norm_params)

    def _sc_prob_from_state(self, normalised_pose: np.ndarray) -> float:
        gen_tensor = image_generator.generate_tensor(normalised_pose)
        _, _, probs = classifier.classify_image(gen_tensor)
        return float(probs[SC_CLASS_IDX])


    def reset(self):
        self.initial_position = (
            random.uniform(float(WS_MIN[0]), float(WS_MAX[0])),
            random.uniform(float(WS_MIN[1]), float(WS_MAX[1])),
            random.uniform(float(WS_MIN[2]), float(WS_MAX[2])),
        )
        q = _random_unit_quat()
        self.initial_rotation = (float(q[0]), float(q[1]),
                                  float(q[2]), float(q[3]))

        self.reset_state()

        init_norm_state = self._get_state()

        assert np.all(np.abs(init_norm_state[:3]) <= 1.0 + 1e-4), (
            f"Normalised start pose out of range: {init_norm_state[:3]}. "
            f"Check that WS bounds are inside NORM bounds."
        )

        self.previous_sc_probability = self._sc_prob_from_state(init_norm_state)

        print(f" RESET | pos=({self.initial_position[0]:.3f},"
              f"{self.initial_position[1]:.3f},{self.initial_position[2]:.3f}) | "
              f"SC_init={self.previous_sc_probability:.4f}")
        return init_norm_state

    def step(self, action):
        self.steps_in_episode += 1
        action_name = (self.action_names[action]
                       if action < len(self.action_names)
                       else f"unknown_{action}")

        self.action_history.append({
            "step":            self.steps_in_episode,
            "action_id":       action,
            "action_name":     action_name,
            "position_before": self.current_position,
            "rotation_before": self.current_rotation,
        })

        new_position, new_rotation = self._apply_action(action)
        new_pose       = np.array([*new_position, *new_rotation], dtype=np.float32)
        new_norm_state = domain_aware_normalize(new_pose, norm_params)

        reward, done = self._calculate_reward(new_norm_state)

        self.current_position = new_position
        self.current_rotation = new_rotation
        self.action_history[-1]["position_after"] = new_position
        self.action_history[-1]["rotation_after"] = new_rotation

        info = {
            "position":       new_position,
            "rotation":       new_rotation,
            "steps":          self.steps_in_episode,
            "last_action":    action_name,
            "sc_probability": self.previous_sc_probability,
        }
        return new_norm_state, reward, done, info

    def _apply_action(self, action):
        x, y, z         = self.current_position
        qx, qy, qz, qw  = self.current_rotation

        # Translation
        if   action == 0: x -= us_step_x
        elif action == 1: x += us_step_x
        elif action == 2: y -= us_step_y
        elif action == 3: y += us_step_y
        elif action == 4: z -= us_step_z
        elif action == 5: z += us_step_z

        # Rotation
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

        x = float(np.clip(x, WS_MIN[0], WS_MAX[0]))
        y = float(np.clip(y, WS_MIN[1], WS_MAX[1]))
        z = float(np.clip(z, WS_MIN[2], WS_MAX[2]))

        return (round(x, 10), round(y, 10), round(z, 10)), \
               (round(qx, 10), round(qy, 10), round(qz, 10), round(qw, 10))

    def _calculate_reward(self, normalised_pose: np.ndarray):

        SCALE_FACTOR = 100.0

        gen_tensor = image_generator.generate_tensor(normalised_pose)
        _, _, probs = classifier.classify_image(gen_tensor)
        current_sc_prob = float(probs[SC_CLASS_IDX])

        # ── shaping ─────────────────────────────────────────────────
        current_potential  = current_sc_prob * SCALE_FACTOR
        previous_potential = self.previous_sc_probability * SCALE_FACTOR
        shaping_reward     = current_potential - previous_potential

        # ── Success check ─────────────────────────────────────────────────────
        base_reward = 0.0
        done        = False

        if current_sc_prob >= SC_SUCCESS_THRESHOLD:
            base_reward = 100.0
            done        = True
            print(f"SUCCESS  SC={current_sc_prob:.4f}  "
                  f"pos=({self.current_position[0]:.4f},"
                  f"{self.current_position[1]:.4f},"
                  f"{self.current_position[2]:.4f}),"
                  f"{self.current_rotation[0]:.4f},"
                  f"{self.current_rotation[1]:.4f},"
                  f"{self.current_rotation[2]:.4f}")

        step_penalty = -0.01

        if shaping_reward > 0:
            sc_reward = 1
        else:
            sc_reward = -1

        sc_reward += current_sc_prob
        total_reward = base_reward + step_penalty + sc_reward

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
        self.actions = []; self.states = []; self.logprobs = []
        self.rewards  = []; self.state_values = []; self.is_terminals = []

    def clear(self):
        del self.actions[:]; del self.states[:]; del self.logprobs[:]
        del self.rewards[:]; del self.state_values[:]; del self.is_terminals[:]


class ParameterOnlyActor(nn.Module):
    def __init__(self, input_dim=7, num_classes=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 256),       nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128,  64),       nn.ReLU(),
            nn.Linear( 64,  32),       nn.ReLU(),
            nn.Linear( 32, num_classes),
        )
    def forward(self, x): return F.softmax(self.net(x), dim=-1)


class ParameterOnlyCritic(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 256),       nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128,  64),       nn.ReLU(),
            nn.Linear( 64,  32),       nn.ReLU(),
            nn.Linear( 32,   1),
        )
    def forward(self, x): return self.net(x)


class ParameterOnlyActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,
                 has_continuous_action_space, action_std_init):
        super().__init__()
        if has_continuous_action_space:
            raise NotImplementedError("Use discrete actions")
        self.actor  = ParameterOnlyActor(input_dim=7, num_classes=action_dim).to(device)
        self.critic = ParameterOnlyCritic(input_dim=7).to(device)

    def set_action_std(self, _): pass
    def forward(self): raise NotImplementedError

    def act(self, state):
        probs  = self.actor(state)
        dist   = Categorical(probs)
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach(), \
               self.critic(state).detach()

    def evaluate(self, state, action):
        probs = self.actor(state)
        dist  = Categorical(probs)
        return dist.log_prob(action), self.critic(state), dist.entropy()


def create_image_env():
    return ImageOnlyEnv(use_rotation=True)

class MonitoredPPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma,
                 K_epochs, eps_clip, has_continuous_action_space,
                 action_std_init=0.6, monitor=None):

        self.gamma    = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.monitor  = monitor
        self.buffer   = RolloutBuffer()

        self.policy = ParameterOnlyActorCritic(
            state_dim, action_dim, has_continuous_action_space, action_std_init
        ).to(device)
        self.optimizer = torch.optim.Adam([
            {"params": self.policy.actor.parameters(),  "lr": lr_actor},
            {"params": self.policy.critic.parameters(), "lr": lr_critic},
        ])
        self.policy_old = ParameterOnlyActorCritic(
            state_dim, action_dim, has_continuous_action_space, action_std_init
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss      = nn.MSELoss()
        self.update_count = 0

    def select_action(self, state):
        with torch.no_grad():
            state_t = preprocess_parameter_state(state)
            action, logprob, val = self.policy_old.act(state_t)
        self.buffer.states.append(state_t)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        self.buffer.state_values.append(val)
        return action.item()

    def update(self, time_step=0, max_training_timesteps=int(10e6)):
        self.update_count += 1

        rewards    = []
        discounted = 0.0
        for r, terminal in zip(reversed(self.buffer.rewards),
                                reversed(self.buffer.is_terminals)):
            if terminal: discounted = 0.0
            discounted = r + self.gamma * discounted
            rewards.insert(0, discounted)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states  = torch.cat(self.buffer.states, dim=0).to(device)
        old_actions = torch.squeeze(
            torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(
            torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_values   = torch.squeeze(
            torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        advantages = rewards.detach() - old_values.detach()

        tot_a = tot_c = tot_e = 0.0
        for _ in range(self.K_epochs):
            logprobs, vals, entropy = self.policy.evaluate(old_states, old_actions)
            vals    = torch.squeeze(vals)
            ratios  = torch.exp(logprobs - old_logprobs.detach())
            surr1   = ratios * advantages
            surr2   = torch.clamp(ratios,
                                  1 - self.eps_clip,
                                  1 + self.eps_clip) * advantages

            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss =  0.5 * self.MseLoss(vals, rewards)
            # Anneal entropy coefficient over training
            ent_coef = max(0.001, 0.02 * (1.0 - time_step / max_training_timesteps))
            ent_loss = -ent_coef * entropy.mean()
            #ent_loss    = -0.02 * entropy.mean()
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

        self.data_dir       = os.path.join(log_dir, "monitoring_data")
        self.validation_dir = os.path.join(log_dir, "validation")
        self.summary_dir    = os.path.join(log_dir, "summary")
        for d in [self.data_dir, self.validation_dir, self.summary_dir]:
            os.makedirs(d, exist_ok=True)

        self.actor_losses    = []
        self.critic_losses   = []
        self.entropies       = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates   = []
        self.max_sc_values   = []

        self.validation_timesteps       = []
        self.validation_rewards         = []
        self.validation_success_rates   = []
        self.validation_episode_lengths = []
        self.validation_max_sc_values   = []

        self.recent_successes = deque(maxlen=100)
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
        self.success_rates.append(
            sum(self.recent_successes) / len(self.recent_successes))

    def log_validation(self, timestep, val_r, val_sr, val_l, val_sc):
        self.validation_timesteps.append(timestep)
        self.validation_rewards.append(float(np.mean(val_r)))
        self.validation_success_rates.append(float(val_sr))
        self.validation_episode_lengths.append(float(np.mean(val_l)))
        self.validation_max_sc_values.append(float(np.mean(val_sc)))
        print(f" Val @ {timestep}: reward={np.mean(val_r):.2f}  "
              f"success={val_sr:.2%}  max_sc={np.mean(val_sc):.4f}")

    def save_monitoring_data(self, step):
        data = {
            "update_step":     int(step),
            "actor_losses":    self.actor_losses,
            "critic_losses":   self.critic_losses,
            "entropies":       self.entropies,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "success_rates":   self.success_rates,
            "max_sc_values":   self.max_sc_values,
        }
        path = os.path.join(self.data_dir, f"training_{step}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_validation_json(self, timestep):
        data = {
            "saved_at_timestep":      int(timestep),
            "validation_timesteps":   self.validation_timesteps,
            "validation_rewards":     self.validation_rewards,
            "validation_success_rates": self.validation_success_rates,
            "validation_episode_lengths": self.validation_episode_lengths,
            "validation_max_sc_values": self.validation_max_sc_values,
        }
        path = os.path.join(self.validation_dir, f"val_{timestep}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f" Validation JSON saved → {path}")

    def generate_summary_report(self, step):

        if not self.episode_rewards:
            return
        recent = self.episode_rewards[-100:]
        sr = sum(self.recent_successes) / max(1, len(self.recent_successes))

        print(f"\n===== SUMMARY (step {step}) =====")
        print(f"  Avg reward (last 100): {np.mean(recent):.2f} ± {np.std(recent):.2f}")
        print(f"  Success rate:          {sr:.2%}")
        if self.entropies:
            print(f"  Entropy:               {self.entropies[-1]:.4f}")
        print("=" * 40)

        data = {
            "step":                     int(step),
            "num_episodes":             len(self.episode_rewards),
            "avg_reward_last100":       float(np.mean(recent)),
            "std_reward_last100":       float(np.std(recent)),
            "success_rate_last100":     float(sr),
            "latest_entropy":           float(self.entropies[-1]) if self.entropies else None,
            "latest_actor_loss":        float(self.actor_losses[-1])  if self.actor_losses  else None,
            "latest_critic_loss":       float(self.critic_losses[-1]) if self.critic_losses else None,
            "latest_max_sc":            float(self.max_sc_values[-1]) if self.max_sc_values else None,
        }
        path = os.path.join(self.summary_dir, f"summary_{step}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def force_generate_plots(self, step):

        self.save_monitoring_data(step)
        self.generate_summary_report(step)
        if self.validation_timesteps:
            self.save_validation_json(self.validation_timesteps[-1])

    def plot_losses(self, step):         self.save_monitoring_data(step)
    def plot_average_reward(self, step): self.save_monitoring_data(step)
    def plot_entropy(self, step):        self.save_monitoring_data(step)
    def plot_validation_only(self, timestep): self.save_validation_json(timestep)

def run_validation(ppo_agent, env, num_episodes=100):
    print(f"\n Validation ({num_episodes} episodes) …")
    val_rewards = []; val_lengths = []; val_successes = []; val_max_sc = []

    was_training = ppo_agent.policy.training
    ppo_agent.policy.eval()

    for ep in range(num_episodes):
        state     = env.reset()
        ep_reward = 0; ep_length = 0; done = False
        max_sc_ep = env.previous_sc_probability

        for _ in range(300):
            with torch.no_grad():
                state_t = preprocess_parameter_state(state)
                probs   = ppo_agent.policy.actor(state_t)
                action  = torch.argmax(probs).item()
            state, reward, done, _ = env.step(action)
            ep_reward += reward; ep_length += 1
            max_sc_ep  = max(max_sc_ep, env.previous_sc_probability)
            if done: break

        val_rewards.append(ep_reward); val_lengths.append(ep_length)
        val_successes.append(1 if done else 0); val_max_sc.append(max_sc_ep)
        if (ep + 1) % 20 == 0:
            print(f"   Val ep {ep+1}/{num_episodes}")

    if was_training:
        ppo_agent.policy.train()

    print(f"Val | reward={np.mean(val_rewards):.2f}  "
          f"success={np.mean(val_successes):.2%}  "
          f"max_sc={np.mean(val_max_sc):.4f}")
    return val_rewards, float(np.mean(val_successes)), val_lengths, val_max_sc


def train_image_with_monitoring():
    print(" QUATERNION DRL — IMAGE-BASED SC REWARD")
    print("=" * 70)

    image_env = create_image_env()

    max_ep_len             = 500 
    update_timestep        = 2048  
    validation_freq        = 10000
    validation_episodes    = 20
    K_epochs               = 10
    eps_clip               = 0.15 
    lr_actor               = 0.0001
    lr_critic              = 0.0005
    max_training_timesteps = int(15e6)
    gamma                  = 0.95
    print_freq             = max_ep_len * 1
    save_model_freq        = int(2e5)
    action_dim             = 12

    log_dir        = "scenario2_log_dir"
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True); os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "case2_sc.pth")

    monitor = TrainingMonitor(log_dir, save_freq=10)
    agent   = MonitoredPPO(
        state_dim=None, action_dim=action_dim,
        lr_actor=lr_actor, lr_critic=lr_critic,
        gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip,
        has_continuous_action_space=False, monitor=monitor,
    )

    # ── Resume from checkpoint if it exists ──────────────────────────────────
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f" Resumed from checkpoint: {checkpoint_path}")
    else:
        print("  No checkpoint found, starting from scratch")
    # ─────────────────────────────────────────────────────────────────────────

    start_time             = datetime.now()
    print_running_reward   = 0
    print_running_episodes = 0
    time_step              = 0
    i_episode              = 0
    last_validation_ts     = 0

    detailed_log = os.path.join(log_dir, f"training_log_{int(time.time())}.csv")
    with open(detailed_log, "w") as f:
        f.write("episode,timestep,total_reward,ep_length,success,"
                "max_sc_prob,actor_loss,critic_loss,entropy\n")

    while time_step <= max_training_timesteps:
        state             = image_env.reset()
        current_ep_reward = 0
        episode_success   = False
        max_sc_episode    = image_env.previous_sc_probability

        for t in range(1, max_ep_len + 1):
            action = agent.select_action(state)
            state, reward, done, info = image_env.step(action)

            max_sc_episode = max(max_sc_episode, image_env.previous_sc_probability)
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step         += 1
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                agent.update(time_step=time_step, 
                 max_training_timesteps=max_training_timesteps)
                monitor.save_monitoring_data(agent.update_count)
                if agent.update_count <= 5 or \
                        agent.update_count % monitor.save_freq == 0:
                    monitor.force_generate_plots(agent.update_count)
                elif agent.update_count % 25 == 0:
                    monitor.generate_summary_report(agent.update_count)

            if time_step % print_freq == 0:
                avg_r = print_running_reward / max(1, print_running_episodes)
                ent   = monitor.entropies[-1] if monitor.entropies else 0
                sr    = sum(monitor.recent_successes) / \
                        max(1, len(monitor.recent_successes))
                print(f"Ep {i_episode:6d} | Step {time_step:8d} | "
                      f"AvgR {avg_r:8.2f} | MaxSC {max_sc_episode:.4f} | "
                      f"Ent {ent:.4f} | SR {sr:.2%}")
                print_running_reward = 0; print_running_episodes = 0

            if time_step % save_model_freq == 0:
                agent.save(checkpoint_path)
                print(f" Checkpoint @ step {time_step}")

            if done:
                episode_success = True
                print(f" Ep {i_episode} SUCCESS @ step {t}  "
                      f"SC={image_env.previous_sc_probability:.4f}")
                break

        if time_step - last_validation_ts >= validation_freq:
            val_r, val_sr, val_l, val_sc = run_validation(
                agent, image_env, validation_episodes)
            monitor.log_validation(time_step, val_r, val_sr, val_l, val_sc)
            monitor.save_validation_json(time_step)   # ← was plot_validation_only
            last_validation_ts = time_step

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
            print(image_env.get_action_summary())

    # Final validation + save
    val_r, val_sr, val_l, val_sc = run_validation(
        agent, image_env, validation_episodes)
    monitor.log_validation(time_step, val_r, val_sr, val_l, val_sc)
    monitor.force_generate_plots(agent.update_count)
    agent.save(checkpoint_path)

    elapsed = datetime.now() - start_time
    print(f"\n DONE | time={elapsed} | episodes={i_episode} | "
          f"model={checkpoint_path}")


if __name__ == "__main__":
    print("=" * 70)
    train_image_with_monitoring()
    print("=" * 70)
