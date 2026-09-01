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
from gym import Env, spaces


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


us_step_x  = 0.0150
us_step_y  = 0.0065
us_step_z  = 0.0045
us_step_rx = 0.05
us_step_ry = 0.05
us_step_rz = 0.05


LABEL_DIM             = 7   # x, y, z, qx, qy, qz, qw
POSE_SUCCESS_THRESHOLD = 0.05

# SC anchor pose (PLAX)
SC_ANCHOR_POSE = np.array([
    -0.08622503478295811,
    -0.38495659325215914,
     0.21505475810317343,
     0.8847871009515743,
    -0.35138325183583785,
    -0.21278587416989284,
     0.2200085636349992,
], dtype=np.float32)

# Pose normalisation bounds
NORM_MIN = np.array([-0.226865, -0.698484,  0.188854,
                     -0.706291, -0.706669, -0.355756, -0.305318], dtype=np.float32)
NORM_MAX = np.array([ 0.206803, -0.334000,  0.377331,
                       0.996767,  0.998525,  0.464118,  0.501245], dtype=np.float32)

norm_params = {"min": NORM_MIN, "max": NORM_MAX}

np.random.seed(42)

print(f" Norm params: min={norm_params['min'].round(3)}, max={norm_params['max'].round(3)}")
print(f"SC anchor quat: qx={SC_ANCHOR_POSE[3]:.4f}  qy={SC_ANCHOR_POSE[4]:.4f}  "
      f"qz={SC_ANCHOR_POSE[5]:.4f}  qw={SC_ANCHOR_POSE[6]:.4f}")


def _quat_multiply(q1, q2):
    """Multiply two quaternions q1 * q2, both in (qx, qy, qz, qw) format."""
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
    return (2 * (values - mn) / (mx - mn + 1e-8) - 1).astype(np.float32)


def compute_pose_distance_to_sc(position: tuple, quaternion: tuple) -> float:

    pos      = np.array(position, dtype=np.float32)
    mn, mx   = NORM_MIN[:3], NORM_MAX[:3]
    pos_norm = (2 * (pos               - mn) / (mx - mn + 1e-8) - 1)
    anc_norm = (2 * (SC_ANCHOR_POSE[:3] - mn) / (mx - mn + 1e-8) - 1)
    pos_dist = float(np.linalg.norm(pos_norm - anc_norm))

    q1  = np.array(quaternion, dtype=np.float32)
    q1 /= (np.linalg.norm(q1) + 1e-8)
    q2  = SC_ANCHOR_POSE[3:].copy()
    q2 /= (np.linalg.norm(q2) + 1e-8)
    dot      = float(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0))
    rot_dist = 1.0 - dot

    return float(math.sqrt(pos_dist**2 + rot_dist**2))


def preprocess_parameter_state(state):
    tensor = torch.from_numpy(state).float() if isinstance(state, np.ndarray) else state.float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


class PoseEnv(Env):

    def __init__(self, use_rotation=True):
        super().__init__()
        self.use_rotation     = use_rotation
        self.steps_in_episode = 0
        self.action_history   = []

        self.initial_position = (-0.01, -0.40, 0.21)
        self.initial_rotation = (0.0, 0.0, -0.7071, 0.7071)
        self.reset_state()

        self.action_space      = spaces.Discrete(12 if use_rotation else 6)
        self.observation_space = spaces.Box(low=-4.0, high=4.0, shape=(7,), dtype=np.float32)

        self.action_names = [
            "move_x_negative", "move_x_positive",
            "move_y_negative", "move_y_positive",
            "move_z_negative", "move_z_positive",
            "rotate_x_negative", "rotate_x_positive",
            "rotate_y_negative", "rotate_y_positive",
            "rotate_z_negative", "rotate_z_positive",
        ]

        print(" PoseEnv (Case 1) Initialized:")
        print(f"   State: 7-dim normalised pose [x,y,z,qx,qy,qz,qw]")
        print(f"   Actions: {12 if use_rotation else 6}")

    def reset_state(self):
        self.current_position   = self.initial_position
        self.current_rotation   = self.initial_rotation
        self.steps_in_episode   = 0
        self.previous_pose_dist = compute_pose_distance_to_sc(
            self.initial_position, self.initial_rotation
        )
        self.action_history = []

    def _get_state(self):
        pose = np.array([*self.current_position, *self.current_rotation], dtype=np.float32)
        return domain_aware_normalize(pose, norm_params)

    def step(self, action):
        self.steps_in_episode += 1
        action_name = self.action_names[action] if action < len(self.action_names) else f"unknown_{action}"
        self.action_history.append({
            "step":            self.steps_in_episode,
            "action_id":       action,
            "action_name":     action_name,
            "position_before": self.current_position,
            "rotation_before": self.current_rotation,
        })

        new_position, new_rotation, boundary_hit = self._apply_action(action)
        reward, done = self._calculate_reward(new_position, new_rotation, boundary_hit)

        self.current_position = new_position
        self.current_rotation = new_rotation
        self.action_history[-1]["position_after"] = new_position
        self.action_history[-1]["rotation_after"] = new_rotation

        observation = self._get_state()
        info = {
            "position":    new_position,
            "rotation":    new_rotation,
            "steps":       self.steps_in_episode,
            "last_action": action_name,
        }
        return observation, reward, done, info

    def _apply_action(self, action):
        boundary_hit = False
        x, y, z = self.current_position
        qx, qy, qz, qw = self.current_rotation

        if   action == 0: x -= us_step_x
        elif action == 1: x += us_step_x
        elif action == 2: y -= us_step_y
        elif action == 3: y += us_step_y
        elif action == 4: z -= us_step_z
        elif action == 5: z += us_step_z
        elif self.use_rotation and action == 6:
            dq = _axis_angle_to_quat('x', -us_step_rx)
            q_new = _quat_multiply(np.array([qx, qy, qz, qw]), dq)
            qx, qy, qz, qw = q_new / np.linalg.norm(q_new)
        elif self.use_rotation and action == 7:
            dq = _axis_angle_to_quat('x', +us_step_rx)
            q_new = _quat_multiply(np.array([qx, qy, qz, qw]), dq)
            qx, qy, qz, qw = q_new / np.linalg.norm(q_new)
        elif self.use_rotation and action == 8:
            dq = _axis_angle_to_quat('y', -us_step_ry)
            q_new = _quat_multiply(np.array([qx, qy, qz, qw]), dq)
            qx, qy, qz, qw = q_new / np.linalg.norm(q_new)
        elif self.use_rotation and action == 9:
            dq = _axis_angle_to_quat('y', +us_step_ry)
            q_new = _quat_multiply(np.array([qx, qy, qz, qw]), dq)
            qx, qy, qz, qw = q_new / np.linalg.norm(q_new)
        elif self.use_rotation and action == 10:
            dq = _axis_angle_to_quat('z', -us_step_rz)
            q_new = _quat_multiply(np.array([qx, qy, qz, qw]), dq)
            qx, qy, qz, qw = q_new / np.linalg.norm(q_new)
        elif self.use_rotation and action == 11:
            dq = _axis_angle_to_quat('z', +us_step_rz)
            q_new = _quat_multiply(np.array([qx, qy, qz, qw]), dq)
            qx, qy, qz, qw = q_new / np.linalg.norm(q_new)

        x = max(NORM_MIN[0], min(x, NORM_MAX[0]))
        y = max(NORM_MIN[1], min(y, NORM_MAX[1]))
        z = max(NORM_MIN[2], min(z, NORM_MAX[2]))

        return (round(x,10), round(y,10), round(z,10)), \
               (round(qx,10), round(qy,10), round(qz,10), round(qw,10)), boundary_hit

    def _calculate_reward(self, position, quaternion, boundary_hit):
        pose_dist_normalised = compute_pose_distance_to_sc(position, quaternion)
        prev_dist            = self.previous_pose_dist
        done                 = pose_dist_normalised <= POSE_SUCCESS_THRESHOLD
        reward               = 0

        if done:
            reward += 100.0
            print(f" TRUE SUCCESS pos=({position[0]:.4f},{position[1]:.4f},{position[2]:.4f}) "
                  f"quat=({quaternion[0]:.4f},{quaternion[1]:.4f},{quaternion[2]:.4f},{quaternion[3]:.4f})")
        else:
            improvement = prev_dist - pose_dist_normalised
            if improvement <= 0:
                reward -= 1

        self.previous_pose_dist = pose_dist_normalised
        return reward, done

    def reset(self):
        self.initial_position = (
            random.uniform(NORM_MIN[0], NORM_MAX[0]),
            random.uniform(NORM_MIN[1], NORM_MAX[1]),
            random.uniform(NORM_MIN[2], NORM_MAX[2]),
        )
        q = _random_unit_quat()
        self.initial_rotation = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        self.reset_state()
        initial_state = self._get_state()
        print(f" RESET | PoseDist to SC: {self.previous_pose_dist:.4f}")
        return initial_state

    def get_action_summary(self):
        if not self.action_history:
            return "No actions taken yet"
        counts = {}
        for r in self.action_history:
            counts[r["action_name"]] = counts.get(r["action_name"], 0) + 1
        summary = f"Episode Summary — {len(self.action_history)} steps:\n"
        for name, count in sorted(counts.items()):
            summary += f"  {name}: {count}x\n"
        return summary

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

    def forward(self, state):
        return F.softmax(self.net(state), dim=-1)


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

    def forward(self, state):
        return self.net(state)


class ParameterOnlyActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super().__init__()
        if has_continuous_action_space:
            raise NotImplementedError("Use discrete actions for pose-based state")
        self.actor  = ParameterOnlyActor(input_dim=7, num_classes=action_dim).to(device)
        self.critic = ParameterOnlyCritic(input_dim=7).to(device)

    def set_action_std(self, new_action_std): pass
    def forward(self): raise NotImplementedError

    def act(self, state):
        action_probs   = self.actor(state)
        dist           = Categorical(action_probs)
        action         = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val      = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs    = self.actor(state)
        dist            = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy    = dist.entropy()
        state_values    = self.critic(state)
        return action_logprobs, state_values, dist_entropy


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
            action, action_logprob, state_val = self.policy_old.act(state_t)
        self.buffer.states.append(state_t)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.item()

    def update(self):
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

        old_states       = torch.cat(self.buffer.states, dim=0).to(device)
        old_actions      = torch.squeeze(torch.stack(self.buffer.actions,      dim=0)).detach().to(device)
        old_logprobs     = torch.squeeze(torch.stack(self.buffer.logprobs,     dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        advantages = rewards.detach() - old_state_values.detach()

        total_actor = total_critic = total_ent = 0.0
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1  = ratios * advantages
            surr2  = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss =  0.5 * self.MseLoss(state_values, rewards)
            entropy_loss = -0.01 * dist_entropy.mean()
            loss         = actor_loss + critic_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_actor  += actor_loss.item()
            total_critic += critic_loss.item()
            total_ent    += dist_entropy.mean().item()

        avg_actor  = total_actor  / self.K_epochs
        avg_critic = total_critic / self.K_epochs
        avg_ent    = total_ent    / self.K_epochs

        if self.monitor:
            self.monitor.log_ppo_update(avg_actor, avg_critic, avg_ent)

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        return {"actor_loss": avg_actor, "critic_loss": avg_critic, "entropy": avg_ent}

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
        self.training_times  = []  # wall-clock seconds per episode

        self.validation_timesteps       = []
        self.validation_rewards         = []
        self.validation_success_rates   = []
        self.validation_episode_lengths = []
        self.validation_max_sc_values   = []

        self.recent_successes = deque(maxlen=100)
        self._start_time      = time.time()
        print(f" TrainingMonitor → {log_dir}")

    def log_ppo_update(self, actor_loss, critic_loss, entropy):
        self.actor_losses.append(float(actor_loss))
        self.critic_losses.append(float(critic_loss))
        self.entropies.append(float(entropy))

    def log_episode(self, episode_reward, episode_length, success, max_sc):
        self.episode_rewards.append(float(episode_reward))
        self.episode_lengths.append(int(episode_length))
        self.max_sc_values.append(float(max_sc))
        self.recent_successes.append(1 if success else 0)
        self.success_rates.append(sum(self.recent_successes) / len(self.recent_successes))
        self.training_times.append(float(time.time() - self._start_time))

    def log_validation(self, timestep, val_rewards, val_success_rate, val_lengths, val_max_sc):
        self.validation_timesteps.append(timestep)
        self.validation_rewards.append(float(np.mean(val_rewards)))
        self.validation_success_rates.append(float(val_success_rate))
        self.validation_episode_lengths.append(float(np.mean(val_lengths)))
        self.validation_max_sc_values.append(float(np.mean(val_max_sc)))
        print(f" Validation @ step {timestep}: reward={np.mean(val_rewards):.2f}  "
              f"success={val_success_rate:.2%}  dist={np.mean(val_max_sc):.4f}")

    def force_generate_plots(self, step):
        self.save_monitoring_data(step)

    def save_monitoring_data(self, step):
        data = {
            "case":            "case1_sc",
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


def run_validation(ppo_agent, env, num_episodes=20):
    print(f"\n Validation ({num_episodes} episodes) …")

    val_rewards   = []; val_lengths = []; val_successes = []; val_pose_dist = []

    was_training = ppo_agent.policy.training
    ppo_agent.policy.eval()

    for ep in range(num_episodes):
        state       = env.reset()
        ep_reward   = 0; ep_length = 0; done = False
        min_dist_ep = env.previous_pose_dist

        for _ in range(300):
            with torch.no_grad():
                state_t      = preprocess_parameter_state(state)
                action_probs = ppo_agent.policy.actor(state_t)
                action       = torch.argmax(action_probs).item()
            state, reward, done, _ = env.step(action)
            ep_reward   += reward; ep_length += 1
            min_dist_ep  = min(min_dist_ep, env.previous_pose_dist)
            if done: break

        val_rewards.append(ep_reward); val_lengths.append(ep_length)
        val_successes.append(1 if done else 0); val_pose_dist.append(min_dist_ep)
        if (ep + 1) % 10 == 0:
            print(f"   Val ep {ep+1}/{num_episodes}")

    if was_training:
        ppo_agent.policy.train()

    print(f" Val done | reward={np.mean(val_rewards):.2f} | "
          f"success={np.mean(val_successes):.2%} | "
          f"min_pose_dist={np.mean(val_pose_dist):.4f}")

    return val_rewards, float(np.mean(val_successes)), val_lengths, val_pose_dist

def train():
    print(" CASE 1 — Pose State, Distance Reward (SC)")
    print("=" * 70)

    env = PoseEnv(use_rotation=True)

    # Hyperparameters
    max_ep_len             = 500 #1000
    update_timestep        = 2048
    validation_freq        = 10000
    validation_episodes    = 20
    K_epochs               = 40
    eps_clip               = 0.2
    lr_actor               = 0.00008
    lr_critic              = 0.0004
    max_training_timesteps = int(10e6)
    gamma                  = 0.99
    print_freq             = max_ep_len * 1
    save_model_freq        = int(2e5)
    action_dim             = 12

    log_dir        = "scenario1_log_dir"
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(log_dir,        exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "case1_sc.pth")

    monitor = TrainingMonitor(log_dir, save_freq=10)
    agent   = MonitoredPPO(
        state_dim=None, action_dim=action_dim,
        lr_actor=lr_actor, lr_critic=lr_critic,
        gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip,
        has_continuous_action_space=False, monitor=monitor,
    )

    start_time             = datetime.now()
    print_running_reward   = 0
    print_running_episodes = 0
    time_step              = 0
    i_episode              = 0
    last_validation_ts     = 0

    detailed_log = os.path.join(log_dir, f"training_log_{int(time.time())}.csv")
    with open(detailed_log, "w") as f:
        f.write("episode,timestep,total_reward,ep_length,success,"
                "min_pose_dist,actor_loss,critic_loss,entropy\n")

    while time_step <= max_training_timesteps:
        state             = env.reset()
        current_ep_reward = 0
        episode_success   = False
        min_dist_episode  = env.previous_pose_dist

        for t in range(1, max_ep_len + 1):
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)

            min_dist_episode = min(min_dist_episode, env.previous_pose_dist)
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step         += 1
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                agent.update()
                monitor.save_monitoring_data(agent.update_count)
                if agent.update_count <= 5 or agent.update_count % monitor.save_freq == 0:
                    monitor.force_generate_plots(agent.update_count)
                elif agent.update_count % 25 == 0:
                    monitor.generate_summary_report(agent.update_count)

            if time_step % print_freq == 0:
                avg_r = print_running_reward / max(1, print_running_episodes)
                ent   = monitor.entropies[-1] if monitor.entropies else 0
                sr    = sum(monitor.recent_successes) / max(1, len(monitor.recent_successes))
                print(f"Ep {i_episode:6d} | Step {time_step:8d} | AvgR {avg_r:8.2f} | "
                      f"MinDist {min_dist_episode:.4f} | Ent {ent:.4f} | SR {sr:.2%}")
                print_running_reward = 0; print_running_episodes = 0

            if time_step % save_model_freq == 0:
                agent.save(checkpoint_path)
                print(f" Checkpoint @ step {time_step}")

            if done:
                episode_success = True
                print(f" Episode {i_episode} SUCCESS @ step {t}! "
                      f"PoseDist={env.previous_pose_dist:.4f}")
                break

        if time_step - last_validation_ts >= validation_freq:
            val_r, val_sr, val_l, val_dist = run_validation(agent, env, validation_episodes)
            monitor.log_validation(time_step, val_r, val_sr, val_l, val_dist)
            last_validation_ts = time_step

        monitor.log_episode(current_ep_reward, t, episode_success, min_dist_episode)

        al = monitor.actor_losses[-1]  if monitor.actor_losses  else 0
        cl = monitor.critic_losses[-1] if monitor.critic_losses else 0
        en = monitor.entropies[-1]     if monitor.entropies     else 0

        with open(detailed_log, "a") as f:
            f.write(f"{i_episode},{time_step},{current_ep_reward},{t},"
                    f"{episode_success},{min_dist_episode:.6f},"
                    f"{al:.6f},{cl:.6f},{en:.6f}\n")

        print_running_reward   += current_ep_reward
        print_running_episodes += 1
        i_episode              += 1

        if i_episode % 10 == 0:
            sr = sum(monitor.recent_successes) / max(1, len(monitor.recent_successes))
            print(f"\n Ep {i_episode} | reward={current_ep_reward:.2f} | "
                  f"steps={t} | dist={min_dist_episode:.4f} | "
                  f"success={'Yes' if episode_success else 'No'} | sr={sr:.2%}")
            print(env.get_action_summary())

    # Final
    val_r, val_sr, val_l, val_dist = run_validation(agent, env, validation_episodes)
    monitor.log_validation(time_step, val_r, val_sr, val_l, val_dist)
    monitor.save_monitoring_data(agent.update_count)
    agent.save(checkpoint_path)

    elapsed = datetime.now() - start_time
    print(f"\n DONE | time={elapsed} | episodes={i_episode} | model={checkpoint_path}")


if __name__ == "__main__":
    print("=" * 70)
    train()
    print("=" * 70)
