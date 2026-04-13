import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_class_weights(pi_pred_pos: float, rho: float):
    pi_pos_clamped = np.clip(pi_pred_pos, 1e-3, 1 - 1e-3)
    w_pos = rho / pi_pos_clamped
    w_neg = (1.0 - rho) / (1.0 - pi_pos_clamped)
    return w_pos, w_neg


def build_sample_weights_from_pseudo_labels(pseudo_labels, w_pos, w_neg):
    return np.where(np.asarray(pseudo_labels) == 1, w_pos, w_neg).astype(np.float64)


class RLSampleSelectionEnv:
    def __init__(
        self,
        mean_log_probs: torch.Tensor,
        predictive_entropy: torch.Tensor,
        mutual_info: torch.Tensor,
        class_weights: np.ndarray,
        budget: int,
        lambda_diversity: float = 0.01,
    ):
        self.mean_log_probs = mean_log_probs.clone().detach()
        self.predictive_entropy = predictive_entropy.clone().detach()
        self.mutual_info = mutual_info.clone().detach()

        self.N, self.C = self.mean_log_probs.shape
        self.budget = budget
        self.lambda_diversity = lambda_diversity

        mi_np = self.mutual_info.cpu().numpy()
        mi_min, mi_max = mi_np.min(), mi_np.max()
        denom = mi_max - mi_min
        if denom < 1e-8:
            mi_norm = np.zeros_like(mi_np, dtype=np.float64)
        else:
            mi_norm = (mi_np - mi_min) / (denom + 1e-8)

        self.utility_np = mi_norm * class_weights
        self.logit_features_np = self.mean_log_probs.cpu().numpy().astype(np.float64)

        self.state_dim = self.C + 2 + 1
        self.reset()

    def reset(self):
        self.idx = 0
        self.selected_indices = []
        self.done = False
        return self._get_state()

    def _selected_fraction(self):
        return len(self.selected_indices) / max(self.budget, 1)

    def _get_state(self):
        if self.idx >= self.N:
            return np.zeros(self.state_dim, dtype=np.float32)

        state = np.concatenate([
            self.mean_log_probs[self.idx].cpu().numpy(),
            np.array([
                float(self.predictive_entropy[self.idx].item()),
                float(self.mutual_info[self.idx].item()),
                self._selected_fraction(),
            ], dtype=np.float64)
        ])
        return state.astype(np.float32)

    def _redundancy_penalty(self, current_idx: int):
        if len(self.selected_indices) == 0:
            return 0.0
        x = self.logit_features_np[current_idx]
        selected = self.logit_features_np[self.selected_indices]
        dists = np.linalg.norm(selected - x, axis=1)
        delta = float(np.min(dists))
        return 1.0 / (1.0 + delta)

    def step(self, action: int):
        if self.done:
            return self._get_state(), 0.0, True

        reward = 0.0
        if action == 1 and len(self.selected_indices) < self.budget:
            red = self._redundancy_penalty(self.idx)
            reward = float(self.utility_np[self.idx] - self.lambda_diversity * red)
            self.selected_indices.append(self.idx)

        self.idx += 1

        if self.idx >= self.N or len(self.selected_indices) >= self.budget:
            self.done = True

        return self._get_state(), reward, self.done


class DQN(nn.Module):
    def __init__(self, state_dim: int, n_actions: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        h = self.feature(x)
        v = self.value_head(h)
        a = self.adv_head(h)
        return v + a - a.mean(dim=1, keepdim=True)


def train_rl_selector_for_prior(
    mean_log_probs,
    predictive_entropy,
    mutual_info,
    class_weights,
    budget,
    num_episodes=300,
    lambda_diversity=0.01,
    gamma=0.95,
    batch_size=64,
    lr=1e-4,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    replay_buffer_size=10000,
    target_update_every=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    env = RLSampleSelectionEnv(
        mean_log_probs=mean_log_probs,
        predictive_entropy=predictive_entropy,
        mutual_info=mutual_info,
        class_weights=class_weights,
        budget=budget,
        lambda_diversity=lambda_diversity,
    )

    agent = DQN(env.state_dim, 2).to(device)
    target_agent = DQN(env.state_dim, 2).to(device)
    target_agent.load_state_dict(agent.state_dict())

    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    replay_buffer = deque(maxlen=replay_buffer_size)

    epsilon = epsilon_start
    episode_rewards = []
    best_avg_reward = -float("inf")
    best_state_dict = None

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = agent(s_t).argmax(dim=1).item()

            next_state, reward, done = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                b_s, b_a, b_r, b_ns, b_d = zip(*batch)

                b_s = torch.FloatTensor(np.array(b_s)).to(device)
                b_a = torch.LongTensor(b_a).unsqueeze(1).to(device)
                b_r = torch.FloatTensor(b_r).unsqueeze(1).to(device)
                b_ns = torch.FloatTensor(np.array(b_ns)).to(device)
                b_d = torch.FloatTensor(b_d).unsqueeze(1).to(device)

                q_curr = agent(b_s).gather(1, b_a)
                with torch.no_grad():
                    q_next = target_agent(b_ns).max(dim=1)[0].unsqueeze(1)
                target = b_r + gamma * q_next * (1 - b_d)

                loss = F.mse_loss(q_curr, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        episode_rewards.append(total_reward)

        if len(episode_rewards) >= 10:
            avg_r = float(np.mean(episode_rewards[-10:]))
            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                best_state_dict = {k: v.cpu().clone() for k, v in agent.state_dict().items()}

        if ep % target_update_every == 0:
            target_agent.load_state_dict(agent.state_dict())

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    if best_state_dict is not None:
        agent.load_state_dict(best_state_dict)

    return agent, env, best_avg_reward


def select_with_trained_agent(agent, env, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agent.eval()
    state = env.reset()
    done = False

    while not done:
        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = agent(s_t).argmax(dim=1).item()
        next_state, _, done = env.step(action)
        state = next_state

    selected = list(dict.fromkeys(env.selected_indices))
    return np.array(selected, dtype=int)