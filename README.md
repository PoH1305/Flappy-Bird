# Flappy Bird with Deep Q-Network (DQN)

A Deep Reinforcement Learning implementation of **DQN (Deep Q-Network)** applied to the classic **Flappy Bird** game using [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium).

---

## What is DQN?

DQN is an **off-policy** deep reinforcement learning algorithm that combines Q-Learning with a neural network to approximate the Q-value function. Unlike tabular methods (like SARSA), DQN can handle continuous or large state spaces by learning a function approximator.

**Key components used:**
- **Policy Network** — selects actions based on current state
- **Target Network** — provides stable Q-value targets during training
- **Experience Replay** — stores past transitions and samples mini-batches to break correlation between updates

**Update Rule:**
```
Q(s, a) ← Q(s, a) + α * [r + γ * max Q_target(s', a') - Q(s, a)]
```

---

## Project Structure

```
├── agent.py                  # Main training & inference logic
├── dqn.py                    # Neural network architecture
├── experience_replay.py      # Replay memory buffer
├── parameters.yaml           # Hyperparameters
└── runs/                     # Auto-created — saves model & logs
    ├── flappybird-v0.pt      # Saved model weights
    └── flappybird-v0.log     # Best reward log
```

---

## Neural Network Architecture

```
Input (state_dim)  →  Linear(256)  →  ReLU  →  Linear(action_dim)
```

| Layer | Size |
|-------|------|
| Input | 12 (observation space) |
| Hidden | 256 |
| Output | 2 (flap / no flap) |

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 0.001 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon_init` | 1.0 | Starting exploration rate |
| `epsilon_min` | 0.05 | Minimum exploration rate |
| `epsilon_decay` | 0.9995 | Epsilon decay per episode |
| `replay_memory_size` | 100,000 | Max transitions stored |
| `min_batch_size` | 32 | Batch size for training |
| `network_sync_rate` | 10 | Steps between target network sync |
| `reward_threshold` | 1000 | Max reward before episode ends early |

---

## Requirements

```bash
pip install gymnasium torch flappy-bird-gymnasium pyyaml
```

For Windows with Intel/AMD GPU (no NVIDIA):
```bash
pip install torch-directml
```

---

## How to Run

### Train
```bash
python agent.py flappybird-v0 --train
```

### Test (watch the agent play)
```bash
python agent.py flappybird-v0
```

---

## Reward Structure

This environment gives small continuous rewards per frame:

| Event | Reward |
|-------|--------|
| Staying alive (per frame) | ~+0.1 |
| Passing a pipe | ~+1.0 |
| Dying (hitting pipe/ground) | ~-1.0 |

Early in training, total episode rewards will be negative (e.g. `-9.3`, `-5.0`). As the agent learns, rewards gradually increase toward positive values. A reward of `+1.5` means the agent successfully passed a pipe.

---

## Device Support

The agent automatically selects the best available device:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")       # NVIDIA GPU
elif torch.xpu.is_available():
    device = torch.device("xpu")        # Intel Arc GPU
else:
    device = torch.device("cpu")        # CPU fallback
```

For Intel integrated graphics on Windows, use `torch-directml` as an alternative.

---

## References

- Mnih et al. — *Playing Atari with Deep Reinforcement Learning* (2013)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Flappy Bird Gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)
