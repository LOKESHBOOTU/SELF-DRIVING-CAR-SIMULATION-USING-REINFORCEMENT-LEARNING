# CarRacing-v2 PPO — Full Implementation

A complete Reinforcement Learning project that trains a PPO agent to drive a car
in OpenAI Gymnasium's CarRacing-v2 environment using pixel observations.

---

## Project Structure

```
car_racing_rl/
├── train.py          # Main training script
├── evaluate.py       # Load & evaluate / record trained agent
├── config.py         # All hyperparameters & presets
├── wrappers.py       # Custom Gymnasium wrappers
├── requirements.txt  # Dependencies
│
├── checkpoints/      # (created on run) periodic .zip saves
├── models/best/      # (created on run) best model by eval reward
├── logs/
│   ├── tensorboard/  # TensorBoard training curves
│   └── eval/         # Evaluation logs
└── videos/           # Recorded episodes
```

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (macOS) Install swig for Box2D
brew install swig

# 4. Verify installation
python -c "import gymnasium; import stable_baselines3; print('OK')"
```

---

## Training

```bash
# Train with default config (2M steps, 8 parallel envs)
python train.py

# Quick smoke test — verifies setup runs without errors (~1 min)
python train.py --timesteps 10000 --n-envs 2

# Resume from latest checkpoint
python train.py --resume

# Resume from a specific checkpoint
python train.py --model-path checkpoints/ppo_car_racing_500000_steps

# Train for longer
python train.py --timesteps 5000000
```

**Monitor training in TensorBoard:**
```bash
tensorboard --logdir logs/tensorboard
# Open http://localhost:6006
```

Key metrics to watch:
- `rollout/ep_rew_mean` — average episode reward (target: 900+)
- `train/policy_gradient_loss` — should decrease and stabilise
- `train/entropy_loss` — should stay slightly negative (exploration)

---

## Evaluation

```bash
# Evaluate best saved model (10 episodes, silent)
python evaluate.py

# Watch the agent drive live
python evaluate.py --render --episodes 5

# Record videos of 10 episodes
python evaluate.py --record --episodes 10

# Verbose per-episode output
python evaluate.py --verbose --episodes 20

# Evaluate a specific checkpoint
python evaluate.py --model checkpoints/ppo_car_racing_1000000_steps
```

---

## CarRacing-v2 Scoring Guide

| Mean Reward | Interpretation                          |
|-------------|-----------------------------------------|
| < 0         | Agent is stuck or driving backwards     |
| 100–300     | Learning to stay on track               |
| 300–600     | Competent driving, some corners missed  |
| 600–800     | Good performance, most laps completed   |
| 800–900     | Excellent, near-expert                  |
| 900+        | **Solved** (human-level benchmark)      |

---

## How It Works

### Environment
- **Observation**: 96×96 RGB image → grayscaled → resized to 84×84 → stacked 4 frames
- **Action space**: Continuous 3D vector `[steering, gas, brake]` ∈ [-1, 1]
- **Reward**: +1000/N per track tile visited (N = total tiles), -0.1 per frame

### Wrappers (applied in order)
1. `CarRacingWrapper` — skips zoom animation, early-stops stuck episodes, clips rewards
2. `GrayScaleObservation` — 3-channel RGB → 1-channel gray (3× fewer pixels)
3. `ResizeObservation` — 96×96 → 84×84 (standard DRL input size)
4. `VecFrameStack(n=4)` — stacks 4 frames so agent can infer velocity
5. `VecTransposeImage` — reorders axes for SB3's CNN policy

### PPO Algorithm
PPO (Proximal Policy Optimisation) clips the policy update ratio to prevent
destructively large updates. The key equation:

```
L_clip = E[ min(r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t) ]
```

Where `r_t = π_new(a|s) / π_old(a|s)` and ε = 0.2 (clip_range).

### CNN Policy Architecture
```
Input: (4, 84, 84) — 4 stacked grayscale frames

NatureCNN (default SB3 extractor):
  Conv2d(4→32,  kernel=8, stride=4)  → ReLU
  Conv2d(32→64, kernel=4, stride=2)  → ReLU
  Conv2d(64→64, kernel=3, stride=1)  → ReLU
  Flatten → Linear(3136→256)         → ReLU

Actor head:  Linear(256→64) → ReLU → Linear(64→3)  [mean of action distribution]
Critic head: Linear(256→64) → ReLU → Linear(64→1)  [state value V(s)]
```

---

## Tuning Tips

**Agent not improving after 500K steps?**
- Increase `ent_coef` to 0.05–0.1 to encourage more exploration
- Try a lower `learning_rate` (1e-4)

**Agent drives in circles?**
- Check that `skip_frames=50` is working (zoom animation skipped)
- Increase `max_negative_steps` penalty sensitivity

**Training is too slow?**
- Reduce `n_envs` to match your CPU core count
- Use `PRESETS["cpu"]` in config.py

**Want faster convergence?**
- Use `ActionRepeatWrapper(env, n_repeat=4)` — reduces decision frequency
- Pre-train with a smaller environment first

---

## Dependencies

See `requirements.txt`. Main packages:
- `gymnasium[box2d]` — environment
- `stable-baselines3[extra]` — PPO implementation
- `torch` — neural network backend
- `tensorboard` — training visualisation
