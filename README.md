# Self-Driving Car Simulation Using Reinforcement Learning

A Python, Gymnasium, and Stable-Baselines3 based project that trains a reinforcement learning agent to drive a car in the `CarRacing-v2` environment. The project uses the PPO algorithm with image observations, frame stacking, custom wrappers, TensorBoard logging, checkpoints, and a saved trained model for evaluation.

## What Is Self-Driving Car Simulation Using Reinforcement Learning

Self-driving car simulation using reinforcement learning is the process of training an agent to control a vehicle inside a simulated driving environment. The agent observes the road as image frames, takes driving actions, receives rewards, and gradually learns better driving behavior through trial and error.

This project focuses on:

- Training a PPO agent on the Gymnasium `CarRacing-v2` environment
- Using pixel-based observations instead of hand-written driving rules
- Saving checkpoints and the best trained model
- Evaluating the saved model with terminal, render, or video output
- Viewing training progress through TensorBoard logs

## How It Works

1. The `CarRacing-v2` environment generates a road track and car state.
2. Image observations are converted to grayscale and resized to `84x84`.
3. Four frames are stacked so the model can understand motion.
4. PPO learns continuous driving actions: steering, gas, and brake.
5. Training checkpoints, evaluation logs, and the best model are saved.
6. `evaluate.py` loads a saved model and runs it in the simulator.

## Saved Model Demo

The repository includes a saved best model at:

```text
models/best/best_model.zip
```

Run the saved model in the terminal:

```bash
python evaluate.py --model models/best/best_model --episodes 1 --verbose
```

Watch the trained car drive visually:

```bash
python evaluate.py --model models/best/best_model --render --episodes 5
```

Record evaluation videos:

```bash
python evaluate.py --model models/best/best_model --record --episodes 5
```

Important: pass the model path without `.zip`. The script checks for the `.zip` file automatically.

## Project Demo

<video src="assets/demo.mp4" width="100%" autoplay muted loop playsinline controls></video>

[Watch the demo video](assets/demo.mp4)

## Sample Output

Example terminal output from a one-episode evaluation:

```text
Episode   1/1  |  reward=   10.3  steps= 271

Mean: 10.30 +/- 0.00
Best: 10.30   Worst: 10.30
```

Scores can vary between runs because the environment can generate different tracks.

## Features

- PPO reinforcement learning agent
- Gymnasium `CarRacing-v2` simulation
- CNN policy for image-based driving
- Frame stacking for motion awareness
- Custom environment wrapper for reward clipping and early stopping
- Training checkpoints
- Best model saving
- TensorBoard training logs
- Terminal evaluation output
- Visual render mode
- Video recording support

## Applications

- Learning reinforcement learning concepts
- Understanding PPO for continuous control tasks
- Experimenting with image-based autonomous driving
- Testing reward shaping and environment wrappers
- Comparing training runs through TensorBoard
- Demonstrating self-driving behavior in a simulator

## Why This Project Is Useful

- Shows how reinforcement learning can control a simulated vehicle
- Provides a ready-to-run trained model
- Includes both training and evaluation scripts
- Helps visualize RL training progress with TensorBoard
- Gives a practical base for improving autonomous driving experiments

## Project Structure

```text
SELF-DRIVING CAR SIMULATION USING REINFORCEMENT LEARNING/
|-- train.py                 # Train PPO agent
|-- evaluate.py              # Evaluate or render a saved model
|-- config.py                # Training hyperparameters and paths
|-- wrappers.py              # Custom Gymnasium environment wrappers
|-- requirements.txt         # Python dependencies
|-- README.md                # Project documentation
|-- .gitignore               # Files and folders ignored by git
|-- models/
|   `-- best/
|       `-- best_model.zip   # Saved trained model
|-- checkpoints/             # Generated training checkpoints
|-- logs/
|   |-- tensorboard/         # TensorBoard event files
|   `-- eval/                # Evaluation logs
`-- videos/                  # Recorded evaluation videos
```

## Requirements

- Python 3.10 or later
- pip
- A virtual environment is recommended
- GPU is optional; CPU also works

Main Python packages:

- Gymnasium
- Box2D
- Pygame
- Stable-Baselines3
- Torch
- TensorBoard
- OpenCV
- NumPy
- tqdm

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import gymnasium; import stable_baselines3; print('OK')"
```

If Box2D installation fails on macOS or Linux, install SWIG first and then run the requirements command again.

## Run Locally

Run the saved trained model:

```bash
python evaluate.py --model models/best/best_model --episodes 1 --verbose
```

Run with visual output:

```bash
python evaluate.py --model models/best/best_model --render --episodes 5
```

Run with video recording:

```bash
python evaluate.py --model models/best/best_model --record --episodes 5
```

Recorded videos are saved in:

```text
videos/eval/
```

## View Training Logs

Start TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

Then open this URL in your browser:

[http://localhost:6006](http://localhost:6006)

Important metrics to watch:

- `rollout/ep_rew_mean`
- `train/policy_gradient_loss`
- `train/value_loss`
- `train/entropy_loss`
- `eval/mean_reward`

## Train The Model

Start a small training run:

```bash
python train.py --timesteps 100000 --n-envs 2
```

Train with the default configuration:

```bash
python train.py
```

Resume from the latest checkpoint:

```bash
python train.py --resume
```

Resume from a specific checkpoint:

```bash
python train.py --model-path checkpoints/ppo_car_20260430_232828_final
```

Training creates or updates:

- `checkpoints/`
- `models/best/best_model.zip`
- `logs/tensorboard/`
- `logs/eval/`

## Evaluation Options

Evaluate the best saved model:

```bash
python evaluate.py
```

Evaluate with per-episode terminal output:

```bash
python evaluate.py --verbose --episodes 5
```

Evaluate a checkpoint:

```bash
python evaluate.py --model checkpoints/ppo_car_20260430_232828_final --episodes 5
```

Render the checkpoint:

```bash
python evaluate.py --model checkpoints/ppo_car_20260430_232828_final --render --episodes 5
```

## PPO Algorithm

PPO stands for Proximal Policy Optimization. It is a reinforcement learning algorithm that updates the policy gradually so the agent does not change its behavior too aggressively after each batch of experience.

In this project:

- The policy receives stacked grayscale image frames
- The actor predicts driving actions
- The critic estimates how good the current state is
- PPO improves the policy using clipped updates
- The best model is selected through evaluation reward

## Scoring Guide

| Mean Reward | Interpretation |
| --- | --- |
| Less than 0 | Agent is stuck or driving backward |
| 100 to 300 | Learning to stay on track |
| 300 to 600 | Competent driving with missed corners |
| 600 to 800 | Good performance |
| 800 to 900 | Excellent performance |
| 900+ | Solved benchmark level |

## Tech Stack

- Python
- Gymnasium
- Box2D
- Pygame
- Stable-Baselines3
- PyTorch
- TensorBoard
- OpenCV
- NumPy

## Future Scope

- Improve reward shaping for smoother driving
- Train for more timesteps to improve performance
- Add custom driving tracks
- Add a web interface for model evaluation
- Compare PPO with SAC or TD3
- Add more visual result examples
- Add automated evaluation reports
- Tune hyperparameters for better scores

## Limitations

- Training can take a long time on CPU
- Results depend on random tracks and training duration
- Rendering requires a working display environment
- Box2D installation can require extra system dependencies
- The included saved model may not represent a fully solved driving agent

## Conclusion

This project demonstrates how reinforcement learning can be used to train a simulated self-driving car from visual observations. It provides a complete workflow for installing dependencies, training a PPO agent, viewing logs, saving models, and evaluating the trained driver.

## Notes

- Do not upload `venv/` or `.venv/`
- Do not upload `__pycache__/`
- Do not upload training logs unless they are intentionally needed
- Do not upload generated videos unless they are intentionally needed
- Install dependencies from `requirements.txt` after cloning
- Use model paths without `.zip` when running `evaluate.py --model`
