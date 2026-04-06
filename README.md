# 🐦 Flappy Bird AI using Deep Reinforcement Learning (DQN)

This project implements a **Deep Q-Network (DQN)** agent using **PyTorch** to learn and play Flappy Bird autonomously. The agent improves over time using reinforcement learning techniques such as experience replay, epsilon-greedy exploration, and target networks.

---

## 🚀 Features

* ✅ Deep Q-Network (DQN) implemented from scratch
* ✅ Experience Replay for stable learning
* ✅ Epsilon-Greedy exploration strategy
* ✅ Target Network for stable Q-value updates
* ✅ Double DQN (DDQN) support
* ✅ Dueling DQN architecture
* ✅ Configurable hyperparameters via YAML
* ✅ Training visualization (reward & epsilon graphs)
* ✅ GPU acceleration (Apple MPS / CPU fallback)

---

## 🧠 How It Works

The agent interacts with the Flappy Bird environment and learns optimal actions through trial and error.

### Core Components:

* **State** → Environment observation (bird + pipes)
* **Action** → Flap or do nothing
* **Reward** → Score gained by passing pipes
* **Policy Network** → Predicts Q-values
* **Target Network** → Stabilizes training

---

## ⚙️ Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch gymnasium flappy-bird-gymnasium pygame matplotlib pyyaml
```

---

## ▶️ Usage

### 🔹 Train the Agent

```bash
python agent.py flappybird --train
```

### 🔹 Test the Agent

```bash
python agent.py flappybird
```

---

## 📁 Project Structure

```
RL_Mini_Project/
│
├── agent.py               # Main training loop
├── dqn.py                 # Neural network (DQN / DDQN / Dueling)
├── experience_replay.py   # Replay buffer
├── hyperparameters.yml    # Config file
├── runs/                  # Logs, models, graphs
└── README.md
```

---

## ⚙️ Hyperparameters (Example)

```yaml
flappybird:
  env_id: FlappyBird-v0
  replay_memory_size: 10000
  mini_batch_size: 32
  epsilon_init: 1.0
  epsilon_decay: 0.9995
  epsilon_min: 0.05
  network_sync_rate: 10
  learning_rate_a: 0.0001
  discount_factor_g: 0.99
  stop_on_reward: 100
  fc1_nodes: 512
  env_make_params:
    use_lidar: False
  enable_double_dqn: True
  enable_dueling_dqn: True
```

---

## 📊 Training Results

* 📈 Agent improves from random actions → intelligent gameplay
* 🧠 Achieved reward: **~90+**
* 📉 Learning stabilizes after long training

---

## 🔥 Key Learnings

* Reinforcement Learning requires careful tuning
* Experience Replay improves stability
* Target Networks reduce oscillations
* Double & Dueling DQN improve performance

---

## 🚀 Future Improvements

* 🎥 Record gameplay videos
* 📊 TensorBoard integration
* 🧠 CNN-based model for pixel input
* ⚡ Faster convergence with better reward shaping

---

## 🙌 Acknowledgement

This project is inspired by Deep Reinforcement Learning tutorials and Flappy Bird Gymnasium implementations.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!
