import torch
import random
import numpy as np
from collections import deque
from .model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class DQLAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(8, 256, 4)  # 8 input features, 4 actions
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # Calculate distances to nearest obstacles and goal
        dx = game.red_rect.centerx - game.player.player_x
        dy = game.red_rect.centery - game.player.player_y
        
        # Get distances to nearest enemies
        enemy_distances = []
        for enemy in [game.enemy, game.enemy2, game.enemy3, game.enemy4]:
            dx_e = enemy.player_x - game.player.player_x
            dy_e = enemy.player_y - game.player.player_y
            dist = (dx_e**2 + dy_e**2)**0.5
            enemy_distances.append(dist)
            
        state = [
            dx/game.screen_w,  # Normalized distance to goal x
            dy/game.screen_h,  # Normalized distance to goal y
            game.player.player_x/game.screen_w,  # Normalized player position x
            game.player.player_y/game.screen_h,  # Normalized player position y
        ] + [d/((game.screen_w**2 + game.screen_h**2)**0.5) for d in enemy_distances]  # Normalized enemy distances
        
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move 