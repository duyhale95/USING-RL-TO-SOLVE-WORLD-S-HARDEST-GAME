import gym
from gym import spaces
import numpy as np
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 4 possible actions: left, right, up, down
        self.observation_space = spaces.Box(low=0, high=1, shape=(20, 20, 3), dtype=np.float32)
        
        self.snake_block = 20
        self.snake_speed = 15
        self.dis_width = 400
        self.dis_height = 400
        
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode((self.dis_width, self.dis_height))
        
        self.reset()
        
    def step(self, action):
        self._take_action(action)
        self.snake_head[0] += self.x1_change
        self.snake_head[1] += self.y1_change
        self.snake_List.append(list(self.snake_head))
        if len(self.snake_List) > self.Length_of_snake:
            del self.snake_List[0]

        reward = 0
        done = False
        if self.snake_head == self.food:
            reward = 1
            self._place_food()
            self.Length_of_snake += 1
        
        if (self.snake_head[0] >= self.dis_width or self.snake_head[0] < 0 or
            self.snake_head[1] >= self.dis_height or self.snake_head[1] < 0 or
            self.snake_head in self.snake_List[:-1]):
            done = True
            reward = -1
        
        self._update_ui()
        return self.state, reward, done, {}
    
    def reset(self):
        self.x1_change = 0
        self.y1_change = 0
        self.snake_head = [self.dis_width // 2, self.dis_height // 2]
        self.snake_List = [self.snake_head]
        self.Length_of_snake = 1
        self._place_food()
        self.state = np.zeros((20, 20, 3), dtype=np.float32)
        return self.state
    
    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        self.display.fill((0, 0, 0))
        
        for segment in self.snake_List:
            pygame.draw.rect(self.display, (0, 255, 0), [segment[0], segment[1], self.snake_block, self.snake_block])
        
        pygame.draw.rect(self.display, (255, 0, 0), [self.food[0], self.food[1], self.snake_block, self.snake_block])
        
        pygame.display.update()
        self.clock.tick(self.snake_speed)
    
    def close(self):
        pygame.quit()
        
    def _place_food(self):
        self.food = [random.randrange(0, self.dis_width // self.snake_block) * self.snake_block,
                     random.randrange(0, self.dis_height // self.snake_block) * self.snake_block]
    
    def _take_action(self, action):
        if action == 0:
            self.x1_change = -self.snake_block
            self.y1_change = 0
        elif action == 1:
            self.x1_change = self.snake_block
            self.y1_change = 0
        elif action == 2:
            self.x1_change = 0
            self.y1_change = -self.snake_block
        elif action == 3:
            self.x1_change = 0
            self.y1_change = self.snake_block
    
    def _update_ui(self):
        self.state = np.zeros((20, 20, 3), dtype=np.float32)
        for segment in self.snake_List:
            x, y = segment
            self.state[y // self.snake_block, x // self.snake_block] = [1, 1, 1]
        fx, fy = self.food
        self.state[fy // self.snake_block, fx // self.snake_block] = [1, 0, 0]

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(20 * 20 * 3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)
    
    def forward(self, x):
        x = x.view(-1, 20 * 20 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = QNetwork()
        self.target_model = QNetwork()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.target_model(next_state)[0]).item())
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    pygame.init()
    env = SnakeEnv()
    state_size = (20, 20, 3)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    episodes = 1000

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size[0], state_size[1], state_size[2]])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size[0], state_size[1], state_size[2]])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            env.render()
            if done:
                agent.update_target_model()
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    env.close()
