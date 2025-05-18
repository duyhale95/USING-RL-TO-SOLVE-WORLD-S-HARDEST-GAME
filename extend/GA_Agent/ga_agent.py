import numpy as np
import random
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=8, hidden_size=24, output_size=4):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class GAAgent:
    def __init__(self, population_size=50):
        self.population_size = population_size
        self.population = [NeuralNetwork() for _ in range(population_size)]
        self.fitness_scores = np.zeros(population_size)
        self.best_fitness = 0
        self.best_model = None
        self.generation = 0
        
    def get_action(self, state, model_idx):
        state = torch.FloatTensor(state)
        model = self.population[model_idx]
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()
        return action
    
    def update_fitness(self, model_idx, score):
        self.fitness_scores[model_idx] = score
        if score > self.best_fitness:
            self.best_fitness = score
            self.best_model = self.population[model_idx]
    
    def crossover(self, parent1, parent2):
        child = NeuralNetwork()
        # Perform crossover for each layer
        for child_param, param1, param2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            mask = torch.rand_like(child_param) < 0.5
            child_param.data = torch.where(mask, param1.data, param2.data)
        return child
    
    def mutate(self, model, mutation_rate=0.1):
        for param in model.parameters():
            mask = torch.rand_like(param) < mutation_rate
            mutation = torch.randn_like(param) * 0.1
            param.data += torch.where(mask, mutation, torch.zeros_like(param))
    
    def evolve(self):
        # Sort population by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        new_population = []
        
        # Keep top 10% of population
        elite_size = self.population_size // 10
        for i in range(elite_size):
            new_population.append(self.population[sorted_indices[i]])
        
        # Create rest of population through crossover and mutation
        while len(new_population) < self.population_size:
            parent1_idx = random.randint(0, elite_size-1)
            parent2_idx = random.randint(0, elite_size-1)
            parent1 = self.population[sorted_indices[parent1_idx]]
            parent2 = self.population[sorted_indices[parent2_idx]]
            
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        self.fitness_scores = np.zeros(self.population_size)
        self.generation += 1
        
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