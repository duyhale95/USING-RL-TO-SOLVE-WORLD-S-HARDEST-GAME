import numpy as np
import random
import torch
import torch.nn as nn
import os

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=12, hidden_size=256, output_size=4):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size//2)
        self.linear3 = nn.Linear(hidden_size//2, output_size)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class GeneticAgent:
    def __init__(self, population_size=50):
        self.population_size = population_size
        self.population = [NeuralNetwork() for _ in range(population_size)]
        self.fitness_scores = np.zeros(population_size)
        self.best_fitness = 0
        self.best_model = None
        self.generation = 0
        self.epsilon = 80  # For exploration
        
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)),
            (dir_l and game.is_collision(point_l)),
            (dir_u and game.is_collision(point_u)),
            (dir_d and game.is_collision(point_d)),
            # Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        return np.array(state, dtype=int)
    
    def get_action(self, state, model_idx):
        # Add exploration like in DQL
        if random.randint(0, 200) < self.epsilon:
            final_move = [0,0,0,0]
            move = random.randint(0, 3)
            final_move[move] = 1
            return final_move
            
        state = torch.FloatTensor(state)
        model = self.population[model_idx]
        with torch.no_grad():
            q_values = model(state)
            move = torch.argmax(q_values).item()
            final_move = [0,0,0,0]
            final_move[move] = 1
        return final_move
    
    def update_fitness(self, model_idx, score, steps_survived):
        # Improved fitness function considering both score and survival
        fitness = score * 100 + steps_survived
        self.fitness_scores[model_idx] = fitness
        
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_model = self.population[model_idx]
            # Save best model
            self.best_model.save('genetic_best.pth')
    
    def crossover(self, parent1, parent2):
        child = NeuralNetwork()
        # Improved crossover with random weight interpolation
        for child_param, param1, param2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            ratio = torch.rand_like(child_param)
            child_param.data = ratio * param1.data + (1 - ratio) * param2.data
        return child
    
    def mutate(self, model, mutation_rate=0.1, mutation_strength=0.1):
        for param in model.parameters():
            mask = torch.rand_like(param) < mutation_rate
            mutation = torch.randn_like(param) * mutation_strength
            param.data += torch.where(mask, mutation, torch.zeros_like(param))
    
    def evolve(self):
        # Sort population by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        new_population = []
        
        # Elitism - keep top 10% unchanged
        elite_size = max(2, self.population_size // 10)
        for i in range(elite_size):
            new_population.append(self.population[sorted_indices[i]])
        
        # Tournament selection for parents
        while len(new_population) < self.population_size:
            # Tournament selection
            tournament_size = 5
            parent1_idx = sorted_indices[np.random.randint(0, tournament_size)]
            parent2_idx = sorted_indices[np.random.randint(0, tournament_size)]
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            child = self.crossover(parent1, parent2)
            self.mutate(child, mutation_rate=0.1, mutation_strength=0.2)
            new_population.append(child)
        
        self.population = new_population
        self.fitness_scores = np.zeros(self.population_size)
        self.generation += 1
        # Decrease epsilon for less exploration in later generations
        self.epsilon = max(0, 80 - self.generation) 