import torch
import numpy as np
from genetic_agent import GeneticAgent
from game_copy import Level1AI
from helper import plot

def train_genetic():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = GeneticAgent(population_size=50)
    
    # Training loop
    while True:
        # Train each individual in population
        for model_idx in range(agent.population_size):
            game = Level1AI()
            score = 0
            steps = 0
            steps_without_food = 0
            
            # Play game until done
            while True:
                # Get current state
                state = agent.get_state(game)
                
                # Get move from current model
                final_move = agent.get_action(state, model_idx)
                
                # Perform move
                reward, done, score = game.play_step(final_move)
                steps += 1
                
                if reward > 0:  # Got food
                    steps_without_food = 0
                else:
                    steps_without_food += 1
                
                # End game if stuck
                if steps_without_food > 100:
                    done = True
                
                if done:
                    break
            
            # Update fitness for this model
            agent.update_fitness(model_idx, score, steps)
            
            # Update display info
            if score > record:
                record = score
                print(f'New Record! Generation: {agent.generation}, Model: {model_idx}, Score: {score}')
        
        # Evolution step
        agent.evolve()
        
        # Calculate statistics
        gen_best_score = max(agent.fitness_scores) / 100  # Convert fitness back to score
        total_score += gen_best_score
        mean_score = total_score / (agent.generation + 1)
        
        # Update plots
        plot_scores.append(gen_best_score)
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores)
        
        print(f'Generation: {agent.generation}, Best Score: {gen_best_score}, Average Score: {mean_score:.2f}')

if __name__ == '__main__':
    train_genetic() 