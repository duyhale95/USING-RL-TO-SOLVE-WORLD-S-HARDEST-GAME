import os
import sys
import pygame
import time

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from level3 import Level3
from level2 import Level2
from level1 import Level1
from DQL_Agent.dql_agent import DQLAgent
from GA_Agent.ga_agent import GAAgent

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Colors
BLACK = (0, 0, 0)
GRAY = (169, 169, 169)
BLUEISH_WHITE = (240, 248, 255)

# Fonts
pygame.font.init()
TITLE_FONT = pygame.font.Font(None, 74)
MENU_FONT = pygame.font.Font(None, 50)

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Main Menu")

# Main menu options
main_menu_options = ["Play as Human", "Play as DQN AI", "Play as Genetic AI", "Quit"]
level_options = ["Level 1", "Level 2", "Level 3"]
selected_option = 0

def draw_main_menu():
    screen.fill(BLUEISH_WHITE)
    title_surface = TITLE_FONT.render("World's Hardest Game", True, BLACK)
    title_rect = title_surface.get_rect(center=(WIDTH / 2, HEIGHT / 4))
    screen.blit(title_surface, title_rect)

    for i, option in enumerate(main_menu_options):
        color = BLACK if i == selected_option else GRAY
        text_surface = MENU_FONT.render(option, True, color)
        text_rect = text_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2 + i * 75))
        screen.blit(text_surface, text_rect)

    pygame.display.flip()

def train_dql_agent(game):
    agent = DQLAgent()
    record = 0
    total_score = 0
    
    # Setup game
    game.setup_tiles()
    game.reset()
    
    while True:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return

        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        state_new, reward, done = game.play_step(final_move)
        
        # Update game display
        game.draw()
        pygame.display.flip()
        
        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory (replay memory)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if reward > record:
                record = reward
                agent.model.save()

            print('Game', agent.n_games, 'Score', reward, 'Record:', record)
            
        # Control game speed
        time.sleep(0.05)

def train_genetic_agent(game, generations=50):
    agent = GAAgent()
    
    # Setup game
    game.setup_tiles()
    game.reset()
    
    for generation in range(generations):
        print(f"Generation {generation + 1}")
        for model_idx in range(agent.population_size):
            game.reset()
            done = False
            score = 0
            while not done:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return
                        
                state = agent.get_state(game)
                action = agent.get_action(state, model_idx)
                _, reward, done = game.play_step(action)
                score += reward
                
                # Update game display
                game.draw()
                pygame.display.flip()
                
                # Control game speed
                time.sleep(0.05)
                
            agent.update_fitness(model_idx, score)
        print(f"Best fitness: {agent.best_fitness}")
        agent.evolve()

def main_screen():
    global selected_option
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % len(main_menu_options)
                elif event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(main_menu_options)
                elif event.key == pygame.K_RETURN:
                    if main_menu_options[selected_option] == "Play as Human":
                        level1 = Level1()
                        level1.run()
                    elif main_menu_options[selected_option] == "Play as DQN AI":
                        level1 = Level1()
                        train_dql_agent(level1)
                    elif main_menu_options[selected_option] == "Play as Genetic AI":
                        level1 = Level1()
                        train_genetic_agent(level1)
                    elif main_menu_options[selected_option] == "Quit":
                        pygame.quit()
                        sys.exit()
        draw_main_menu()

if __name__ == "__main__":
    main_screen()
