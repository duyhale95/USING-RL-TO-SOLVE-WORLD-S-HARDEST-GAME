import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from pytmx.util_pygame import load_pygame
from player import Player
pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
BLOCK_SIZE = 32
SPEED = 20

class SnakeGameAI:

    def __init__(self, w=1320, h=720):
        self.w = w
        self.h = h
        self.spawnpoint_x = 140
        self.spawnpoint_y = 275
        self.food_x = 300
        self.food_y = 500
        # init display
        # self.display = pygame.display.set_mode((self.w, self.h))
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Worlds Hardest Game')
        self.clock = pygame.time.Clock()
        self.hascollided = False
        self.reset()
        # for tiles
        self.tmx_data = load_pygame('Tiles/Level1.tmx')
        self.sprite_group = pygame.sprite.Group()
        self.tile_rect = []
        self.collider_rects = [] # List of tile colliders
        self.can_move_left = True
        self.can_move_up = True
        self.can_move_down = True
        self.setup_tiles()
        # Values
        
    class Tile(pygame.sprite.Sprite):
        def __init__(self, pos, surf, groups):
            super().__init__(groups)
            self.image = surf
            self.rect = self.image.get_rect(topleft=pos)


    def setup_tiles(self):
        for layer in self.tmx_data.visible_layers:
            if hasattr(layer, 'data'):
                if layer.name == "Main":
                    for x_val, y_val, surf in layer.tiles():
                        pos = (x_val * 64, y_val * 64)
                        self.Tile(pos=pos, surf=surf, groups=self.sprite_group)
                        self.tile_rect.append(pygame.Rect(x_val * 64, y_val * 64, 64, 64))
                        self.collider_rects.append(pygame.Rect(x_val * 64, y_val * 64, 64, 64))

        for layer in self.tmx_data.visible_layers:
            if hasattr(layer, 'data'):
                for x_val, y_val, surf in layer.tiles():
                    pos = (x_val * 64, y_val * 64)
                    self.Tile(pos=pos, surf=surf, groups=self.sprite_group)

    def drawColliders(self):
        pass
    
    def reset(self):
        
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.spawnpoint_x, self.spawnpoint_y)
        self.snake = [self.head]

        self.score = 0
        self.food = None
        self.food_x = 300
        self.food_y = 500
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        
        self.food = Point(self.food_x, self.food_y)
        self.food_rect = pygame.Rect(self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE)
        self.food_x = 1050
        self.food_y = 200
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1
        
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*(self.score+10):
            self.hascollided = False
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        food_collide = pygame.Rect.colliderect(self.head_rect, self.food_rect)
        if food_collide:
            self.score += 1
            reward = 10
            self._place_food()
            self.snake.pop()
        else:
            self.snake.pop()
            # reward = self.food.x - self.head.y
        index = 0
        pt = self.head
        
        for b in self.tile_rect:
            cleft_point = pygame.Rect.colliderect(self.left_point, self.collider_rects[index])
            if cleft_point:
                action = [0,0,1,0]
                self.can_move_left = False
            elif not cleft_point:
                self.can_move_left = True
            c_up_point = pygame.Rect.colliderect(self.up_point, self.collider_rects[index])
            if c_up_point:
                action = [0,1,0,0]
                self.can_move_up = False
            elif not c_up_point:
                self.can_move_up = True
            c_down_point = pygame.Rect.colliderect(self.down_point, self.collider_rects[index])
            if c_down_point:
                action = [0,0,0,1]
                self.can_move_down = False
            elif not c_down_point:
                self.can_move_down = True
            index += 1
        # 5. update ui and clock
        self.drawColliders()
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        if self.hascollided == True:
            return True
        return False


    def _update_ui(self):
        
        self.screen.fill("white")
        self.sprite_group.draw(self.screen)
        for pt in self.snake:
            pygame.draw.rect(self.screen, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.screen, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.screen, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.screen.blit(text, [0, 0])
        pygame.display.update()



    def _move(self, action):
        # [right, down, left, up]
        self.head_rect = pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE*2, BLOCK_SIZE*1.2)  
        self.left_point = pygame.Rect(self.head.x + BLOCK_SIZE + 20, self.head.y + BLOCK_SIZE / 2, 40, 40)
        self.left_point.topleft = (self.head.x + BLOCK_SIZE + 20, self.head.y + BLOCK_SIZE)
        self.up_point = pygame.Rect(self.head.x + BLOCK_SIZE / 2, self.head.y - 1, 40, 40)
        self.up_point.topleft = (self.head.x + BLOCK_SIZE / 2, self.head.y -20)
        self.down_point = pygame.Rect(self.head.x + BLOCK_SIZE / 2, self.head.y + BLOCK_SIZE, 40, 40)
        self.down_point.topleft = (self.head.x + BLOCK_SIZE / 2, self.head.y + BLOCK_SIZE + 20)

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        if np.array_equal(action, [1, 0, 0, 0]):
            if not self.can_move_left:
                x -= SPEED
                new_dir = clock_wise[2] # left    
            else:
                new_dir = clock_wise[0] # right
        elif np.array_equal(action, [0, 1, 0, 0]):
            if not self.can_move_down:
                y -= SPEED
                new_dir = clock_wise[3] # up
                
            else:
                new_dir = clock_wise[1] # down
        elif np.array_equal(action, [0, 0, 1, 0]):
            new_dir = clock_wise[2] # left
        elif np.array_equal(action, [0, 0, 0, 1]):
            if not self.can_move_up:
                y += SPEED
                new_dir = clock_wise[1] # down
                
            else:
                new_dir = clock_wise[3] # up
            
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += SPEED
        elif self.direction == Direction.LEFT:
            x -= SPEED
        elif self.direction == Direction.DOWN:
            y += SPEED
        elif self.direction == Direction.UP:
            y -= SPEED

        self.head = Point(x, y)
