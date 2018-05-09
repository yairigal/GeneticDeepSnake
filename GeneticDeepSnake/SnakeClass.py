import sys

sys.path.append('../')
import random
import math
import pygame


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class Snake:
    def __init__(self, height, width, snake_dir="../creeper.png", food_dir="../food.gif", moving_func=None,
                 refresh_rate=100, max_steps=None, simulation=False, food_values=None):
        self.simulation = simulation
        self.moving_function = moving_func
        self.snake_dir = snake_dir
        self.food_dir = food_dir
        snake = pygame.image.load(snake_dir)
        snake_rect = snake.get_rect()
        self.snake_size = snake_rect.height
        self.refresh_rate = refresh_rate
        self.food = pygame.image.load(food_dir)
        self.food_height = self.food.get_rect().height
        self.speed = [[0, 0]]
        self.background_color = 30, 63, 159
        self.moving_speed = self.snake_size
        self.game_over = False
        self.last_direction = None
        self.height = height * self.snake_size
        self.width = width * self.snake_size
        snake_rect.x = (self.width - self.snake_size) / 2
        snake_rect.y = (self.height - self.snake_size) / 2
        if not simulation:
            self.snake = [[snake, snake_rect]]
        else:
            self.snake = [(snake_rect.x, snake_rect.y)]
        if not max_steps:
            max_steps = height * width
        self.max_steps = max_steps
        self.food_values = food_values

        if not simulation:
            pygame.init()
            pygame.font.init()
            self.endgame_text = pygame.font.SysFont('Comic Sans MS', 30)
            self.screen = pygame.display.set_mode((self.width, self.height))
        self.food_cords = self.generate_random_cords()

    def generate_random_cords(self):
        if not self.food_values:
            x = random.choice(range(0, self.width - self.snake_size, self.snake_size))
            y = random.choice(range(0, self.height - self.snake_size, self.snake_size))
            return x, y
        else:
            return self.food_values.pop()

    def reset(self):
        self.speed = [[0, 0]]
        self.food_cords = self.generate_random_cords()
        self.game_over = False
        self.last_direction = None
        if not self.simulation:
            snake = self.snake[0][0]
            snake_rect = snake.get_rect()
            snake_rect.x = (self.width - self.snake_size) / 2
            snake_rect.y = (self.height - self.snake_size) / 2
            self.snake = [[snake, snake_rect]]
        else:
            x = (self.width - self.snake_size) / 2
            y = (self.height - self.snake_size) / 2
            self.snake = [(x, y)]

    def start(self):
        i = 0
        max_steps = self.max_steps
        last_size = 1
        while True:
            i += 1
            if last_size < len(self.snake):
                last_size = len(self.snake)
                i = 1
            self.player_events()
            self.move_snake()
            self.check_food_collision()
            if not self.simulation:
                self.refresh_screen()
            self.check_snake_collision()
            self.check_boundaries_collision()
            if self.game_over or i >= max_steps:
                if not self.simulation:
                    x, y = self.snake[0][1].x, self.snake[0][1].y
                else:
                    x, y = self.snake[0][0], self.snake[0][1]
                return len(self.snake), self.game_over, i, distance((x, y), self.food_cords)

    def player_events(self):
        if not self.moving_function:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and self.last_direction != pygame.K_DOWN and self.last_direction != pygame.K_UP:
                        self.speed[0] = [0, -self.moving_speed]
                    if event.key == pygame.K_DOWN and self.last_direction != pygame.K_UP and self.last_direction != pygame.K_DOWN:
                        self.speed[0] = [0, self.moving_speed]
                    if event.key == pygame.K_RIGHT and self.last_direction != pygame.K_LEFT and self.last_direction != pygame.K_RIGHT:
                        self.speed[0] = [self.moving_speed, 0]
                    if event.key == pygame.K_LEFT and self.last_direction != pygame.K_RIGHT and self.last_direction != pygame.K_LEFT:
                        self.speed[0] = [-self.moving_speed, 0]
                    self.last_direction = event.key
        else:
            """
            0 - up
            1 - down
            2 - right
            3 - left
            """
            direction = self.moving_function(self.get_snake_input())
            if direction == 0 and self.last_direction != 1 and self.last_direction != 0:
                self.speed[0] = [0, -self.moving_speed]
            elif direction == 1 and self.last_direction != 0 and self.last_direction != 1:
                self.speed[0] = [0, self.moving_speed]
            elif direction == 2 and self.last_direction != 3 and self.last_direction != 2:
                self.speed[0] = [self.moving_speed, 0]
            elif direction == 3 and self.last_direction != 2 and self.last_direction != 3:
                self.speed[0] = [-self.moving_speed, 0]
            self.last_direction = direction

    def move_snake(self):
        last_cords = self.move_head()
        self.move_body(last_cords)
        self.update_speed()

    def move_body(self, last_cords):
        for i in range(1, len(self.snake)):
            if not self.simulation:
                temp = self.snake[i][1].x, self.snake[i][1].y
                self.snake[i][1].x = last_cords[0]
                self.snake[i][1].y = last_cords[1]
                last_cords = temp
            else:
                temp = self.snake[i][0], self.snake[i][1]
                self.snake[i] = last_cords
                last_cords = temp

    def move_head(self):
        if not self.simulation:
            last_cords = self.snake[0][1].x, self.snake[0][1].y
            self.snake[0][1] = self.snake[0][1].move(self.speed[0])
        else:
            last_cords = self.snake[0][0], self.snake[0][1]
            x, y = self.snake[0]
            x += self.speed[0][0]
            y += self.speed[0][1]
            self.snake[0] = (x, y)
        return last_cords

    def update_speed(self):
        for i in reversed(range(1, len(self.snake))):
            self.speed[i] = self.speed[i - 1]

    def refresh_screen(self):
        self.screen.fill(self.background_color)
        if not self.game_over:
            for items in self.snake:
                self.screen.blit(items[0], items[1])
        else:
            textsurface = self.endgame_text.render('rrrrrr...Game Over points:' + str(len(self.snake) - 1), False,
                                                   (0, 0, 0))
            self.screen.blit(textsurface, (0, 0))
        self.screen.blit(self.food, self.food_cords)
        pygame.display.flip()
        pygame.time.delay(self.refresh_rate)

    def check_food_collision(self):
        if not self.simulation:
            if self.food_cords == (self.snake[0][1].x, self.snake[0][1].y):
                self.move_food()
                self.make_snake_bigger()
        else:
            if self.food_cords == (self.snake[0][0], self.snake[0][1]):
                self.move_food()
                self.make_snake_bigger()

    def horizontal_food_collision(self):
        snake_head = self.snake[0][1]
        x, y = self.food_cords
        if y - self.snake_size < snake_head.top < y + self.food_height:
            if x <= snake_head.right <= x + self.food_height or x <= snake_head.left <= x + self.food_height:
                return True
        return False

    def vertical_food_collision(self):
        snake_head = self.snake[0][1]
        x, y = self.food_cords
        if x - self.snake_size < snake_head.left < x + self.food_height:
            if y <= snake_head.top <= y + self.food_height and y <= snake_head.bottom <= y + self.food_height:
                return True
        return False

    def move_food(self):
        self.food_cords = self.generate_random_cords()

    def make_snake_bigger(self):
        if not self.simulation:
            new_img = pygame.image.load(self.snake_dir)
            new_rect = new_img.get_rect()
            self.set_last_snake_position(new_rect)
            self.snake.append([new_img, new_rect])
            self.speed.append(self.speed[-1])
        else:
            last_x, last_y = self.snake[-1][0], self.snake[-1][1]
            last_speed = self.speed[-1]
            x_s = last_speed[0]
            y_s = last_speed[1]
            x, y = last_x, last_y
            if x_s > 0:
                x -= self.snake_size
            elif x_s < 0:
                x += self.snake_size
            elif y_s > 0:
                y -= self.snake_size
            elif y_s < 0:
                y += self.snake_size
            self.snake.append((x, y))
            self.speed.append(self.speed[-1])

    def set_last_snake_position(self, new_rect):
        last_x, last_y = self.snake[-1][1].x, self.snake[-1][1].y
        last_speed = self.speed[-1]
        x_s = last_speed[0]
        y_s = last_speed[1]
        x, y = last_x, last_y
        if x_s > 0:
            x -= self.snake_size
        elif x_s < 0:
            x += self.snake_size
        elif y_s > 0:
            y -= self.snake_size
        elif y_s < 0:
            y += self.snake_size
        new_rect.x = x
        new_rect.y = y

    def check_snake_collision(self):
        if not self.simulation:
            current_item = self.snake[0][1]
            runner_point = current_item.x, current_item.y
            for body in self.snake[1:]:
                point = body[1].x, body[1].y
                if point == runner_point:
                    self.game_over = True
                    return
        else:
            current_item = self.snake[0]
            runner_point = current_item[0], current_item[1]
            for body in self.snake[1:]:
                point = body[0], body[1]
                if point == runner_point:
                    self.game_over = True
                    return

    def check_boundaries_collision(self):
        if not self.simulation:
            current = self.snake[0][1]
            if current.left < 0 or current.right > self.width or current.top < 0 or current.bottom > self.height:
                self.game_over = True
        else:
            current = self.snake[0]
            if current[0] < 0 or current[0] + self.snake_size > self.width or current[1] < 0 or current[
                1] + self.snake_size > self.height:
                self.game_over = True

    def get_whole_map(self):
        map = [[0] * int(self.height / self.snake_size) for _ in range(int(self.width / self.snake_size))]
        map[int(self.food_cords[1] / self.snake_size)][int(self.food_cords[0] / self.snake_size)] = -1
        sign = 0.5
        for sn in self.snake:
            map[int(sn[1].y / self.snake_size)][int(sn[1].x / self.snake_size)] = sign
            if sign == 0.5:
                sign = 1
        one_array_map = []
        for item in map:
            one_array_map += item
        return one_array_map

    def get_snake_input(self):
        """
        inputs:
        distances:
            4 x wall
            8 x food
            8 x itself
        """
        base = distance((0, 0), (self.width, self.height))
        wall = self.distance_from_wall()
        food = self.distance_from_food()
        itself = self.distance_from_itself()
        inputs = wall + food + itself
        inputs = [x / base for x in inputs]
        return inputs

    def distance_from_wall(self):
        if not self.simulation:
            x, y = self.snake[0][1].x + self.snake_size / 2, self.snake[0][1].y + self.snake_size / 2
        else:
            x, y = self.snake[0][0] + self.snake_size / 2, self.snake[0][1] + self.snake_size / 2
        return [distance((x, y), (x, 0)), distance((x, y), (self.width, y)), distance((x, y), (x, self.height)),
                distance((x, y), (0, y))]

    def distance_from_food(self):
        if not self.simulation:
            x, y = self.snake[0][1].x + self.snake_size / 2, self.snake[0][1].y + self.snake_size / 2
        else:
            x, y = self.snake[0][0] + self.snake_size / 2, self.snake[0][1] + self.snake_size / 2
        food_x, food_y = self.food_cords
        sub_x = food_x - x
        sub_y = food_y - y
        outputs = [0] * 8
        index = 0
        if sub_y < 0:
            if sub_x < 0:
                index = 0
            elif sub_x == 0:
                index = 1
            else:
                index = 2
        elif sub_y == 0:
            if sub_x < 0:
                index = 7
            elif sub_x > 0:
                index = 3
        else:
            if sub_x < 0:
                index = 6
            elif sub_x == 0:
                index = 5
            else:
                index = 4
        outputs[index] = distance((x, y), self.food_cords)
        return outputs

    def distance_from_itself(self):
        outputs = [0] * 8
        if not self.simulation:
            x, y = self.snake[0][1].x + self.snake_size / 2, self.snake[0][1].y + self.snake_size / 2
        else:
            x, y = self.snake[0][0] + self.snake_size / 2, self.snake[0][1] + self.snake_size / 2
        for part in self.snake[1:]:
            if not self.simulation:
                body_x, body_y = part[1].x, part[1].y
            else:
                body_x, body_y = part[0], part[1]
            sub_x = body_x - x
            sub_y = body_y - y
            index = 0
            if sub_y < 0:
                if sub_x < 0:
                    index = 0
                elif sub_x == 0:
                    index = 1
                else:
                    index = 2
            elif sub_y == 0:
                if sub_x < 0:
                    index = 7
                elif sub_x > 0:
                    index = 3
            else:
                if sub_x < 0:
                    index = 6
                elif sub_x == 0:
                    index = 5
                else:
                    index = 4
            dist = distance((x, y), self.food_cords)
            if outputs[index] > dist:
                outputs[index] = dist
        return outputs
