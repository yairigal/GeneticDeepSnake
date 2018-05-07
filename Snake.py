import random
import sys, pygame


class Snake:
    def __init__(self, height, width, snake_dir="creeper.png", food_dir="food.gif"):
        self.height = height
        self.width = width
        self.snake_dir = snake_dir
        self.food_dir = food_dir
        snake = pygame.image.load(snake_dir)
        snake_rect = snake.get_rect()
        self.snake_height = snake_rect.height
        snake_rect.x = width/2 - self.snake_height
        snake_rect.y = height/2 - self.snake_height
        self.snake = [[snake, snake_rect]]
        self.food = pygame.image.load(food_dir)
        self.food_height = self.food.get_rect().height
        self.speed = [[0, 0]]
        self.background_color = 30, 63, 159
        self.moving_speed = self.snake_height
        self.game_over = False
        self.last_direction = None

        pygame.init()
        pygame.font.init()
        self.endgame_text = pygame.font.SysFont('Comic Sans MS', 30)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.food_cords = self.generate_random_cords()

    def generate_random_cords(self):
        return random.randint(0, self.width - self.snake_height), random.randint(0, self.height - self.snake_height)

    def start(self):
        while True:
            self.player_events()
            self.move_snake()
            self.check_food_collision()
            self.refresh_screen()
            self.check_snake_collision()
            self.check_boundaries_collision()

    def player_events(self):
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

    def move_snake(self):
        last_cords = self.move_head()
        self.move_body(last_cords)
        self.update_speed()

    def move_body(self, last_cords):
        for i in range(1, len(self.snake)):
            temp = self.snake[i][1].x, self.snake[i][1].y
            self.snake[i][1].x = last_cords[0]
            self.snake[i][1].y = last_cords[1]
            last_cords = temp

    def move_head(self):
        last_cords = self.snake[0][1].x, self.snake[0][1].y
        self.snake[0][1] = self.snake[0][1].move(self.speed[0])
        return last_cords

    def update_speed(self):
        for i in reversed(range(1, len(self.snake))):
            self.speed[i] = self.speed[i - 1]

    def refresh_screen(self, refresh_rate=100):
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
        pygame.time.delay(refresh_rate)

    def check_food_collision(self):
        if self.horizontal_food_collision() or self.vertical_food_collision():
            self.move_food()
            self.make_snake_bigger()

    def horizontal_food_collision(self):
        snake_head = self.snake[0][1]
        x, y = self.food_cords
        if y - self.snake_height <= snake_head.top <= y + self.food_height:
            if x <= snake_head.right <= x + self.food_height or x <= snake_head.left <= x + self.food_height:
                return True
        return False

    def vertical_food_collision(self):
        snake_head = self.snake[0][1]
        x, y = self.food_cords
        if x - self.snake_height <= snake_head.left <= x + self.food_height:
            if y <= snake_head.top <= y + self.food_height and y <= snake_head.bottom <= y + self.food_height:
                return True
        return False

    def move_food(self):
        self.food_cords = self.generate_random_cords()

    def make_snake_bigger(self):
        new_img = pygame.image.load(self.snake_dir)
        new_rect = new_img.get_rect()
        self.set_last_snake_position(new_rect)
        self.snake.append([new_img, new_rect])
        self.speed.append(self.speed[-1])

    def set_last_snake_position(self, new_rect):
        last_x, last_y = self.snake[-1][1].x, self.snake[-1][1].y
        last_speed = self.speed[-1]
        x_s = last_speed[0]
        y_s = last_speed[1]
        x, y = last_x, last_y
        if x_s > 0:
            x -= self.snake_height
        elif x_s < 0:
            x += self.snake_height
        elif y_s > 0:
            y -= self.snake_height
        elif y_s < 0:
            y += self.snake_height
        new_rect.x = x
        new_rect.y = y

    def check_snake_collision(self):
        current_item = self.snake[0][1]
        runner_point = current_item.x, current_item.y
        for body in self.snake[1:]:
            point = body[1].x, body[1].y
            if point == runner_point:
                self.game_over = True
                return

    def check_boundaries_collision(self):
        current = self.snake[0][1]
        if current.left < 0 or current.right > self.width or current.top < 0 or current.bottom > self.height:
            self.game_over = True


if __name__ == '__main__':
    game = Snake(height=500, width=500)
    game.start()
