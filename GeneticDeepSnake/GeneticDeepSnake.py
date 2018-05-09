import sys
from SnakeClass import Snake

from SnakeClass import distance

sys.path.append('../')
from NN.NonMatrixArtificialNeuralNetwork import NonMatrixArtificialNeuralNetwork as NeuralNet
import random


def normalize_output(y_hat):
    return y_hat.index(max(y_hat))


class Genetic_Snake:
    def __init__(self, population_size=99, mutation_rate=0.1, board_height=25, board_width=25, refresh_rate=100,
                 max_steps=625, same_food=False):
        self.pop = []
        self.layers = [20, 10, 10, 4]
        self.N = population_size
        self.height = board_height
        self.width = board_width
        self.mut_rate = mutation_rate
        self.refresh_rate = refresh_rate
        self.fitness_avg = 0
        self.max_steps = max_steps
        self.snake = Snake(self.height, self.width, refresh_rate=self.refresh_rate, max_steps=self.max_steps,
                           simulation=True)
        self.same_food = same_food

        self.died_from_hunger = 0
        self.died_from_hit = 0
        self.max_points = 0
        self.max_distance = distance((0, 0), (self.width * self.snake.snake_size, self.height * self.snake.snake_size))

    def reset_stats(self):
        self.fitness_avg = 0
        self.died_from_hit = 0
        self.died_from_hunger = 0
        self.max_points = 0

    def fitness(self, nn):
        self.snake.reset()
        if self.same_food:
            self.snake.food_values = self.food_vals.copy()
        moving_func = lambda x: nn.predict(x, normalization_function=normalize_output)
        self.snake.moving_function = moving_func
        points, failed, steps, distance_from_food = self.snake.start()
        return self.calc_points(failed, points, steps, distance_from_food)

    def calc_points(self, failed, points, steps, distance_from_food):
        if points - 1 > self.max_points:
            self.max_points = points - 1
        ftns = 0.0000001
        if failed:  # [0 - max,10]          [0 - 50]
            ftns += 10 * (points - 1) ** 2 + (steps * 50 / self.max_steps)

            self.died_from_hit += 1

        else:  # died from hunger
            #          [0-max,20]
            ftns += 20 * (points - 1) ** 2

            self.died_from_hunger += 1

        # ftns += (self.max_distance - distance_from_food) * 10 / self.max_distance
        self.fitness_avg += ftns
        return ftns

    def generate_population(self):
        for i in range(self.N):
            self.pop.append(NeuralNet(self.layers))

    def kill_bad_ones(self):
        self.pop = sorted(self.pop, key=lambda x: self.fitness(self.pop.index(x)))
        self.pop = self.pop[int(self.N * (1 / 3)):]

    def start(self):
        self.generate_population()
        i = 0
        overall_max_pts = 0
        while True:
            i += 1
            if self.same_food:
                self.food_vals = self.generate_food_values()
            self.reset_stats()
            # big_pop, max_ftns = self.calc_fitness()
            # self.crossover(big_pop)
            max_nns, max_ftnses = self.calc_fitness_not_random()
            self.crossover_not_random(max_nns, max_ftnses)
            overall_max_pts = self.print_msg(i, max_ftnses[0], overall_max_pts)

    def print_msg(self, i, max_ftns, overall_max_pts):
        avg = round(self.fitness_avg / self.N, 3)
        max_ftns = round(max_ftns, 3)
        hunger = round(self.died_from_hunger * 100 / self.N, 3)
        hit = round(self.died_from_hit * 100 / self.N, 3)
        max_pts = self.max_points
        if max_pts > overall_max_pts:
            overall_max_pts = max_pts
        msg = "Generation {}:\n" \
              "\t- Population: {} | Mutation rate: {}%\n" \
              "\t- Average Fitness: {} | Max Fitness: {}\n" \
              "\t- Died from hunger: {}%\n" \
              "\t- Died from hit: {}%\n" \
              "\t- Top Score: {} | Top Score Overall: {}" \
            .format(
            i, self.N, self.mut_rate * 100, avg, max_ftns, hunger, hit, max_pts, overall_max_pts)
        print(msg)
        return overall_max_pts

    def crossover(self, big_pop):
        self.pop = []
        for i in range(self.N):
            nn1, ftns1 = random.choice(big_pop)
            nn2, ftns2 = random.choice(big_pop)
            self.pop.append(self.merge_nn_mut_fitness(nn1, ftns1, nn2, ftns2))

    def crossover_not_random(self, max_nns, max_ftns):
        self.pop = []
        zipped = list(zip(max_nns, max_ftns))
        for i in range(int(self.N)):
            f, f_ftns = random.choice(zipped)
            s, s_ftns = random.choice(zipped)
            self.pop.append(self.merge_nn_mut_fitness(f, f_ftns, s, s_ftns))

    def is_mutation(self):
        if self.mut_rate >= random.random():
            return True
        return False

    def merge_nn(self, f, s):
        # f_w = f.W
        # s_w = s.W
        # c = NonMatrixArtificialNeuralNetwork(self.layers)
        # for i in range(len(s.layers) - 1):
        #     for j in range(len(f_w[i])):
        #         for k in range(len(f_w[i][j])):
        #             mut_rate = self.mut_rate * 100
        #             if self.is_mutation(mut_rate):
        #                 c.W[i][j][k] = random.uniform(-1, 1)
        #             else:
        #                 if random.randint(0, 1) == 0:
        #                     c.W[i][j][k] = f_w[i][j][k]
        #                 else:
        #                     c.W[i][j][k] = s_w[i][j][k]
        # return c
        c = NeuralNet(self.layers)
        if random.randint(0, 1) == 0:
            chosen_nn = s
        else:
            chosen_nn = f

        for i in range(len(chosen_nn.layers) - 1):
            for j in range(len(chosen_nn.W[i])):
                for k in range(len(chosen_nn.W[i][j])):
                    if self.is_mutation():
                        # c.W[i][j][k] = (s.W[i][j][k] + f.W[i][j][k])/2 + random.uniform(-1, 1)
                        c.W[i][j][k] = chosen_nn.W[i][j][k] + random.uniform(-150, 150)
                    else:
                        # c.W[i][j][k] = (s.W[i][j][k] + f.W[i][j][k])/2
                        c.W[i][j][k] = chosen_nn.W[i][j][k]
        # for i in range(1, len(chosen_nn.layers)):
        #     for j in range(len(chosen_nn.B[i])):
        #         if self.is_mutation():
        #             c.B[i][j] = chosen_nn.B[i][j] + random.uniform(-10, 10)
        #         else:
        #             c.B[i][j] = chosen_nn.B[i][j]
        return c

    def merge_nn_mut_fitness(self, f, f_fitness, s, s_fitness):
        # f_w = f.W
        # s_w = s.W
        # c = NonMatrixArtificialNeuralNetwork(self.layers)
        # for i in range(len(s.layers) - 1):
        #     for j in range(len(f_w[i])):
        #         for k in range(len(f_w[i][j])):
        #             mut_rate = self.mut_rate * 100
        #             if self.is_mutation(mut_rate):
        #                 c.W[i][j][k] = random.uniform(-1, 1)
        #             else:
        #                 if random.randint(0, 1) == 0:
        #                     c.W[i][j][k] = f_w[i][j][k]
        #                 else:
        #                     c.W[i][j][k] = s_w[i][j][k]
        # return c
        c = NeuralNet(self.layers)
        if random.randint(0, 1) == 0:
            chosen_nn = s
            ftns = s_fitness
        else:
            chosen_nn = f
            ftns = f_fitness

        for i in range(len(chosen_nn.layers) - 1):
            for j in range(len(chosen_nn.W[i])):
                for k in range(len(chosen_nn.W[i][j])):
                    if self.is_mutation():
                        # mut_max = 1000 / ftns
                        # c.W[i][j][k] = (s.W[i][j][k] + f.W[i][j][k])/2 + random.uniform(-1, 1)
                        c.W[i][j][k] = chosen_nn.W[i][j][k] + random.uniform(-10, 10)
                    else:
                        # c.W[i][j][k] = (s.W[i][j][k] + f.W[i][j][k])/2
                        c.W[i][j][k] = chosen_nn.W[i][j][k]
        # for i in range(1, len(chosen_nn.layers)):
        #     for j in range(len(chosen_nn.B[i])):
        #         if self.is_mutation():
        #             c.B[i][j] = chosen_nn.B[i][j] + random.uniform(-10, 10)
        #         else:
        #             c.B[i][j] = chosen_nn.B[i][j]
        return c

    def create_layer(self, layer, biases=False):
        if not biases:
            new_layer = []
            for i in range(len(layer)):
                row = []
                for j in range(len(layer[i])):
                    row.append(random.uniform(-1, 1))
                new_layer.append(row)
            return new_layer
        else:
            new_layer = []
            for item in layer:
                new_layer.append(item + random.uniform(-1, 1))
            return new_layer

    def calc_fitness(self):
        # calculating fitnesses
        fitnesses = [0] * self.N
        for i, nn in enumerate(self.pop):
            fitnesses[i] = self.fitness(nn)
        big_pop = []
        max_fitness = max(fitnesses)
        # saving best nn
        self.pop[fitnesses.index(max_fitness)].save(dir="./best_snake")
        # creating big population based on the probability of each nn
        probs = [int(round(f / max_fitness, 2) * 100) for f in fitnesses]
        for times, nn, ftns in zip(probs, self.pop, fitnesses):
            big_pop += [(nn, ftns)] * times
        return big_pop, max_fitness

    def calc_fitness_not_random(self):
        # calculating fitnesses
        fitnesses = [0] * self.N
        for i, nn in enumerate(self.pop):
            fitnesses[i] = self.fitness(nn)
        pop = zip(self.pop, fitnesses)
        max_nns, max_fitnesses = zip(*sorted(pop, key=lambda x: x[1], reverse=True)[:int(self.N * 0.01)])
        # saving best nn
        max_nns[0].save(dir="./best_snake")

        return max_nns, max_fitnesses

    def generate_food_values(self):
        vals = []
        for _ in range(int(self.snake.height * self.snake.width / self.snake.snake_size ** 2)):
            x, y = self.snake.generate_random_cords()
            vals.append((x, y))
        return vals


# improvements:
# avg nn
# -make same game each generation
# -changed fitness
# -add statistics : died from hunger, died from hitting,max points, m
# -changed inputs to be the minimum distance from body.
# -cross over picks the best each generation
# -fitness is bound to distance from food

if __name__ == '__main__':
    choice = input("=== Menu ===\n\t1.run genetic alg\n\t2.run best snake\n\t3.play snake\n\t")
    choice = int(choice)
    if choice == 1:
        genetic = Genetic_Snake(refresh_rate=0, mutation_rate=0.05, population_size=999, same_food=True)
        genetic.start()
    elif choice == 2:
        nn = NeuralNet.load("./best_snake")
        mov_func = lambda x: nn.predict(x, normalization_function=normalize_output)
        snake = Snake(25, 25, moving_func=mov_func, refresh_rate=30)
        snake.start()
    elif choice == 3:
        s = Snake(25, 25)
        s.start()
