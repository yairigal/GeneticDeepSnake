import random

from pickle import dump, load

import time

from NN.math_functions import sigmoid, dsigmoid


class NonMatrixArtificialNeuralNetwork:
    def __init__(self, layers, activation=sigmoid, dactivation=dsigmoid):
        self.layers = layers
        self.activation = activation
        self.dactivation = dactivation
        self.last_cost = 0
        self.last_acc = 0
        self.W = []
        self.Z = []
        self.A = []
        self.B = [[0 for _ in range(layers[0])]]
        for i in range(1, len(layers)):
            w = [[random.uniform(-1, 1) for _ in range(layers[i])] for _ in
                 range(layers[i - 1])]
            b = [0 for _ in range(layers[i])]
            self.W.append(w)
            self.B.append(b)

    def forward_prop(self, x):
        output = x
        self.A = [x]
        self.Z = [x]
        for k in range(len(self.W)):
            new_output = []
            current_z = []
            current_a = []
            for j in range(len(self.W[k][0])):
                summer = 0
                for i in range(len(output)):
                    summer += output[i] * self.W[k][i][j]
                summer += self.B[k + 1][j]
                current_z.append(summer)
                summer = self.activation(summer)
                current_a.append(summer)
                new_output.append(summer)
            self.Z.append(current_z)
            self.A.append(current_a)
            output = new_output
        return output

    def back_prop(self, x, y):
        y_hat = self.forward_prop(x)
        cost_derivative = self.dcost(y, y_hat)
        deltas = [None] * len(self.layers)
        deltas[-1] = [cost_derivative[i] * self.dactivation(self.Z[-1][i]) for i in range(len(cost_derivative))]
        changes = [None] * len(self.W)
        for k in reversed(range(len(self.layers) - 1)):
            changes[k] = self.calc_changes_for_weights(deltas, k)
            deltas[k] = self.calc_deltas_for_current_layer(k, deltas)

        return changes, deltas, y_hat

    def train(self, data, epochs=500, batch_size=100, lr=0.7, test=False, test_data=[], normal_func=[], log=False,
              save_dir=None):
        for epoch in range(epochs):
            if log: print("epoch {}/{}".format(epoch + 1, epochs))
            self.iterate_over_dataset(batch_size, data, log, lr)
            if test: print("acc=", self.test(test_data, normal_func))
            self.save(log=log, dir=save_dir)

    def iterate_over_dataset(self, batch_size, data, log, lr):
        random.shuffle(data)
        batches = self.split_to_batches(batch_size, data)
        run_batch_func = self.run_batch
        for batch in batches:
            run_batch_func(batch, batch_size, lr)
            if log: print("cost:", self.last_cost)

    def split_to_batches(self, batch_size, data):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def run_batch(self, batch, batch_size, lr):
        sum_w, sum_b = self.init_empty_arrays()
        sum_weights_func = self.sum_weights_changes
        sum_biases_func = self.sum_biases_changes
        back_prop = self.back_prop
        for x, y in batch:
            w_chgs, b_chgs, y_hat = back_prop(x, y)
            sum_w = sum_weights_func(sum_w, w_chgs)
            sum_b = sum_biases_func(sum_b, b_chgs)
        self.update_weights_and_biases(sum_w, sum_b, lr, batch_size)
        self.last_cost = self.calc_cost(y, y_hat)

    def calc_cost(self, y, y_hat):
        return sum(map(lambda i: 0.5 * ((y[i] - y_hat[i]) ** 2), range(len(y))))
        # return sum([0.5 * ((y[i] - y_hat[i]) ** 2) for i in range(len(y))])

    def test(self, data, normal):
        print("Started Testing")
        random.shuffle(data)
        avg = 0.0
        size = len(data)
        for x, y in data:
            y_hat = self.predict(x, normalization_function=normal)
            if y == y_hat:
                avg += 1
        self.last_acc = avg * 100 / size
        return self.last_acc

    def save(self, dir=None, log=False):
        if log: print("Saving...")
        if not dir: dir = "/data/data"
        with open(dir, "wb+") as file:
            dump(self, file)

    @staticmethod
    def load(dir=None):
        if not dir:
            dir = "data/data"
        with open(dir, "rb") as file:
            return load(file)

    def sum_weights_changes(self, w1, w2):
        assert len(w1) == len(w2)
        for i in range(len(w1)):
            assert len(w1[i]) == len(w2[i])
            for j in range(len(w1[i])):
                assert len(w1[i][j]) == len(w2[i][j])
                for k in range(len(w1[i][j])):
                    w1[i][j][k] += w2[i][j][k]
        return w1

    def sum_biases_changes(self, b1, b2):
        assert len(b1) == len(b2)
        for i in range(len(b1)):
            assert len(b1[i]) == len(b2[i])
            for j in range(len(b1[i])):
                b1[i][j] += b2[i][j]
        return b1

    def init_empty_arrays(self):
        empty = []
        for i in range(len(self.W)):
            row = []
            for j in range(len(self.W[i])):
                col = [0] * len(self.W[i][j])
                row.append(col)
            empty.append(row)

        empty_b = []
        for i in range(len(self.B)):
            row = [0] * len(self.B[i])
            empty_b.append(row)
        return empty, empty_b

    def update_weights_and_biases(self, w_change, b_change, lr, batch_size):
        # update
        for k in range(len(self.W)):
            current_w = self.W[k]
            for i in range(len(current_w)):
                for j in range(len(current_w[i])):
                    current_w[i][j] -= lr * (w_change[k][i][j] / batch_size)
            self.W[k] = current_w

        for k in range(len(self.layers)):
            current_biases = self.B[k]
            for i in range(self.layers[k]):
                current_biases[i] -= lr * (b_change[k][i] / batch_size)
            self.B[k] = current_biases

    def dcost(self, y, y_hat):
        assert len(y) == len(y_hat)
        return list(map(lambda i: y_hat[i] - y[i], range(len(y))))

    def calc_deltas_for_current_layer(self, k, deltas):
        deltas_output = []
        for i in range(self.layers[k]):
            holder = 0
            for j in range(len(deltas[k + 1])):
                d = deltas[k + 1][j]
                w = self.W[k][i][j]
                holder += d * w
            dsig = self.dactivation(self.Z[k][i])
            delta = holder * dsig
            deltas_output.append(delta)
        return deltas_output

    def calc_changes_for_weights(self, deltas, k):
        changes = []
        for i in range(len(self.W[k])):
            new_row = []
            for j in range(len(self.W[k][i])):
                d = deltas[k + 1][j]
                a = self.A[k][i]
                result = d * a
                new_row.append(result)
            changes.append(new_row)
        return changes

    def predict(self, x, normalization_function=None):
        y = self.forward_prop(x)
        if normalization_function:
            return normalization_function(y)
        return y


if __name__ == '__main__':
    nn = NonMatrixArtificialNeuralNetwork([2, 2, 1])
    x = time.time()
    out = nn.calc_cost([1] * 100000000, [2] * 100000000)
    print("time", time.time() - x, "output", out)
    # print("output", nn.forward_prop([1, 1]))
