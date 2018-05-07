from math import exp


def sigmoid(x):
    try:
        if x > 500:
            return 1
        if x < -500:
            return 0
        return 1.0 / (1.0 + exp(-x))
    except OverflowError as e:
        print("math error x=", x)
        raise e


def dsigmoid(x):
    sig = sigmoid(x)
    return sig * (1.0 - sig)


def ReLU(x):
    return max(0, x)


def dReLU(x):
    if x < 0:
        return 0
    else:
        return 1
