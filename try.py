def model(red, eta, acu, tol):
    def regularization(fx, n):
        print("%s iteration, proportion of the model is: %s" % (n, fx))

        if fx >= tol:
            return

        ret_x = (1 - fx) * acu
        reg_x = eta * ret_x
        fx = red * fx + reg_x
        regularization(fx, n + 1)

    return regularization


def exponentialWeight(beta):
    def calcualteTrend(n):
        if n == 0:
            return 0
        else:
            return beta * calcualteTrend(n - 1) + (1 - beta)

    return calcualteTrend


if __name__ == "__main__":
    my_model = exponentialWeight(0.9)
    print(my_model(900))