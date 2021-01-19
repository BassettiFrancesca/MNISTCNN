import testing
import training


def mnist_cnn():

    training.train()
    testing.test()


if __name__ == '__main__':
    mnist_cnn()
