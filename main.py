import testing
import training
import time


def mnist_cnn():

    start = time.time()

    training.train()
    testing.test()

    finish = time.time()

    print('Seconds passed : %f' % (finish - start))


if __name__ == '__main__':
    mnist_cnn()
