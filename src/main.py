from test import test

from config import Config
from train import train

if __name__ == "__main__":
    config = Config()
    train(config)
    # test(config)
