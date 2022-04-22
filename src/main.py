from config import Config
from train import train
from predict import predict

if __name__ == "__main__":
    config = Config()
    train(config)
    # predict(config)
