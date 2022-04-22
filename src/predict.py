from config import Config
import torch

from model import QAModel


def predict(config: Config):
    model_path = 'model.pth'
    model = QAModel(config)
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
