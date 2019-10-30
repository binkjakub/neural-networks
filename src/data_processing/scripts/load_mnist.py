from settings import DATA_PATH
from src.data_processing.io import load_mnist

data = load_mnist(DATA_PATH)
