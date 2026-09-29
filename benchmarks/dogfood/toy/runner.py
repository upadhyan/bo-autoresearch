import boautoresearch as bo
from train import train_and_eval

bo.run(lambda: train_and_eval(epochs=bo.fidelity().get("epochs", 8), seed=bo.seed()))
