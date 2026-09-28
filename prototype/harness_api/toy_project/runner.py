"""PROTOTYPE — the runner script the orchestrator generates. It's this short on purpose:
it names the objective and nothing else. It's a protected path once generated."""
import bo
from train import train_and_eval

if __name__ == "__main__":
    bo.run(train_and_eval)
