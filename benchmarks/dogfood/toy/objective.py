"""The validation loss (protected: research never edits it). An external scorer computes it from the
whole cfg dict: a setting added to cfg in train_and_eval reaches the scorer as it is."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location("scorer", os.environ["DOGFOOD_SCORER"])
_scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scorer)


def evaluate(cfg, epochs, seed):
    return _scorer.evaluate(cfg, epochs, seed)
