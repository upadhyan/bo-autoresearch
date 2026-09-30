"""The validation loss (protected: research never edits it). The evaluation service at $EVAL_SOCKET
computes it from the whole cfg dict, the epochs and the seed."""
import json
import os
import socket


def evaluate(cfg, epochs, seed):
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(os.environ["EVAL_SOCKET"])
        s.sendall(json.dumps({"cfg": cfg, "epochs": epochs, "seed": seed}).encode() + b"\n")
        reply = json.loads(s.makefile().readline())
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply["loss"]
