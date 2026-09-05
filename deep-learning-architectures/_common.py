# -*- coding: utf-8 -*-
"""Shared helpers for building deep-learning-architectures notebooks."""
import json, os

MODULE = os.path.dirname(os.path.abspath(__file__))
KERNEL = "deep-learning-architectures"

def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}

def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}

IMPORTS = code("""import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Microsoft YaHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

print("PyTorch version:", torch.__version__)
torch.manual_seed(42)
""")

def write_nb(lesson_dir, nb_name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": f"Python ({KERNEL})", "language": "python", "name": KERNEL},
            "language_info": {"name": "python", "version": "3.12.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = os.path.join(MODULE, lesson_dir, nb_name)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(nb, fp, ensure_ascii=False, indent=1)
    print("written:", out, "cells:", len(cells))

def write_readme(lesson_dir, content):
    out = os.path.join(MODULE, lesson_dir, "README.md")
    with open(out, "w", encoding="utf-8") as fp:
        fp.write(content)
    print("written:", out)
