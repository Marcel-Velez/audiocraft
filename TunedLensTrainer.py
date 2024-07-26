# -*- coding: utf-8 -*-
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from audiocraft.models import MusicGen
import os
import locale
import numpy as np
from scipy.special import softmax
import math
import random
from functools import partial
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Literal, get_args
import pickle as pkl

from TunedLensUtils import generate_music, generate_corpus
from TunedLensData import MusicGenDataModule
from TunedLens import MusicGenModel




# Training Script
if __name__ == "__main__":
  data_dir = "path_to_data"
  layer_num = 12  # Example layer number
  batch_size = 32
  num_workers = 4
  DEBUG = False

  data_module = MusicGenDataModule(data_dir, layer_num, batch_size, num_workers, DEBUG)

  model = MusicGenModel()

  trainer = Trainer(max_epochs=10, gpus=1 if torch.cuda.is_available() else 0)
  trainer.fit(model, data_module)
  trainer.test(model, data_module)
