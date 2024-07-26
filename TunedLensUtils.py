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
from collections import defaultdict



PROMPT_TEMPLATE = "Compose a [MOOD] [GENRE] piece, with a [INSTRUMENTS] melody. Use a [TEMPO] tempo."
MOODS = ['emotional', 'passionate', 'groovy', 'energetic', 'happy']
GENRES = ['rock', 'pop', 'electronic', 'jazz', 'classical']
INSTRUMENTS = ['guitar', 'piano', 'trumpet', 'violin', 'flute']
TEMPI = ['fast', 'medium', 'slow', 'slow to medium', 'medium to fast']
type_str_to_list = {'[GENRE]': GENRES, '[MOOD]': MOODS, '[INSTRUMENTS]': INSTRUMENTS, '[TEMPO]': TEMPI}


def generate_music(model, prompts, duration=4, output_file="parler_tts_out.wav", save_file=False, stop_layer_idx=None,
                   linear_layer=None):
  generation = model.generate(prompts, duration, stop_layer_idx=stop_layer_idx, linear_layer=linear_layer)
  if isinstance(prompts, list) and len(prompts) > 1:
    for ind, aud in enumerate(generation):
      if save_file:
        sf.write(f'{ind}_{output_file}', aud.cpu().numpy().squeeze(), 32000)
  else:
    if save_file:
      sf.write(f'{output_file}', generation.cpu().numpy().squeeze(), 32000)
  return generation


def generate_corpus(const_subj, prompt_type, n_total=99999, n_per_type=99999):
  type_list = ['[GENRE]', '[MOOD]', '[INSTRUMENTS]', '[TEMPO]']
  assert prompt_type in type_list, f"type {prompt_type} not in {type_list}"
  type_list.remove(prompt_type)

  corpus = []
  type_a, type_b, type_c = type_list[0], type_list[1], type_list[2]
  for subj_a in type_str_to_list[type_a][:n_per_type]:
    for subj_b in type_str_to_list[type_b][:n_per_type]:
      for subj_c in type_str_to_list[type_c][:n_per_type]:
        new_prompt = PROMPT_TEMPLATE.replace(type_a, subj_a).replace(type_b, subj_b).replace(type_c, subj_c).replace(
          prompt_type, const_subj)
        corpus.append(new_prompt)
  return corpus[:n_total]


def clear_music_model_hooks(music_model):
  for i, layer in enumerate(music_model.lm.transformer.layers):
    layer._forward_hooks.clear()


def get_batch_prompt_activations(music_model, descs: list, output_file: str) -> dict:
  assert isinstance(descs, list), f"input text should be a list of prompts, got '{type(descs) =}' with '{len(descs) =}'"
  # Create a dictionary to store the activations of a single sample.
  activations = defaultdict(list)

  def self_attention_hook_fn(layer_id):
    def hook(module, input, output):
      # Store the output tensor and the module name.
      activations[layer_id].append(output.detach())

    return hook

  # Register the hooks.
  for i, layer in enumerate(music_model.lm.transformer.layers):
    layer.register_forward_hook(self_attention_hook_fn(i))

  # Go through your dataset and compute the forward passes.
  res = generate_music(music_model, descs, output_file=output_file)

  # Unregister the hooks to make sure they don't interfere with the next dataset.
  clear_music_model_hooks(music_model)

  activations_split_per_prompt = [{key: val[i] for key, val in activations.items()} for i in range(len(descs))]
  for i, (audio_out, desc) in enumerate(zip(res, descs)):
    filename = desc.replace(' ', '_')
    print(f'{output_file}_{i}.pkl')
    with open(f'{output_file}_{i}.pkl', 'wb') as handle:
      pkl.dump({'audio': audio_out, 'activations': activations_split_per_prompt[i]}, handle,
               protocol=pkl.HIGHEST_PROTOCOL)

  return

import yaml

#### from https://github.com/Spijkervet/CLMR/blob/master/clmr/utils/yaml_config_hook.py
def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg




if __name__ == '__main__':
  # Load the model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  music_model = MusicGen.get_pretrained("small", device=device)
  print(music_model, '\n\n\n')

  all_prompts = []
  for concept_list, prompt_type in [(GENRES,'[GENRE]'), (MOODS,'[MOOD]'), (INSTRUMENTS,'[INSTRUMENTS]'), (TEMPI,'[TEMPO]')]:
    for concept in concept_list:
      all_prompts.extend(generate_corpus(concept, prompt_type))

  # Batch the prompts to fit GPU memory
  batch_size = 25  # Adjust based on your GPU capacity
  prompt_batches = [all_prompts[i:i + batch_size] for i in range(0, len(all_prompts), batch_size)]

  # Process each batch
  for index, batch in enumerate(prompt_batches):
    if index < 66:
      continue
    print(f'Batch {index+1}/{len(prompt_batches)}')
    get_batch_prompt_activations(music_model, batch, output_file=f'./data/{index}_batch_output')
