import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from audiocraft.models import MusicGen
import os

from pytorch_lightning.loggers import WandbLogger

from collections import defaultdict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


from TunedLensUtils import generate_music, generate_corpus, yaml_config_hook
from TunedLensData import MusicGenDataModule

def kl_loss(logit_pred, logit_target):
  """
  Calculates the KL divergence loss between two logit tensors.

  Args:

    logit_pred: Predicted logits.
    logit_target: Target logits.

  Returns:
    The KL divergence loss.
  """
  loss = torch.sum(
      logit_target.log_softmax(-1).exp() * (logit_target.log_softmax(-1) - logit_pred.log_softmax(-1)), dim=-1
  ).mean()

  # pred = F.softmax(logit_pred, dim=-1)
  # target = F.softmax(logit_target, dim=-1)
  return loss # F.kl_div(pred.log(), target, reduction='batchmean')


def generate_music(model, prompts, duration=4, output_file="parler_tts_out.wav", save_file=False, stop_layer_idx=None,
                   linear_layer=None):
  generation = model(prompts, duration, stop_layer_idx, linear_layer)
  if isinstance(prompts, list) and len(prompts) > 1:
    for ind, aud in enumerate(generation):
      if save_file:
        sf.write(f'{ind}_{output_file}', aud.cpu().numpy().squeeze(), 32000)
  else:
    if save_file:
      sf.write(f'{output_file}', generation.cpu().numpy().squeeze(), 32000)
  return generation


import os
import pickle as pkl
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


class MusicGenMLPs(pl.LightningModule):
  def __init__(self, args, musicgen_model, hidden_dim=1024):
    super().__init__()
    self.save_hyperparameters(args)
    self.mlps = nn.ModuleList()
    for _ in range(len(musicgen_model.lm.transformer.layers) - 1):
      self.mlps.append(nn.Sequential(
        nn.Linear(musicgen_model.lm.transformer.layers[0].cross_attention.out_proj.out_features,
                  musicgen_model.lm.linears[0].in_features),
      ))

    self.criterion = self.configure_criterion()

  def forward(self, x, layer_idx):
    return self.mlps[layer_idx](x)

  def training_step(self, batch, batch_idx):
    batch_activations, target, audio = batch
    loss_dict = defaultdict(float)
    for i in range(len(self.mlps)):
      if i == 23:
        pass
      out = self(batch_activations[i], i)
      loss = self.criterion(out, target)
      loss_dict[f"Train/loss_{i}"] = loss

    self.log_dict(loss_dict, loss)
    return loss_dict

  def validation_step(self, batch, batch_idx):
    batch_activations, target, audio = batch
    loss_dict = defaultdict(float)
    for i in range(len(self.mlps)):
      if i == 23:
        pass
      out = self(batch_activations[i], i)
      loss = self.criterion(out, target)
      loss_dict[f"Val/loss_{i}"] = loss

    self.log_dict(loss_dict, loss)
    return loss_dict


  def configure_optimizers(self):
    # Define optimizer and scheduler
    if self.hparams.optim == 'adam':
      optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=0.05,
                                    betas=(.9, .95), )
    else:
      optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.25, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.hparams.start_factor, end_factor=self.hparams.end_factor, total_iters=self.hparams.total_iters)
    return {'optimizer': optimizer, 'lr_scheduler': scheduler}

  def configure_criterion(self) -> nn.Module:
      # PT lightning aggregates differently in DP mode
      if self.hparams.criterion == "kl":
        criterion = kl_loss
      else:
        criterion = nn.MSELoss()

      return criterion


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="TunedMusicLens")
  parser = Trainer.add_argparse_args(parser)

  config = yaml_config_hook("./config.yaml")
  for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))

  args = parser.parse_args()
  pl.seed_everything(args.seed)

  data_module = MusicGenDataModule(args.data_dir, args.layer_num, args.batch_size, args.num_workers, args.DEBUG)
  musicgen_model = MusicGen.get_pretrained("small", device=torch.device('cuda' if (args.gpus != 0 and torch.cuda.is_available()) else 'cpu'))

  model = MusicGenMLPs(args, musicgen_model)

  # trainer = Trainer(max_epochs=10, gpus=1 if torch.cuda.is_available() else 0)
  # trainer.fit(model, data_module)
  # trainer.test(model, data_module)

  # early_stopping = EarlyStopping(monitor="Val/loss", patience=20)
  logger = WandbLogger(project="tuned-music-lens", log_model=True)

  # effective_steps = args.max_epochs * len(data_module.train_dataset) // args.batch_size
  tot_steps = 250 * gradient_accumulation_steps # currently 250 * 12 * 5
  trainer = Trainer.from_argparse_args(
    args,
    logger=logger,
    max_steps=tot_steps,
    log_every_n_steps=10,
    check_val_every_n_epoch=1,
    accelerator=args.accelerator,
    gradient_clip_val=args.gradient_clip_val,
    accumulate_grad_batches=args.accumulate_grad_batches,
  )

  data_module.setup()
  trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

# Example usage:
# data_module = MusicGenDataModule(data_dir='path/to/data', layer_num=12, batch_size=32, num_workers=4, DEBUG=True)
# trainer = pl.Trainer()
# trainer.fit(model, data_module)
