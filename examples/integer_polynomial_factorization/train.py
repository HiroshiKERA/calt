from omegaconf import OmegaConf

from calt.io.pipeline import IOPipeline
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline

cfg = OmegaConf.load("configs/train.yaml")

io_dict = IOPipeline.from_config(cfg.data).build()
model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
trainer = TrainerPipeline.from_io_dict(cfg.train, model, io_dict, cfg.wandb).build()

trainer.train()
success_rate = trainer.evaluate_and_save_generation()
print(f"Success rate: {100 * success_rate:.1f}%")

