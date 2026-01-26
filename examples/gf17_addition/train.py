import os

from omegaconf import OmegaConf

from calt.io import IOPipeline
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline

cfg = OmegaConf.load("configs/train.yaml")
save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

io_dict = IOPipeline.from_config(cfg.data).build()
model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

trainer_pipeline.train()
trainer_pipeline.save_model()
success_rate = trainer_pipeline.evaluate_and_save_generation()
print(f"Success rate: {100 * success_rate:.1f}%")
