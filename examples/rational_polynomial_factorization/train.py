from omegaconf import OmegaConf
from calt.io.pipeline import IOPipeline
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline


cfg = OmegaConf.load("configs/train.yaml")

result = IOPipeline.from_config(cfg.data).build()
model = ModelPipeline(cfg.model, result["tokenizer"]).build()
trainer = TrainerPipeline(
    cfg.train,
    model=model,
    tokenizer=result["tokenizer"],
    train_dataset=result["train_dataset"],
    eval_dataset=result["test_dataset"],
    data_collator=result["data_collator"],
    wandb_config=cfg.get("wandb", None),
).build()

trainer.train()
success_rate = trainer.evaluate_and_save_generation()
print(f"Success rate: {100 * success_rate:.1f}%")
