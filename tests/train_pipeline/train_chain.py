import os
import sys

sys.path.append("src")

# Environment variables for reproducibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import click
import torch
from omegaconf import OmegaConf
from transformers import BartConfig
from transformers import BartForConditionalGeneration as Transformer
from transformers import TrainingArguments

import wandb
from calt import Trainer, count_cuda_devices, load_data
from calt.data_loader.utils.preprocessor import (CoefficientPostfixProcessor,
                                                 PolynomialToInternalProcessor,
                                                 ProcessorChain)


@click.command()
@click.option("--config", type=str, default="config/train.yaml")
@click.option("--dryrun", is_flag=True)
@click.option("--no_wandb", is_flag=True)
def main(config, dryrun, no_wandb):
    # Load config
    cfg = OmegaConf.load(config)

    # Override config if dryrun or no_wandb is set in command line
    cfg.train.dryrun = dryrun if dryrun else cfg.train.dryrun
    cfg.wandb.no_wandb = no_wandb if no_wandb else cfg.wandb.no_wandb

    if cfg.train.dryrun:
        cfg.train.num_train_epochs = 1
        cfg.data.num_train_samples = 1000
        cfg.wandb.group = "dryrun"
        cfg.train.output_dir = "results/dryrun"

        print("-" * 100)
        print("Dryrun mode is enabled. The training setup is modified as follows:")
        print("-" * 100)
        print(f"output_dir: {cfg.train.output_dir}")
        print(f"num_train_epochs: {cfg.train.num_train_epochs}")
        print(f"num_train_samples: {cfg.data.num_train_samples}")
        if not cfg.wandb.no_wandb:
            print(f"wandb.project: {cfg.wandb.project}")
            print(f"wandb.group: {cfg.wandb.group}")
            print(f"wandb.name: {cfg.wandb.name}")

        print("-" * 100)

    # Create output directory if it doesn't exist
    os.makedirs(cfg.train.output_dir, exist_ok=True)

    # Save config
    config_file_name = os.path.basename(config)
    with open(os.path.join(cfg.train.output_dir, config_file_name), "w") as f:
        OmegaConf.save(cfg, f)

    report_target = "wandb"
    if cfg.wandb.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        report_target = "none"
    else:
        wandb.init(
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            name=cfg.wandb.name,
        )
    poly = PolynomialToInternalProcessor(num_variables=2, max_degree=3, max_coeff=200)
    postfix = CoefficientPostfixProcessor()
    chain = ProcessorChain([poly, postfix])

    # Load dataset
    dataset, tokenizer, data_collator = load_data(
        train_dataset_path=cfg.data.train_dataset_path,
        test_dataset_path=cfg.data.test_dataset_path,
        field=cfg.data.field,
        num_variables=cfg.data.num_variables,
        max_degree=cfg.data.max_degree,
        max_coeff=cfg.data.max_coeff,
        max_length=cfg.model.max_sequence_length,
        processor=chain,
        num_train_samples=cfg.data.num_train_samples,
        num_test_samples=cfg.data.num_test_samples,
    )

    # Load model
    model_cfg = BartConfig(
        encoder_layers=cfg.model.num_encoder_layers,
        encoder_attention_heads=cfg.model.num_encoder_heads,
        decoder_layers=cfg.model.num_decoder_layers,
        decoder_attention_heads=cfg.model.num_decoder_heads,
        vocab_size=len(tokenizer.vocab),
        d_model=cfg.model.d_model,
        encoder_ffn_dim=cfg.model.encoder_ffn_dim,
        decoder_ffn_dim=cfg.model.decoder_ffn_dim,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        unk_token_id=tokenizer.unk_token_id,
        max_position_embeddings=cfg.model.max_sequence_length,
        decoder_start_token_id=tokenizer.bos_token_id,
    )
    model = Transformer(config=model_cfg)

    if torch.cuda.is_available():
        train_batch_size = cfg.train.batch_size // count_cuda_devices()
        test_batch_size = cfg.train.test_batch_size // count_cuda_devices()
    else:
        train_batch_size = cfg.train.batch_size
        test_batch_size = cfg.train.test_batch_size
    # Set up trainer
    args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.num_train_epochs,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=test_batch_size,
        lr_scheduler_type="constant"
        if cfg.train.lr_scheduler_type == "constant"
        else "linear",
        max_grad_norm=cfg.train.max_grad_norm,
        optim=cfg.train.optimizer,  # Set optimizer type
        # Dataloader settings
        dataloader_num_workers=cfg.train.num_workers,
        dataloader_pin_memory=True,
        # Evaluation and saving settings
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        label_names=["labels"],
        save_safetensors=False,
        # Logging settings
        logging_strategy="steps",
        logging_steps=50,
        report_to=report_target,
        # Others
        remove_unused_columns=False,
        seed=cfg.train.seed,
        disable_tqdm=True,
    )
    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    # Execute training and evaluation
    train_results = trainer.train()
    trainer.save_model()

    # Calculate evaluation metrics
    metrics = train_results.metrics
    eval_metrics = trainer.evaluate()
    metrics.update(eval_metrics)
    success_rate = trainer.evaluate_and_save_generation()
    metrics["test_success_rate"] = success_rate

    trainer.save_metrics("all", metrics)
    if not cfg.wandb.no_wandb:
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
