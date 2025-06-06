from omegaconf import OmegaConf
from transformers import BartConfig, TrainingArguments
from transformers import BartForConditionalGeneration as Transformer
from calt import (
    PolynomialTrainer,
    count_cuda_devices,
)
from calt import data_loader
import wandb


def main():
    cfg = OmegaConf.load("config/train_example.yaml")

    dataset, tokenizer, data_collator = data_loader(
        train_dataset_path=cfg.train_dataset_path,
        test_dataset_path=cfg.test_dataset_path,
        field=cfg.field,
        num_variables=cfg.num_variables,
        max_degree=cfg.max_degree,
        max_coeff=cfg.max_coeff,
        max_length=cfg.model.max_sequence_length,
        # processor_name="numeric", # Uncomment if using numeric processor
    )

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

    args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.num_train_epochs,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        per_device_train_batch_size=cfg.train.batch_size // count_cuda_devices(),
        per_device_eval_batch_size=cfg.train.test_batch_size // count_cuda_devices(),
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
        report_to="wandb",
        # Others
        remove_unused_columns=False,
        seed=cfg.train.seed,
        disable_tqdm=True,
    )
    trainer = PolynomialTrainer(
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
    acc = trainer.generate_evaluation(max_length=cfg.model.max_sequence_length)
    metrics["test_accuracy"] = acc

    trainer.save_metrics("all", metrics)
    wandb.log(metrics)
    wandb.finish()


if __name__ == "__main__":
    main()
