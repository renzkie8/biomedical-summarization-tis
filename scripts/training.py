from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.data import DatasetRecord
    from scripts.metrics import aggregate_metric_sets, summary_metrics
    from scripts.models import HuggingFaceSummarizer
    from scripts.text import lead_summary, oracle_summary
except ImportError:
    try:
        from .data import DatasetRecord
        from .metrics import aggregate_metric_sets, summary_metrics
        from .models import HuggingFaceSummarizer
        from .text import lead_summary, oracle_summary
    except (ImportError, ValueError):
        from data import DatasetRecord
        from metrics import aggregate_metric_sets, summary_metrics
        from models import HuggingFaceSummarizer
        from text import lead_summary, oracle_summary


@dataclass(frozen=True)
class TrainingArtifacts:
    output_dir: Path
    model_name: str
    max_source_length: int
    max_target_length: int


def _lazy_import_hf() -> tuple[Any, Any, Any, Any]:
    try:
        datasets_module = importlib.import_module("datasets")
        transformers_module = importlib.import_module("transformers")
        Dataset = datasets_module.Dataset
        DataCollatorForSeq2Seq = transformers_module.DataCollatorForSeq2Seq
        Seq2SeqTrainer = transformers_module.Seq2SeqTrainer
        Seq2SeqTrainingArguments = transformers_module.Seq2SeqTrainingArguments
    except Exception as exc:
        raise RuntimeError(
            "Training requires `datasets` and `transformers`. Install the project dependencies first."
        ) from exc
    return Dataset, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


def _lazy_import_peft() -> tuple[Any, Any, Any]:
    """Import peft (parameter-efficient fine-tuning) lazily.

    Returns (get_peft_model, LoraConfig, TaskType).
    Raises RuntimeError with install hint if peft is not available.
    """
    try:
        peft = importlib.import_module("peft")
        return peft.get_peft_model, peft.LoraConfig, peft.TaskType
    except Exception as exc:
        raise RuntimeError(
            "LoRA training requires the `peft` library. "
            "Install it with:  pip install peft"
        ) from exc


def records_to_dataset(records: Sequence[DatasetRecord]):
    Dataset, _, _, _ = _lazy_import_hf()
    return Dataset.from_list([{"article": record.article, "abstract": record.abstract} for record in records])


def prepare_tokenized_dataset(records: Sequence[DatasetRecord], tokenizer, max_source_length: int, max_target_length: int):
    dataset = records_to_dataset(records)

    def _encode(batch):
        model_inputs = tokenizer(
            batch["article"],
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["abstract"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(_encode, batched=True, remove_columns=dataset.column_names)


def train_seq2seq_model(
    model_name: str,
    train_records: Sequence[DatasetRecord],
    validation_records: Sequence[DatasetRecord],
    output_dir: str | Path,
    *,
    max_source_length: int = 1024,
    max_target_length: int = 256,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 3,
    learning_rate: float = 3e-5,
    lora_rank: int = 0,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    gradient_checkpointing: bool = False,
    use_safetensors: bool | None = None,
    resume_from_checkpoint: bool | str | None = None,
):
    _, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments = _lazy_import_hf()
    transformers_module = importlib.import_module("transformers")
    AutoModelForSeq2SeqLM = transformers_module.AutoModelForSeq2SeqLM
    AutoTokenizer = transformers_module.AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_safetensors is None:
        model_path = Path(model_name)
        if model_path.exists() and model_path.is_dir():
            use_safetensors = (model_path / "model.safetensors").exists()
        else:
            use_safetensors = True

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=use_safetensors)

    if lora_rank > 0:
        get_peft_model, LoraConfig, TaskType = _lazy_import_peft()
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    train_dataset = prepare_tokenized_dataset(train_records, tokenizer, max_source_length, max_target_length)
    eval_dataset = prepare_tokenized_dataset(validation_records, tokenizer, max_source_length, max_target_length)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        report_to=[],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=0.01,
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
    )

    init_sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    accepted = set(p.name for p in init_sig.parameters.values() if p.name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))

    if "eval_strategy" in accepted:
        kwargs["eval_strategy"] = "epoch"
        if "evaluation_strategy" in kwargs:
            del kwargs["evaluation_strategy"]
    elif "evaluation_strategy" in accepted:
        kwargs["evaluation_strategy"] = "epoch"
        if "eval_strategy" in kwargs:
            del kwargs["eval_strategy"]

    filtered = {k: v for k, v in kwargs.items() if k in accepted}

    if resume_from_checkpoint is True:
        # Check if checkpoints exist to avoid crash if resuming from non-existent state
        checkpoint_path = Path(output_dir)
        if not checkpoint_path.exists() or not list(checkpoint_path.glob("checkpoint-*")):
            print(f"No valid checkpoints found in {output_dir}. Starting fresh training.")
            resume_from_checkpoint = None

    try:
        args = Seq2SeqTrainingArguments(**filtered)
    except Exception as exc:
        raise RuntimeError(f"Failed to instantiate Training Arguments. Filtered parameters: {filtered}") from exc

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        import numpy as np
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        metric_results = []
        for pred, label in zip(decoded_preds, decoded_labels):
            metric_results.append(summary_metrics(pred, label))
        
        aggregated = aggregate_metric_sets(metric_results)
        return {
            "rouge_l": aggregated["rouge_l"],
            "bertscore": aggregated["bertscore"]
        }

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if lora_rank > 0:
        model = model.merge_and_unload()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer


def run_training_pipeline(
    model_name: str,
    train_records: Sequence[DatasetRecord],
    val_records: Sequence[DatasetRecord],
    output_dir: Path,
    epochs: int = 3,
    lr: float = 3e-5,
    is_led: bool = False,
    resume_from_checkpoint: bool | str | None = None,
    max_source_length: int | None = None
):
    if is_led:
        lora_rank = 4
        lora_alpha = 16
        batch_size = 1
        grad_acc = 16
        checkpointing = True
    else:
        lora_rank = 8
        lora_alpha = 32
        batch_size = 4
        grad_acc = 4
        checkpointing = False
        
    print(f"Starting fine-tuning for {model_name}...")
    print(f"Config: LoRA(r={lora_rank}, a={lora_alpha}), Batch={batch_size}, GradAcc={grad_acc}")
    
    trainer = train_seq2seq_model(
        model_name=model_name,
        train_records=train_records,
        validation_records=val_records,
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        gradient_checkpointing=checkpointing,
        resume_from_checkpoint=resume_from_checkpoint,
        max_source_length=max_source_length if max_source_length else (16384 if is_led else 1024)
    )
    
    print(f"Training complete. Model saved to {output_dir}")
    return trainer


def evaluate_extractive_baselines(records: Sequence[DatasetRecord], *, max_tokens: int = 256) -> dict[str, dict[str, float]]:
    lead_scores = []
    oracle_scores = []
    for record in records:
        lead_prediction = lead_summary(record.article, max_tokens=max_tokens)
        oracle_prediction = oracle_summary(record.article, record.abstract, max_tokens=max_tokens)
        lead_scores.append(summary_metrics(lead_prediction, record.abstract))
        oracle_scores.append(summary_metrics(oracle_prediction, record.abstract))
    return {
        "lead": aggregate_metric_sets(lead_scores),
        "oracle": aggregate_metric_sets(oracle_scores),
    }


def run_inference(model_name: str, articles: Sequence[str], *, max_source_length: int = 1024, max_target_length: int = 256) -> list[str]:
    summarizer = HuggingFaceSummarizer(model_name, max_source_length=max_source_length, max_target_length=max_target_length)
    return summarizer.generate_many(articles)

if __name__ == "__main__":
    print("Training module. Use run_training_pipeline() for orchestration.")
