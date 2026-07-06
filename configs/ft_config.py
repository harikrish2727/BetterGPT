from pathlib import Path
import torch
from trl import SFTConfig
from src.utils.paths import FINETUNE_OUT_DIR

output_dir = FINETUNE_OUT_DIR / "alpaca_model"


sft_config = SFTConfig(
    output_dir=Path(output_dir),
    assistant_only_loss=True,
    max_length=512,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=0.03,
    num_train_epochs=3,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    gradient_checkpointing=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    dataloader_num_workers=4,
    dataset_kwargs={"num_proc": 4},
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
)
