import os
import sys
import json
import math
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))


def load_triplets(
    path: str = "data/triplets_clean.json"
) -> list:
    """Load clean triplets for training."""
    with open(path, encoding='utf-8') as f:
        triplets = json.load(f)
    print(f"Loaded {len(triplets)} triplets")
    return triplets


def prepare_training_data(triplets: list) -> tuple:
    """
    Convert triplets to sentence-transformers format.

    sentence-transformers expects:
    - InputExample objects with texts=[anchor, pos, neg]
    - DataLoader that feeds batches to training

    Returns train_dataset and eval_dataset
    """
    from datasets import Dataset

    # Split 80% train, 20% eval
    split = int(len(triplets) * 0.8)
    train_data = triplets[:split]
    eval_data = triplets[split:]

    print(f"Train size: {len(train_data)}")
    print(f"Eval size:  {len(eval_data)}")

    # Convert to HuggingFace Dataset format
    def to_dict_list(data):
        return {
            "anchor": [t["anchor"] for t in data],
            "positive": [t["positive"] for t in data],
            "negative": [t["negative"] for t in data],
        }

    train_dataset = Dataset.from_dict(
        to_dict_list(train_data))
    eval_dataset = Dataset.from_dict(
        to_dict_list(eval_data))

    return train_dataset, eval_dataset


def setup_model_and_loss(
    base_model: str = "all-MiniLM-L6-v2"
):
    """
    Load the base model and configure the loss function.

    Loss function: MultipleNegativesRankingLoss
    - Industry standard for embedding fine-tuning
    - For each anchor, treats other batch items as negatives
    - Forces model to rank positive higher than all negatives
    - Very efficient — no explicit negatives needed in loss

    Args:
        base_model: HuggingFace model name to fine-tune

    Returns:
        model, loss function
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses import ( # type: ignore
        MultipleNegativesRankingLoss
    )

    print(f"Loading base model: {base_model}")
    model = SentenceTransformer(base_model)

    print(f"Model dimensions: "
          f"{model.get_sentence_embedding_dimension()}")
    print(f"Max sequence length: "
          f"{model.max_seq_length}")

    # MultipleNegativesRankingLoss
    # Works by: in each batch, for anchor i,
    # positive i is correct, all others are negatives
    # More samples per batch = more negatives = better
    loss = MultipleNegativesRankingLoss(model=model)
    print("Loss function: MultipleNegativesRankingLoss")

    return model, loss


def get_training_args(
    output_dir: str = "models/finetuned",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """
    Configure training arguments.

    Key parameters explained:
    - num_epochs: How many times to go through all data
      (3 is standard for small datasets)
    - batch_size: Samples per gradient update
      (16 is safe for CPU, 32+ for GPU)
    - learning_rate: How fast to update weights
      (2e-5 is standard for fine-tuning)
    - warmup_ratio: % of steps for learning rate warmup
      (prevents unstable training at start)
    """
    from sentence_transformers import (
        SentenceTransformerTrainingArguments
    )

    os.makedirs(output_dir, exist_ok=True)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        fp16=False,          # disable for CPU
        bf16=False,          # disable for CPU
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        save_total_limit=2,  # keep only 2 checkpoints
    )

    print(f"\nTraining configuration:")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Output dir:    {output_dir}")

    return args


def run_finetuning(
    triplets_path: str = "data/triplets_clean.json",
    output_dir: str = "models/finetuned",
    num_epochs: int = 1,
    batch_size: int = 16
):
    """
    Run the complete fine-tuning pipeline.

    Steps:
    1. Load triplets
    2. Prepare datasets
    3. Load model + loss
    4. Configure training
    5. Train
    6. Save model

    Args:
        triplets_path: Path to clean triplets
        output_dir: Where to save fine-tuned model
        num_epochs: Training epochs (start with 1)
        batch_size: Training batch size
    """
    from sentence_transformers import (
        SentenceTransformerTrainer
    )
    import mlflow

    print("=" * 60)
    print("FINE-TUNING PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading triplets...")
    triplets = load_triplets(triplets_path)

    # Step 2: Prepare datasets
    print("\n[2/5] Preparing training data...")
    train_dataset, eval_dataset = \
        prepare_training_data(triplets)

    # Step 3: Setup model
    print("\n[3/5] Setting up model and loss...")
    model, loss = setup_model_and_loss()

    # Step 4: Configure training
    print("\n[4/5] Configuring training...")
    args = get_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    # Step 5: Train
    print("\n[5/5] Starting training...")
    print("This will take 10-30 minutes on CPU.")
    print("Watch the loss decrease with each step.\n")

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss
    )

    start_time = time.time()

    # Log to MLflow
    mlflow.set_experiment("fine-tuning")
    with mlflow.start_run(
        run_name=f"epoch-{num_epochs}-bs{batch_size}"
    ):
        mlflow.log_param("base_model",
                         "all-MiniLM-L6-v2")
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("train_size",
                         len(train_dataset))
        mlflow.log_param("eval_size",
                         len(eval_dataset))
        mlflow.log_param("triplets_total",
                         len(triplets))

        # Train
        train_result = trainer.train()

        elapsed = time.time() - start_time

        # Log metrics
        if hasattr(train_result, "metrics"):
            for key, val in \
                    train_result.metrics.items():
                try:
                    mlflow.log_metric(key, float(val))
                except Exception:
                    pass

        mlflow.log_metric("training_time_minutes",
                          elapsed / 60)

    elapsed = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Time taken: {elapsed/60:.1f} minutes")

    # Save final model
    final_path = f"{output_dir}/final"
    model.save(final_path)
    print(f"Model saved: {final_path}")

    return model, final_path


if __name__ == "__main__":
    print("Starting fine-tuning with 1 epoch first...")
    print("(Test run — full 3 epochs in Day 12)")
    print()

    model, path = run_finetuning(
        num_epochs=1,
        batch_size=16
    )

    print(f"\nFine-tuned model saved at: {path}")
    print("Ready for benchmarking tomorrow!")