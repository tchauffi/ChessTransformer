from pathlib import Path
from datetime import datetime

from tqdm.auto import tqdm
import numpy as np
import torch
import accelerate
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from chesstransformer.datasets import LichessSimpleUciDataset
from chesstransformer.models.transformer.gpt import GPT2Model


ROOT_PATH = Path(__file__).parents[1]


def get_next_run_number(log_dir="logs"):
    """Get the next run number by checking existing run directories."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return 1
    
    # Find all directories that match the pattern "run_XXX"
    existing_runs = [d.name for d in log_path.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    if not existing_runs:
        return 1
    
    # Extract run numbers and find the maximum
    run_numbers = []
    for run_name in existing_runs:
        try:
            num = int(run_name.split("_")[1])
            run_numbers.append(num)
        except (IndexError, ValueError):
            continue
    
    return max(run_numbers) + 1 if run_numbers else 1

def main():
    # Load the full dataset with LichessSimpleUciDataset
    print("Loading dataset...")
    full_dataset = LichessSimpleUciDataset(
        dataset_path= ROOT_PATH.parents[1] / "data/Lichess Rated Games 2017.pgn.zst",
        vocab_path= ROOT_PATH / "data/tokenizer_models/uci_vocab.json",
        num_samples=500_000,  # Load all available games
    )

    vocab_size = len(full_dataset.vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Split the dataset into training, validation, and test sets
    val = int(0.1 * len(full_dataset))
    test = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val - test
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val, test],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    def collate_fn(batch):
        max_length = max(item["input_ids"].shape[0] for item in batch)
        input_ids = torch.full((len(batch), max_length), fill_value=-100, dtype=torch.long)
        target_ids = torch.full((len(batch), max_length), fill_value=-100, dtype=torch.long)
        legal_moves_mask = torch.zeros((len(batch), max_length, vocab_size), dtype=torch.float32)

        for i, item in enumerate(batch):
            length = item["input_ids"].shape[0]
            input_ids[i, :length] = item["input_ids"]
            target_ids[i, :length] = item["target_ids"]
            legal_moves_mask[i, :length, :] = item["legal_moves_mask"]

        if max_length > config["context_size"]:
            input_ids = input_ids[:, :config["context_size"]]
            target_ids = target_ids[:, :config["context_size"]]
            legal_moves_mask = legal_moves_mask[:, :config["context_size"], :]

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "legal_moves_mask": legal_moves_mask,
        }

    train_dl = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_dl = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)


    config = {
        "vocab_size": vocab_size,
        "context_size": 256,
        "embed_dim": 512,
        "num_heads": 8,
        "num_layers": 8,
        "dropout": 0.1,
        "qkv_bias": False,
    }

    model = GPT2Model(config)
    print(model)

    # Generate unique run identifier
    run_number = get_next_run_number("logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_number:03d}_{timestamp}"
    
    print(f"Starting training run: {run_name}")

    # Initialize accelerator with logging
    accelerator = accelerate.Accelerator(
        log_with="tensorboard", 
        project_dir=f"logs/{run_name}",
        mixed_precision='fp16' if torch.cuda.is_available() else 'no'
    )
    device = accelerator.device
    print(f"Using device: {device}")
    model.to(device)
    
    # Initialize TensorBoard tracker
    accelerator.init_trackers(
        project_name="chess_transformer",
        config={
            **config,
            "run_name": run_name,
            "run_number": run_number,
            "timestamp": timestamp,
            "learning_rate": 1e-3,
            "weight_decay": 1e-1,
            "num_epochs": 10,
            "batch_size": 16,
            "warmup_steps": 1000,
        },
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)
    
    # Calculate total training steps
    num_epochs = 10
    total_steps = len(train_dl) * num_epochs
    warmup_steps = 1000
    
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    # Create linear scheduler with warmup
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, num_warmup_steps))
            # Linear decay
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, scheduler
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_llm_loss = 0.0
        total_l2_loss = 0.0
        loader = tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)
            legal_moves_mask = batch["legal_moves_mask"].to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            llm_loss = criterion(logits.flatten(0, 1), targets.flatten())

            probas = torch.nn.functional.softmax(logits, dim=-1)

            ilegal_moves_probas = probas * (1 - legal_moves_mask)
            l2_loss = torch.sum(ilegal_moves_probas ** 2, dim=-1).mean()
            loss = llm_loss + l2_loss

            loader.set_postfix({"llm_loss": llm_loss.item(), "l2_loss": l2_loss.item(), "total_loss": loss.item()})

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()  # Update learning rate

            total_loss += loss.item()
            total_llm_loss += llm_loss.item()
            total_l2_loss += l2_loss.item()
            
            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0]
            
            # Log step metrics
            accelerator.log({
                "train/step_loss": loss.item(),
                "train/step_llm_loss": llm_loss.item(),
                "train/step_l2_loss": l2_loss.item(),
                "train/learning_rate": current_lr,
            }, step=global_step)
            
            global_step += 1

        avg_train_loss = total_loss / len(train_dl)
        avg_train_llm_loss = total_llm_loss / len(train_dl)
        avg_train_l2_loss = total_l2_loss / len(train_dl)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_val_llm_loss = 0.0
        total_val_l2_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                inputs = batch["input_ids"].to(device)
                targets = batch["target_ids"].to(device)
                legal_moves_mask = batch["legal_moves_mask"].to(device)

                logits = model(inputs)

                llm_loss = criterion(logits.flatten(0, 1), targets.flatten())
                
                probas = torch.nn.functional.softmax(logits, dim=-1)
                ilegal_moves_probas = probas * (1 - legal_moves_mask)
                l2_loss = torch.sum(ilegal_moves_probas ** 2, dim=-1).mean()
                loss = llm_loss + l2_loss
                
                total_val_loss += loss.item()
                total_val_llm_loss += llm_loss.item()
                total_val_l2_loss += l2_loss.item()

        avg_val_loss = total_val_loss / len(val_dl)
        avg_val_llm_loss = total_val_llm_loss / len(val_dl)
        avg_val_l2_loss = total_val_l2_loss / len(val_dl)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
        
        # Log epoch metrics
        accelerator.log({
            "train/epoch_loss": avg_train_loss,
            "train/epoch_llm_loss": avg_train_llm_loss,
            "train/epoch_l2_loss": avg_train_l2_loss,
            "val/epoch_loss": avg_val_loss,
            "val/epoch_llm_loss": avg_val_llm_loss,
            "val/epoch_l2_loss": avg_val_l2_loss,
            "epoch": epoch + 1,
        }, step=global_step)

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), ROOT_PATH / "models/best_model.pth")
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
            
            # Log best metrics
            accelerator.log({
                "best/val_loss": best_val_loss,
                "best/epoch": epoch + 1,
            }, step=global_step)

    # Testing
    model.eval()
    total_test_loss = 0.0
    total_test_llm_loss = 0.0
    total_test_l2_loss = 0.0
    with torch.no_grad():
        for batch in test_dl:
            inputs = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)
            legal_moves_mask = batch["legal_moves_mask"].to(device)

            logits = model(inputs)

            llm_loss = criterion(logits.flatten(0, 1), targets.flatten())
            
            probas = torch.nn.functional.softmax(logits, dim=-1)
            ilegal_moves_probas = probas * (1 - legal_moves_mask)
            l2_loss = torch.sum(ilegal_moves_probas ** 2, dim=-1).mean()
            loss = llm_loss + l2_loss
            
            total_test_loss += loss.item()
            total_test_llm_loss += llm_loss.item()
            total_test_l2_loss += l2_loss.item()

    avg_test_loss = total_test_loss / len(test_dl)
    avg_test_llm_loss = total_test_llm_loss / len(test_dl)
    avg_test_l2_loss = total_test_l2_loss / len(test_dl)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Log test metrics
    accelerator.log({
        "test/loss": avg_test_loss,
        "test/llm_loss": avg_test_llm_loss,
        "test/l2_loss": avg_test_l2_loss,
    }, step=global_step)
    
    # End tracking
    accelerator.end_training()
    print(f"\nTraining complete! Run: {run_name}")
    print(f"TensorBoard logs saved to 'logs/{run_name}/'")
    print("To view logs, run: tensorboard --logdir=logs")
    print(f"To view this run only: tensorboard --logdir=logs/{run_name}")


if __name__ == "__main__":
    main()