import os
import random
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from clearml import Task
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config_loader import get_config
from dataset.dataset import FlickrDataset, collate_fn
from model.model import CaptionGenerator

warnings.filterwarnings("ignore")
CLEARML_AVAILABLE = True


try:
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
except Exception as e:
    print(f"NLTK download failed: {e}, some features may not work")


def setup_reproducibility(config):
    seed = config.get("reproducibility.seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if config.get("reproducibility.deterministic", True):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_clearml(config):
    clearml_logger = None
    task = None

    if CLEARML_AVAILABLE and config.get("hardware.experiment_tracking.enabled", True):
        project_name = config.get(
            "hardware.experiment_tracking.project_name", "image_caption_generator"
        )
        experiment_name = config.get(
            "hardware.experiment_tracking.experiment_name", "training_experiment"
        )

        if Task is not None:
            task = Task.init(project_name=project_name, task_name=experiment_name)
            clearml_logger = task.get_logger()

            if config.get("hardware.experiment_tracking.log_hyperparameters", True):
                task.connect(config.get_config())

    return clearml_logger, task


def setup_data(config):
    dataset = FlickrDataset()

    train_size = config.get("dataset.train_size", 0.8)
    val_size = config.get("dataset.val_size", 0.1)

    total_size = len(dataset)
    train_len = int(train_size * total_size)
    val_len = int(val_size * total_size)
    test_len = total_size - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_len, val_len, test_len]
    )

    batch_size = config.get("training.batch_size", 32)
    num_workers = config.get("hardware.num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=config.get("hardware.pin_memory", True),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("evaluation.batch_size", 64),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=config.get("hardware.pin_memory", True),
    )

    return train_loader, val_loader, dataset


def setup_model(config, vocab_size, device):
    embedding_dim = config.get("model.decoder.embedding_dim", 256)
    hidden_dim = config.get("model.decoder.hidden_dim", 512)
    num_layers = config.get("model.decoder.num_layers", 2)

    model = CaptionGenerator(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    return model


def setup_training_components(config, model, word2idx):
    pad_idx = word2idx.get("<PAD>", 0)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    lr = config.get("training.learning_rate", 1e-4)
    weight_decay = config.get("training.weight_decay", 1e-4)

    if config.get("training.optimizer", "adam").lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )

    scheduler_type = config.get("training.scheduler", "step")
    scheduler = None

    if scheduler_type == "step":
        step_size = config.get("training.step_size", 25)
        gamma = config.get("training.gamma", 0.5)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get("training.num_epochs", 50)
        )

    return criterion, optimizer, scheduler


def calculate_bleu_score(predictions, targets, idx2word, pad_idx, start_idx, end_idx):
    references = []
    candidates = []

    for pred, target in zip(predictions, targets):
        pred_words = [
            idx2word.get(idx, "<UNK>")
            for idx in pred
            if idx not in [pad_idx, start_idx]
        ]
        target_words = [
            idx2word.get(idx, "<UNK>")
            for idx in target
            if idx not in [pad_idx, start_idx]
        ]

        if end_idx in pred:
            end_idx_pos = list(pred).index(end_idx)
            pred_words = pred_words[:end_idx_pos]
        if end_idx in target:
            end_idx_pos = list(target).index(end_idx)
            target_words = target_words[:end_idx_pos]

        candidates.append(pred_words)
        references.append([target_words])

    smoothing_function = SmoothingFunction().method4
    bleu_score = corpus_bleu(
        references, candidates, smoothing_function=smoothing_function
    )
    return bleu_score


def calculate_meteor_score(predictions, targets, idx2word, pad_idx, start_idx, end_idx):
    meteor_scores = []

    for pred, target in zip(predictions, targets):
        pred_words = [
            idx2word.get(idx, "<UNK>")
            for idx in pred
            if idx not in [pad_idx, start_idx]
        ]
        target_words = [
            idx2word.get(idx, "<UNK>")
            for idx in target
            if idx not in [pad_idx, start_idx]
        ]

        if end_idx in pred:
            end_idx_pos = list(pred).index(end_idx)
            pred_words = pred_words[:end_idx_pos]
        if end_idx in target:
            end_idx_pos = list(target).index(end_idx)
            target_words = target_words[:end_idx_pos]

        if pred_words and target_words:
            try:
                score = meteor_score([target_words], pred_words)
                meteor_scores.append(score)
            except Exception:
                continue

    return np.mean(meteor_scores) if meteor_scores else 0.0


def print_sample_predictions(predictions, targets, idx2word, word2idx, num_samples=3):
    """Print sample predictions for debugging"""
    pad_idx = word2idx.get("<PAD>", 0)
    start_idx = word2idx.get("<START>", 1)
    end_idx = word2idx.get("<END>", 2)

    print("\n--- Sample Predictions ---")
    for i in range(min(num_samples, len(predictions))):
        pred_words = []
        true_words = []

        for idx in predictions[i]:
            if idx == pad_idx or idx == end_idx:
                break
            if idx != start_idx:
                pred_words.append(idx2word.get(idx, "<UNK>"))

        for idx in targets[i]:
            if idx == pad_idx or idx == end_idx:
                break
            if idx != start_idx:
                true_words.append(idx2word.get(idx, "<UNK>"))

        print(f"Pred {i + 1}: {' '.join(pred_words)}")
        print(f"True {i + 1}: {' '.join(true_words)}")
        print()


def train_epoch(
    model, train_loader, criterion, optimizer, config, device, clearml_logger, epoch
):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        images = batch["images"].to(device)
        captions = batch["captions"].to(device)
        caption_lengths = batch["caption_lengths"].to(device)

        optimizer.zero_grad()

        predictions, encoded_captions, decode_lengths, alphas, sort_ind = model(
            images, captions, caption_lengths
        )

        # Calculate loss
        targets = encoded_captions[:, 1:]

        # Pack padded sequences for efficient computation
        predictions = torch.cat(
            [predictions[i, : decode_lengths[i]] for i in range(len(decode_lengths))],
            dim=0,
        )
        targets = torch.cat(
            [targets[i, : decode_lengths[i]] for i in range(len(decode_lengths))], dim=0
        )

        loss = criterion(predictions, targets)
        loss.backward()

        if config.get("training.gradient_clipping", 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.get("training.gradient_clipping")
            )

        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

        if (
            clearml_logger
            and batch_idx % config.get("hardware.log_every_n_steps", 100) == 0
        ):
            clearml_logger.report_scalar(
                "Training",
                "Batch Loss",
                loss.item(),
                batch_idx + len(train_loader) * (epoch - 1),
            )

    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device, idx2word, word2idx):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch["images"].to(device)
            captions = batch["captions"].to(device)
            caption_lengths = batch["caption_lengths"].to(device)

            predictions, encoded_captions, decode_lengths, alphas, sort_ind = model(
                images, captions, caption_lengths
            )

            targets = encoded_captions[:, 1:]
            pred_packed = torch.cat(
                [
                    predictions[i, : decode_lengths[i]]
                    for i in range(len(decode_lengths))
                ],
                dim=0,
            )
            targets_packed = torch.cat(
                [targets[i, : decode_lengths[i]] for i in range(len(decode_lengths))],
                dim=0,
            )

            loss = criterion(pred_packed, targets_packed)
            total_loss += loss.item()

            pred_words = torch.argmax(predictions, dim=-1)
            all_predictions.extend(pred_words.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(val_loader)

    pad_idx = word2idx.get("<PAD>", 0)
    start_idx = word2idx.get("<START>", 1)
    end_idx = word2idx.get("<END>", 2)

    if len(all_predictions) > 0:
        print_sample_predictions(all_predictions, all_targets, idx2word, word2idx)

    bleu_score = calculate_bleu_score(
        all_predictions, all_targets, idx2word, pad_idx, start_idx, end_idx
    )
    meteor_score_val = calculate_meteor_score(
        all_predictions, all_targets, idx2word, pad_idx, start_idx, end_idx
    )

    return avg_loss, bleu_score, meteor_score_val


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_val_loss,
    vocab_size,
    word2idx,
    idx2word,
    config,
    is_best=False,
):
    checkpoint_dir = config.get("training.checkpoint_dir", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "vocab_size": vocab_size,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "config": config.get_config(),
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(checkpoint, best_path)


def plot_training_curves(
    train_losses, val_losses, bleu_scores, meteor_scores, clearml_logger
):
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    axes[0, 0].plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    axes[0, 0].plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    axes[0, 0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # BLEU score
    axes[0, 1].plot(epochs, bleu_scores, "g-", label="BLEU Score", linewidth=2)
    axes[0, 1].set_title("BLEU Score Over Time", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("BLEU Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # METEOR score
    axes[1, 0].plot(epochs, meteor_scores, "m-", label="METEOR Score", linewidth=2)
    axes[1, 0].set_title("METEOR Score Over Time", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("METEOR Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Metrics comparison
    axes[1, 1].plot(epochs, bleu_scores, "g-", label="BLEU", linewidth=2)
    axes[1, 1].plot(epochs, meteor_scores, "m-", label="METEOR", linewidth=2)
    axes[1, 1].set_title("Metrics Comparison", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = "training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    if clearml_logger:
        clearml_logger.report_matplotlib_figure(
            "Training Curves", "All Metrics", figure=fig, iteration=len(train_losses)
        )

    plt.show()
    return plot_path


def plot_metrics_distribution(bleu_scores, meteor_scores, clearml_logger):
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # BLEU distribution
    axes[0].hist(bleu_scores, bins=20, alpha=0.7, color="green", edgecolor="black")
    axes[0].set_title("BLEU Scores Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("BLEU Score")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, alpha=0.3)

    # METEOR distribution
    axes[1].hist(meteor_scores, bins=20, alpha=0.7, color="purple", edgecolor="black")
    axes[1].set_title("METEOR Scores Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("METEOR Score")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = "metrics_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    if clearml_logger:
        clearml_logger.report_matplotlib_figure(
            "Metrics Distribution", "Histograms", figure=fig, iteration=len(bleu_scores)
        )

    plt.show()
    return plot_path


def main():
    # Setup
    config = get_config()
    device = torch.device(
        config.get("hardware.device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    setup_reproducibility(config)
    clearml_logger, task = setup_clearml(config)

    train_loader, val_loader, dataset = setup_data(config)
    vocab_size = dataset.get_vocab_size()
    word2idx = dataset.get_word2idx()
    idx2word = dataset.get_idx2word()

    model = setup_model(config, vocab_size, device)
    criterion, optimizer, scheduler = setup_training_components(config, model, word2idx)

    if clearml_logger and config.get(
        "hardware.experiment_tracking.log_model_architecture", True
    ):
        clearml_logger.report_text("Model Architecture", str(model))

    num_epochs = config.get("training.num_epochs", 50)
    save_every_n_epochs = config.get("training.save_every_n_epochs", 5)
    early_stopping_patience = config.get("training.early_stopping.patience", 10)

    best_val_loss = float("inf")
    patience_counter = 0

    train_losses = []
    val_losses = []
    bleu_scores = []
    meteor_scores = []

    print(f"Starting training for {num_epochs} epochs on {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Vocabulary size: {vocab_size}")

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 50)

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            config,
            device,
            clearml_logger,
            epoch,
        )
        val_loss, bleu_score, meteor_score_val = validate_epoch(
            model, val_loader, criterion, device, idx2word, word2idx
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)
        meteor_scores.append(meteor_score_val)

        if scheduler:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"BLEU Score: {bleu_score:.4f}")
        print(f"METEOR Score: {meteor_score_val:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        if clearml_logger:
            clearml_logger.report_scalar("Loss", "Train", train_loss, epoch)
            clearml_logger.report_scalar("Loss", "Validation", val_loss, epoch)
            clearml_logger.report_scalar("Metrics", "BLEU", bleu_score, epoch)
            clearml_logger.report_scalar("Metrics", "METEOR", meteor_score_val, epoch)
            clearml_logger.report_scalar("Learning Rate", "LR", current_lr, epoch)

        # Early stopping and checkpointing
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"New best validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        if config.get("training.save_checkpoints", True):
            if epoch % save_every_n_epochs == 0 or is_best:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_val_loss,
                    vocab_size,
                    word2idx,
                    idx2word,
                    config,
                    is_best,
                )

        if config.get("training.early_stopping.enabled", True):
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final BLEU score: {bleu_scores[-1]:.4f}")
    print(f"Final METEOR score: {meteor_scores[-1]:.4f}")

    plot_training_curves(
        train_losses, val_losses, bleu_scores, meteor_scores, clearml_logger
    )
    plot_metrics_distribution(bleu_scores, meteor_scores, clearml_logger)

    summary_stats = {
        "best_val_loss": best_val_loss,
        "final_bleu": bleu_scores[-1],
        "final_meteor": meteor_scores[-1],
        "max_bleu": max(bleu_scores),
        "max_meteor": max(meteor_scores),
        "total_epochs": len(train_losses),
    }

    if clearml_logger:
        for key, value in summary_stats.items():
            clearml_logger.report_single_value(key, value)

    print("\nTraining Summary:")
    for key, value in summary_stats.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
