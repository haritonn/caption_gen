import os
import random
import warnings

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from clearml import Task
except ImportError:
    Task = None
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_loader import get_config
from dataset.dataset import (
    FlickrDataset,
    FlickrMetricsDataset,
    collate_fn,
    metrics_collate_fn,
    split_image_names,
)
from model.model import CaptionGenerator

warnings.filterwarnings("ignore")


def ensure_nltk_resources():
    resources = {
        "corpora/wordnet": "wordnet",
    }

    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                print(f"Warning: failed to download NLTK resource '{resource_name}'.")


ensure_nltk_resources()


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
    tracking_enabled = config.get("hardware.experiment_tracking.enabled", False)
    if not tracking_enabled:
        return None, None

    if Task is None:
        print(
            "ClearML is enabled in config, but package is not installed. Skipping tracking."
        )
        return None, None

    task = Task.init(
        project_name=config.get(
            "hardware.experiment_tracking.project_name",
            "image_caption_generator",
        ),
        task_name=config.get(
            "hardware.experiment_tracking.experiment_name", "training_experiment"
        ),
    )

    if config.get("hardware.experiment_tracking.log_hyperparameters", True):
        task.connect(config.get_config())

    return task.get_logger(), task


def setup_data(config):
    base_dataset = FlickrDataset(is_train=False)
    train_images, val_images, test_images = split_image_names(
        base_dataset.image_names,
        train_size=config.get("dataset.train_size", 0.8),
        val_size=config.get("dataset.val_size", 0.1),
        seed=config.get("reproducibility.seed", 42),
    )

    train_dataset = FlickrDataset(is_train=True, allowed_images=train_images)
    val_dataset = FlickrDataset(
        is_train=False,
        allowed_images=val_images,
        word2idx=train_dataset.word2idx,
        word_freq=train_dataset.word_freq,
    )
    metrics_dataset = FlickrMetricsDataset(
        is_train=False,
        allowed_images=val_images,
        word2idx=train_dataset.word2idx,
        word_freq=train_dataset.word_freq,
    )

    assert set(train_images).isdisjoint(val_images)
    assert set(train_images).isdisjoint(test_images)
    assert set(val_images).isdisjoint(test_images)

    num_workers = config.get("hardware.num_workers", 4)
    pin_memory = config.get("hardware.pin_memory", True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("training.batch_size", 32),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("evaluation.batch_size", 64),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    metrics_loader = DataLoader(
        metrics_dataset,
        batch_size=config.get("evaluation.batch_size", 64),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=metrics_collate_fn,
        pin_memory=pin_memory,
    )

    print(
        "Dataset split by images:",
        f"train={len(train_images)} images / {len(train_dataset)} captions,",
        f"val={len(val_images)} images / {len(val_dataset)} captions,",
        f"test={len(test_images)} images",
    )

    return train_loader, val_loader, train_dataset, metrics_loader


def setup_model(config, vocab_size, device):
    model = CaptionGenerator(
        vocab_size=vocab_size,
        embedding_dim=config.get("model.decoder.embedding_dim", 256),
        hidden_dim=config.get("model.decoder.hidden_dim", 512),
        encoder_pretrained=config.get("model.encoder.pretrained", False),
    ).to(device)
    return model


def focal_loss(inputs, targets, alpha=1.0, gamma=2.0, ignore_index=-100):
    ce_loss = nn.functional.cross_entropy(
        inputs,
        targets,
        ignore_index=ignore_index,
        reduction="none",
    )
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()


def setup_training_components(config, model, word2idx):
    pad_idx = word2idx.get("<PAD>", 0)

    if config.get("training.use_focal_loss", False):

        def criterion(inputs, targets):
            return focal_loss(
                inputs,
                targets,
                alpha=0.25,
                gamma=2.0,
                ignore_index=pad_idx,
            )

    else:
        criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=config.get("training.label_smoothing", 0.1),
        )

    lr = config.get("training.learning_rate", 1e-4)
    weight_decay = config.get("training.weight_decay", 1e-4)

    if config.get("training.optimizer", "adam").lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    scheduler_type = config.get("training.scheduler", "step")
    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("training.step_size", 25),
            gamma=config.get("training.gamma", 0.5),
        )
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("training.num_epochs", 50),
        )

    return criterion, optimizer, scheduler


def calculate_bleu_score(
    predictions, reference_groups, idx2word, pad_idx, start_idx, end_idx
):
    references = []
    candidates = []

    for pred, reference_group in zip(predictions, reference_groups):
        candidates.append(tokens_to_words(pred, idx2word, pad_idx, start_idx, end_idx))
        references.append(
            [
                tokens_to_words(reference, idx2word, pad_idx, start_idx, end_idx)
                for reference in reference_group
            ]
        )

    return corpus_bleu(
        references,
        candidates,
        smoothing_function=SmoothingFunction().method4,
    )


def calculate_meteor_score(
    predictions, reference_groups, idx2word, pad_idx, start_idx, end_idx
):
    scores = []

    for pred, reference_group in zip(predictions, reference_groups):
        pred_words = tokens_to_words(pred, idx2word, pad_idx, start_idx, end_idx)
        reference_words = [
            tokens_to_words(reference, idx2word, pad_idx, start_idx, end_idx)
            for reference in reference_group
        ]
        if pred_words and reference_words:
            try:
                scores.append(meteor_score(reference_words, pred_words))
            except Exception:
                pass

    return np.mean(scores) if scores else 0.0


def print_sample_predictions(predictions, targets, idx2word, word2idx, num_samples=3):
    pad_idx = word2idx.get("<PAD>", 0)
    start_idx = word2idx.get("<START>", 1)
    end_idx = word2idx.get("<END>", 2)

    print("\n--- Sample Predictions ---")
    for index in range(min(num_samples, len(predictions))):
        pred_words = []
        true_words = []

        for token_idx in predictions[index]:
            if token_idx in (pad_idx, end_idx):
                break
            if token_idx != start_idx:
                pred_words.append(idx2word.get(token_idx, "<UNK>"))

        for token_idx in targets[index]:
            if token_idx in (pad_idx, end_idx):
                break
            if token_idx != start_idx:
                true_words.append(idx2word.get(token_idx, "<UNK>"))

        print(f"Pred {index + 1}: {' '.join(pred_words)}")
        print(f"True {index + 1}: {' '.join(true_words)}")
        print()


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    config,
    device,
    clearml_logger=None,
    epoch=None,
):
    model.train()
    total_loss = 0.0
    sampling_prob = config.get("training.max_sampling_prob", 0.5)

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        images = batch["images"].to(device)
        captions = batch["captions"].to(device)
        caption_lengths = batch["caption_lengths"].to(device)

        optimizer.zero_grad()
        predictions, encoded_captions, decode_lengths, alphas, _ = model(
            images,
            captions,
            caption_lengths,
            sampling_prob,
        )

        targets = encoded_captions[:, 1:]
        predictions = torch.cat(
            [predictions[i, : decode_lengths[i]] for i in range(len(decode_lengths))],
            dim=0,
        )
        targets = torch.cat(
            [targets[i, : decode_lengths[i]] for i in range(len(decode_lengths))],
            dim=0,
        )

        loss = criterion(predictions, targets)

        attention_regularization = config.get("training.attention_regularization", 0.0)
        if attention_regularization > 0:
            loss = loss + attention_regularization * torch.mean(
                torch.sum(alphas**2, dim=-1)
            )

        loss.backward()

        gradient_clipping = config.get("training.gradient_clipping", 0.0)
        if gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

        if clearml_logger is not None and epoch is not None:
            log_every_n_steps = config.get("hardware.log_every_n_steps", 100)
            if (progress_bar.n - 1) % log_every_n_steps == 0:
                clearml_logger.report_scalar(
                    "Training",
                    "Batch Loss",
                    loss.item(),
                    (epoch - 1) * len(train_loader) + progress_bar.n,
                )

    return total_loss / len(train_loader)


def tokens_to_words(token_ids, idx2word, pad_idx, start_idx, end_idx):
    words = []

    for token_id in token_ids:
        if token_id == end_idx:
            break
        if token_id in (pad_idx, start_idx):
            continue
        word = idx2word.get(token_id, "<UNK>")
        words.append(word)

    return words


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch["images"].to(device)
            captions = batch["captions"].to(device)
            caption_lengths = batch["caption_lengths"].to(device)

            predictions, encoded_captions, decode_lengths, _, _ = model(
                images,
                captions,
                caption_lengths,
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

            total_loss += criterion(pred_packed, targets_packed).item()

    return total_loss / len(val_loader)


def evaluate_generation(model, metrics_loader, device, idx2word, word2idx, max_length):
    model.eval()
    pad_idx = word2idx.get("<PAD>", 0)
    start_idx = word2idx.get("<START>", 1)
    end_idx = word2idx.get("<END>", 2)
    all_predictions = []
    all_reference_groups = []

    with torch.inference_mode():
        for batch in tqdm(metrics_loader, desc="Generation metrics"):
            images = batch["images"].to(device)
            generated_tokens = model.generate(
                images,
                start_idx=start_idx,
                end_idx=end_idx,
                max_length=max_length,
            )

            all_predictions.extend(generated_tokens.cpu().tolist())
            all_reference_groups.extend(batch["references"])

    if all_predictions:
        sample_targets = [references[0] for references in all_reference_groups]
        print_sample_predictions(all_predictions, sample_targets, idx2word, word2idx)

    bleu_score = calculate_bleu_score(
        all_predictions,
        all_reference_groups,
        idx2word,
        pad_idx,
        start_idx,
        end_idx,
    )
    meteor_score_val = calculate_meteor_score(
        all_predictions,
        all_reference_groups,
        idx2word,
        pad_idx,
        start_idx,
        end_idx,
    )

    return bleu_score, meteor_score_val


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
        torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))


def main():
    config = get_config()
    device = torch.device(
        config.get("hardware.device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    setup_reproducibility(config)
    clearml_logger, _ = setup_clearml(config)
    train_loader, val_loader, dataset, metrics_loader = setup_data(config)

    vocab_size = dataset.get_vocab_size()
    word2idx = dataset.get_word2idx()
    idx2word = dataset.get_idx2word()

    model = setup_model(config, vocab_size, device)
    criterion, optimizer, scheduler = setup_training_components(config, model, word2idx)

    if clearml_logger is not None and config.get(
        "hardware.experiment_tracking.log_model_architecture", True
    ):
        clearml_logger.report_text(str(model))

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
    print(
        f"Model has {sum(parameter.numel() for parameter in model.parameters())} parameters"
    )
    print(f"Vocabulary size: {vocab_size}")

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
        val_loss = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
        )
        bleu_score, meteor_score_val = evaluate_generation(
            model,
            metrics_loader,
            device,
            idx2word,
            word2idx,
            max_length=config.get("dataset.max_caption_length", 50) - 1,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)
        meteor_scores.append(meteor_score_val)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"BLEU Score: {bleu_score:.4f}")
        print(f"METEOR Score: {meteor_score_val:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        if clearml_logger is not None:
            clearml_logger.report_scalar("Loss", "Train", train_loss, epoch)
            clearml_logger.report_scalar("Loss", "Validation", val_loss, epoch)
            clearml_logger.report_scalar("Metrics", "BLEU", bleu_score, epoch)
            clearml_logger.report_scalar("Metrics", "METEOR", meteor_score_val, epoch)
            clearml_logger.report_scalar("Learning Rate", "LR", current_lr, epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"New best validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        if config.get("training.save_checkpoints", True) and (
            epoch % save_every_n_epochs == 0 or is_best
        ):
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
            min_epochs = config.get("training.early_stopping.min_epochs", 20)
            if epoch >= min_epochs and patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final BLEU score: {bleu_scores[-1]:.4f}")
    print(f"Final METEOR score: {meteor_scores[-1]:.4f}")

    if clearml_logger is not None:
        clearml_logger.report_single_value("best_val_loss", best_val_loss)
        clearml_logger.report_single_value("final_bleu", bleu_scores[-1])
        clearml_logger.report_single_value("final_meteor", meteor_scores[-1])


if __name__ == "__main__":
    main()
