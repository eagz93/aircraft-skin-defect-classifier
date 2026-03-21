"""
train.py — Loop de entrenamiento genérico para clasificación y modelos generativos.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Entrenamiento de clasificadores
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, is_hf_model=False):
    """Entrena una epoch de un modelo de clasificación."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if is_hf_model:
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, is_hf_model=False):
    """Evalúa el modelo en un DataLoader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)

        if is_hf_model:
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_classifier(
    model,
    train_loader,
    val_loader,
    num_epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    class_weights=None,
    device: str = "cuda",
    save_dir: str | Path = "results/models",
    model_name: str = "model",
    patience: int = 7,
    scheduler_type: str = "plateau",
    is_hf_model: bool = False,
):
    """
    Loop completo de entrenamiento con early stopping.

    Returns:
        dict con historial de métricas y ruta del mejor modelo.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Loss con pesos de clase (para desbalance)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # Scheduler
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=True)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_model_path = save_dir / f"{model_name}_best.pt"
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, is_hf_model
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, is_hf_model
        )

        # Scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        elapsed = time.time() - t0

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:>3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | {elapsed:.1f}s"
        )

        # Early stopping + save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            if is_hf_model:
                model.save_pretrained(str(save_dir / f"{model_name}_best_hf"))
            else:
                torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Mejor modelo guardado (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping en epoch {epoch} (paciencia={patience})")
                break

    # Cargar mejor modelo
    if is_hf_model:
        pass  # Se carga desde save_dir en el notebook
    elif best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

    history["best_val_acc"] = best_val_acc
    return history


# ---------------------------------------------------------------------------
# Entrenamiento de VAE
# ---------------------------------------------------------------------------
def vae_loss_fn(recon_x, x, mu, logvar, beta: float = 1.0):
    """VAE ELBO loss: reconstrucción + KL divergence."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae(
    model,
    train_loader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: str = "cuda",
    save_dir: str | Path = "results/models",
):
    """Entrena un VAE."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"total_loss": [], "recon_loss": [], "kl_loss": []}
    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_total, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        n = 0

        for images, _ in tqdm(train_loader, desc=f"VAE Epoch {epoch}", leave=False):
            images = images.to(device)
            recon, mu, logvar = model(images)

            total, recon_l, kl_l = vae_loss_fn(recon, images, mu, logvar, beta)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            epoch_total += total.item()
            epoch_recon += recon_l.item()
            epoch_kl += kl_l.item()
            n += images.size(0)

        scheduler.step()

        avg_total = epoch_total / n
        avg_recon = epoch_recon / n
        avg_kl = epoch_kl / n

        history["total_loss"].append(avg_total)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)

        print(f"Epoch {epoch:>3d}/{num_epochs} | Total: {avg_total:.2f} | Recon: {avg_recon:.2f} | KL: {avg_kl:.2f}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(model.state_dict(), save_dir / "vae_best.pt")

    return history


# ---------------------------------------------------------------------------
# Entrenamiento de GAN
# ---------------------------------------------------------------------------
def train_cgan(
    generator,
    discriminator,
    train_loader,
    num_epochs: int = 100,
    lr: float = 2e-4,
    latent_dim: int = 100,
    device: str = "cuda",
    save_dir: str | Path = "results/models",
):
    """Entrena una Conditional GAN (DCGAN)."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.BCELoss()
    opt_g = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    history = {"g_loss": [], "d_loss": []}

    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()
        epoch_g_loss, epoch_d_loss = 0.0, 0.0
        n_batches = 0

        for images, labels in tqdm(train_loader, desc=f"GAN Epoch {epoch}", leave=False):
            batch_size = images.size(0)
            images, labels = images.to(device), labels.to(device)

            real_target = torch.ones(batch_size, 1, device=device) * 0.9   # label smoothing
            fake_target = torch.zeros(batch_size, 1, device=device) + 0.1

            # --- Train Discriminator ---
            opt_d.zero_grad()
            real_pred = discriminator(images, labels)
            d_real_loss = criterion(real_pred, real_target)

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(z, labels)
            fake_pred = discriminator(fake_images.detach(), labels)
            d_fake_loss = criterion(fake_pred, fake_target)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            opt_d.step()

            # --- Train Generator ---
            opt_g.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_images = generator(z, labels)
            gen_pred = discriminator(gen_images, labels)
            g_loss = criterion(gen_pred, torch.ones(batch_size, 1, device=device))
            g_loss.backward()
            opt_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            n_batches += 1

        avg_g = epoch_g_loss / n_batches
        avg_d = epoch_d_loss / n_batches
        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3d}/{num_epochs} | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")

        # Save checkpoint
        if epoch % 25 == 0:
            torch.save(generator.state_dict(), save_dir / f"generator_epoch{epoch}.pt")
            torch.save(discriminator.state_dict(), save_dir / f"discriminator_epoch{epoch}.pt")

    # Save final
    torch.save(generator.state_dict(), save_dir / "generator_final.pt")
    torch.save(discriminator.state_dict(), save_dir / "discriminator_final.pt")

    return history
