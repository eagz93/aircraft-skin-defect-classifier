"""
models.py — Definiciones de modelos: MLP, CNN custom, wrappers para ViT y ResNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Etapa 1: MLP Base
# ---------------------------------------------------------------------------
class MLPClassifier(nn.Module):
    """
    Perceptrón multicapa para clasificación de imágenes aplanadas.
    Input: vector 1D (img_size * img_size * channels)
    """

    def __init__(self, input_dim: int = 128 * 128, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Etapa 2: CNN Profunda (desde cero)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Bloque convolucional: Conv → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_channels, out_channels, kernel_size=3, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNNClassifier(nn.Module):
    """
    CNN personalizada para clasificación de defectos.
    Arquitectura: 4 bloques Conv + GlobalAvgPool + Clasificador.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.5, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),    # 224 → 112
            ConvBlock(32, 64),              # 112 → 56
            ConvBlock(64, 128),             # 56 → 28
            ConvBlock(128, 256),            # 28 → 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class CNNClassifierDeep(nn.Module):
    """
    CNN más profunda (6 bloques) para ablation study.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),               # 224 → 112
            ConvBlock(32, 64),              # 112 → 56
            ConvBlock(64, 128),             # 56 → 28
            ConvBlock(128, 256),            # 28 → 14
            ConvBlock(256, 512),            # 14 → 7
            ConvBlock(512, 512, pool=False), # 7 → 7 (no pool)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Etapa 3: Wrappers para modelos preentrenados
# ---------------------------------------------------------------------------
def create_resnet50(num_classes: int = 5, pretrained: bool = True, freeze_backbone: bool = True):
    """
    ResNet50 preentrenado con cabeza de clasificación personalizada.
    """
    import torchvision.models as models

    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Reemplazar cabeza
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )

    return model


def create_vit(
    num_classes: int = 5,
    model_name: str = "google/vit-base-patch16-224",
    freeze_backbone: bool = True,
    gradient_checkpointing: bool = False,
):
    """
    Vision Transformer preentrenado desde Hugging Face.
    """
    from transformers import ViTForImageClassification, ViTConfig

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    return model


def unfreeze_last_n_layers(model, n: int = 2):
    """
    Descongela las últimas n capas del encoder de un ViT o las últimas n layers de ResNet.
    Útil para fine-tuning gradual.
    """
    # Para ViT (Hugging Face)
    if hasattr(model, "vit"):
        encoder_layers = model.vit.encoder.layer
        total = len(encoder_layers)
        for i, layer in enumerate(encoder_layers):
            if i >= total - n:
                for param in layer.parameters():
                    param.requires_grad = True
        return

    # Para ResNet (torchvision)
    children = list(model.children())
    for child in children[-n:]:
        for param in child.parameters():
            param.requires_grad = True


def count_parameters(model) -> dict:
    """Cuenta parámetros totales vs entrenables."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Etapa 4: Modelos generativos
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    """
    Variational Autoencoder convolucional.
    Entrada/Salida: imágenes RGB de tamaño img_size x img_size.
    """

    def __init__(self, img_size: int = 64, latent_dim: int = 128, in_channels: int = 3):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),   # /2
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),            # /4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),           # /8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),          # /16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        enc_size = img_size // 16
        self.enc_flat = 256 * enc_size * enc_size

        self.fc_mu = nn.Linear(self.enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_flat, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.enc_flat)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),
            nn.Tanh(),
        )

        self._enc_size = enc_size

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, self._enc_size, self._enc_size)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def generate(self, num_samples: int, device: str = "cpu"):
        """Genera imágenes sintéticas muestreando del espacio latente."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decode(z)


class DCGANGenerator(nn.Module):
    """Generator de DCGAN condicional por clase."""

    def __init__(self, latent_dim: int = 100, num_classes: int = 5, img_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        init_size = img_size // 16  # 4 para img_size=64

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * init_size * init_size),
            nn.BatchNorm1d(256 * init_size * init_size),
            nn.ReLU(inplace=True),
        )

        self.init_size = init_size

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        gen_input = z * label_input
        h = self.fc(gen_input)
        h = h.view(h.size(0), 256, self.init_size, self.init_size)
        return self.conv(h)


class DCGANDiscriminator(nn.Module):
    """Discriminator de DCGAN condicional por clase."""

    def __init__(self, num_classes: int = 5, img_size: int = 64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        self.img_size = img_size

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1),   # 3 (img) + 1 (label map) → 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        label_map = self.label_emb(labels)
        label_map = label_map.view(label_map.size(0), 1, self.img_size, self.img_size)
        x = torch.cat([img, label_map], dim=1)
        features = self.conv(x)
        return self.head(features)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_model(name: str, **kwargs):
    """Factory: crea modelo por nombre."""
    models = {
        "mlp": MLPClassifier,
        "cnn": CNNClassifier,
        "cnn_deep": CNNClassifierDeep,
    }
    if name in models:
        return models[name](**kwargs)
    elif name == "resnet50":
        return create_resnet50(**kwargs)
    elif name == "vit":
        return create_vit(**kwargs)
    else:
        raise ValueError(f"Modelo desconocido: {name}")
