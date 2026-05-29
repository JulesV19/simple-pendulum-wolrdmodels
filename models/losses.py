import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    """
    Perceptual loss basée sur VGG16 (Johnson et al., 2016).

    Compare les feature maps à trois profondeurs :
      relu1_2 → bords, couleurs        (64×64)
      relu2_2 → textures               (32×32)
      relu3_3 → structures             (16×16)

    Pourquoi ça évite le flou MSE :
      MSE minimise E[(p-t)²] → solution optimale = moyenne des modes
      → flou. VGG compare dans un espace où les activations sont éparses
      et non-linéaires → la "moyenne" n'est plus une solution facile.

    Input : tenseurs [0, 1] RGB. Normalisation ImageNet appliquée en interne.
    Handles (B, 3, H, W) et (B, T, 3, H, W).
    """

    # Indices dans vgg16.features (voir architecture VGG16)
    _SLICES = [
        (0,  4),   # relu1_2
        (4,  9),   # relu2_2  (MaxPool + block2)
        (9, 16),   # relu3_3  (MaxPool + block3)
    ]

    def __init__(
        self,
        weights: tuple[float, ...] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        from torchvision import models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.features.children())[a:b])
            for a, b in self._SLICES
        ])
        self.weights = weights

        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target : (..., 3, H, W) ∈ [0, 1]"""
        if pred.dim() == 5:
            B, T, C, H, W = pred.shape
            pred   = pred.reshape(B * T, C, H, W)
            target = target.reshape(B * T, C, H, W)

        xp = (pred   - self.mean) / self.std
        xt = (target - self.mean) / self.std

        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for w, slice_ in zip(self.weights, self.slices):
            xp = slice_(xp)
            xt = slice_(xt).detach()
            loss = loss + w * F.mse_loss(xp, xt)

        return loss


class FrequencyLoss(nn.Module):
    """
    Perte dans le domaine fréquentiel (FFT 2D).

    Pénalise les différences de spectre amplitude et phase entre pred et target.
    Complémentaire à MSE : MSE est aveugle aux hautes fréquences (textures,
    contours fins) car leur contribution à l'erreur L2 est faible en amplitude
    mais perceptuellement importante.

    Pas de dépendances externes — utilise torch.fft.

    high_freq_boost : multiplie le spectre par une rampe fréquentielle pour
      amplifier les hautes fréquences → force encore plus de netteté.
    """

    def __init__(self, high_freq_boost: bool = True):
        super().__init__()
        self.high_freq_boost = high_freq_boost

    @staticmethod
    def _freq_weight(h: int, w: int, device: torch.device) -> torch.Tensor:
        """Rampe radiale : poids proportionnel à la fréquence spatiale."""
        fy = torch.fft.fftfreq(h, device=device).abs()
        fx = torch.fft.rfftfreq(w, device=device).abs()
        return (fy[:, None] + fx[None, :]).clamp(min=1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target : (..., 3, H, W) ∈ [0, 1]"""
        if pred.dim() == 5:
            B, T, C, H, W = pred.shape
            pred   = pred.reshape(B * T, C, H, W)
            target = target.reshape(B * T, C, H, W)

        # FFT 2D réelle → (B, C, H, W//2+1) complexe
        pred_f   = torch.fft.rfft2(pred,   norm="ortho")
        target_f = torch.fft.rfft2(target, norm="ortho")

        if self.high_freq_boost:
            weight = self._freq_weight(pred.shape[-2], pred.shape[-1], pred.device)
            pred_f   = pred_f   * weight
            target_f = target_f * weight

        loss = F.mse_loss(pred_f.real, target_f.real.detach()) \
             + F.mse_loss(pred_f.imag, target_f.imag.detach())
        return loss
