"""
04 - Transformer bloğunun bileşenleri

  1) Artık bağlantı (residual) ve katman normalizasyonu: Post-LN / Pre-LN
  2) LayerNorm ve RMSNorm, aynı vektör üzerinde adım adım
  3) İleri beslemeli katman aktivasyonları: ReLU, GELU, SwiGLU
  4) Kodlayıcı ve kod çözücü bloklarının karşılaştırılması
  5) Parametre sayısı: formülle el hesabı ve PyTorch ile doğrulama
  6) Post-LN ile Pre-LN gradyan davranışı (başlangıçta, katman katman)

Çalıştırma:  python transformer_blogu.py
"""

import math
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)  # nn.Transformer iç uyarıları
np.set_printoptions(precision=4, suppress=True)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ---------------------------------------------------------------------------
# 2) LayerNorm ve RMSNorm (NumPy)
# ---------------------------------------------------------------------------
def layer_norm(x, gamma=1.0, beta=0.0, eps=1e-5):
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    return gamma * (x - mu) / np.sqrt(var + eps) + beta, mu, var


def rms_norm(x, gamma=1.0, eps=1e-5):
    rms = np.sqrt((x ** 2).mean() + eps)
    return gamma * x / rms, rms


def norm_ornegi():
    x = np.array([2.0, -1.0, 4.0, 3.0])
    print("=" * 60)
    print("LAYERNORM vs RMSNORM    x =", x)
    y, mu, var = layer_norm(x)
    print(f"LayerNorm: ortalama = {mu:.4f}, varyans = {var:.4f}")
    print("  (x - mu)/sqrt(var+eps) =", y, "  -> ortalaması", round(y.mean(), 6), ", varyansı", round(y.var(), 4))
    z, rms = rms_norm(x)
    print(f"RMSNorm  : RMS = sqrt(mean(x^2)+eps) = {rms:.4f}")
    print("  x / RMS                =", z, "  -> ortalaması", round(z.mean(), 4), "(sıfırlanmaz), RMS'i", round(np.sqrt((z**2).mean()), 4))


# ---------------------------------------------------------------------------
# 3) Aktivasyonlar
# ---------------------------------------------------------------------------
def relu(x):
    return np.maximum(0, x)


def gelu(x):
    # tam form: x * Phi(x),  Phi = standart normal CDF
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))


def silu(x):  # Swish-1
    return x / (1 + np.exp(-x))


def aktivasyon_tablosu():
    xs = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    print("\n" + "=" * 60)
    print("AKTİVASYON DEĞERLERİ")
    print(f"{'x':>6} {'ReLU':>8} {'GELU':>8} {'SiLU':>8}")
    for x in xs:
        print(f"{x:>6.1f} {relu(x):>8.4f} {gelu(np.array(x)):>8.4f} {silu(x):>8.4f}")
    return xs


class FFN(nn.Module):
    """Standart ileri beslemeli katman: W2 act(W1 x). Parametre: 2*d*d_ff (+bias)."""

    def __init__(self, d, d_ff, act="relu", bias=True):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d, bias=bias)
        self.act = {"relu": nn.ReLU(), "gelu": nn.GELU()}[act]

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class SwiGLU(nn.Module):
    """SwiGLU: W2( SiLU(W_gate x) * (W_up x) ). Üç matris -> 3*d*d_ff parametre."""

    def __init__(self, d, d_ff, bias=False):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=bias)
        self.up = nn.Linear(d, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d, bias=bias)

    def forward(self, x):
        return self.down(nn.functional.silu(self.gate(x)) * self.up(x))


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.g * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# ---------------------------------------------------------------------------
# 1) Post-LN / Pre-LN kodlayıcı bloğu
# ---------------------------------------------------------------------------
class KodlayiciBlogu(nn.Module):
    def __init__(self, d=64, h=4, d_ff=256, norm_konumu="pre"):
        super().__init__()
        self.konum = norm_konumu
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn = FFN(d, d_ff, act="relu")
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        if self.konum == "post":  # x -> LN(x + Attn(x)) -> LN(. + FFN(.))
            x = self.ln1(x + self.attn(x, x, x, need_weights=False)[0])
            x = self.ln2(x + self.ffn(x))
        else:  # x -> x + Attn(LN(x)) -> . + FFN(LN(.))
            y = self.ln1(x)
            x = x + self.attn(y, y, y, need_weights=False)[0]
            x = x + self.ffn(self.ln2(x))
        return x


def gradyan_deneyi(katman=12, d=64, seed=0):
    """
    Başlangıç ağırlıklarında, kayıp.backward() sonrası her katmanın FFN
    ağırlığının gradyan normu. Xiong vd. (2020): Post-LN'de gradyan çıkışa
    yakın katmanlarda büyük, girişe doğru küçük; Pre-LN'de katmanlar arası
    daha dengeli.
    """
    sonuc = {}
    for konum in ("post", "pre"):
        torch.manual_seed(seed)
        bloklar = nn.ModuleList([KodlayiciBlogu(d, 4, 4 * d, konum) for _ in range(katman)])
        x = torch.randn(8, 16, d)
        y = x
        for b in bloklar:
            y = b(y)
        if konum == "pre":
            y = nn.functional.layer_norm(y, (d,))  # Pre-LN'de son LN
        y.pow(2).mean().backward()
        sonuc[konum] = [b.ffn.w2.weight.grad.norm().item() for b in bloklar]
    return sonuc


# ---------------------------------------------------------------------------
# 5) Parametre sayısı
# ---------------------------------------------------------------------------
def say(m):
    return sum(p.numel() for p in m.parameters())


def parametre_formulleri(d=512, d_ff=2048, N=6):
    """Orijinal Transformer (base): her doğrusal katmanda bias var."""
    attn = 4 * d * d + 4 * d  # W_Q, W_K, W_V, W_O  (+ biaslar)
    ffn = 2 * d * d_ff + d_ff + d
    ln = 2 * d  # gamma + beta
    kodlayici = attn + ffn + 2 * ln
    kodcozucu = 2 * attn + ffn + 3 * ln  # öz-dikkat + çapraz dikkat
    return attn, ffn, ln, kodlayici, kodcozucu, N * (kodlayici + kodcozucu)


def parametre_hesabi():
    d, d_ff, N, h = 512, 2048, 6, 8
    print("\n" + "=" * 60)
    print("PARAMETRE SAYISI: Transformer-base  (d=512, d_ff=2048, h=8, N=6)")
    attn, ffn, ln, enc, dec, toplam = parametre_formulleri(d, d_ff, N)
    print(f"Çok başlı dikkat : 4*d^2 + 4*d            = {attn:,}")
    print(f"FFN              : 2*d*d_ff + d_ff + d    = {ffn:,}")
    print(f"LayerNorm        : 2*d                    = {ln:,}")
    print(f"Kodlayıcı katmanı: attn + ffn + 2*LN      = {enc:,}")
    print(f"Kod çözücü katmanı: 2*attn + ffn + 3*LN   = {dec:,}")
    print(f"Toplam (N=6, gömme hariç, son LN hariç)   = {toplam:,}")

    # PyTorch ile doğrulama
    ek = nn.TransformerEncoderLayer(d, h, d_ff, dropout=0.0)
    dk = nn.TransformerDecoderLayer(d, h, d_ff, dropout=0.0)
    print("\nPyTorch doğrulaması:")
    print(f"  TransformerEncoderLayer = {say(ek):,}   (formül {enc:,}) -> {'EŞİT' if say(ek) == enc else 'FARKLI'}")
    print(f"  TransformerDecoderLayer = {say(dk):,}   (formül {dec:,}) -> {'EŞİT' if say(dk) == dec else 'FARKLI'}")
    tam = nn.Transformer(d, h, N, N, d_ff, dropout=0.0)
    print(f"  nn.Transformer (6+6, son LN'ler dahil) = {say(tam):,}   (formül + 2*2*d = {toplam + 4 * d:,})")

    # SwiGLU: aynı bütçeyle d_ff' = 2/3 * d_ff
    print("\nSwiGLU parametre bütçesi (bias yok):")
    d2 = 4096
    d_ff_std = 4 * d2
    d_ff_swi = int(2 * d_ff_std / 3)
    print(f"  d={d2}: standart FFN 2*d*4d = {2 * d2 * d_ff_std:,}")
    print(f"  SwiGLU 3*d*d_ff' (d_ff' = 2/3*4d = {d_ff_swi}) = {3 * d2 * d_ff_swi:,}  (~aynı)")
    print(f"  PyTorch SwiGLU(d=64, d_ff=170) = {say(SwiGLU(64, 170)):,}  (3*64*170 = {3 * 64 * 170:,})")
    print(f"  PyTorch RMSNorm(d=64) = {say(RMSNorm(64))}  vs LayerNorm(d=64) = {say(nn.LayerNorm(64))}")


def kodlayici_kodcozucu_farki():
    print("\n" + "=" * 60)
    print("KODLAYICI vs KOD ÇÖZÜCÜ BLOĞU")
    print(f"{'':<22}{'Kodlayıcı':<22}{'Kod çözücü':<30}")
    for a, b, c in [
        ("Dikkat alt katmanları", "1 (öz-dikkat)", "2 (maskeli öz + çapraz dikkat)"),
        ("Dikkat maskesi", "yok (çift yönlü)", "nedensel (yalnız geçmiş)"),
        ("K, V kaynağı", "kendi girdisi", "çapraz dikkatte kodlayıcı çıktısı"),
        ("Alt katman sayısı", "2 (dikkat, FFN)", "3 (dikkat, çapraz dikkat, FFN)"),
        ("Katman başına param.", "3,152,384", "4,204,032 (base)"),
    ]:
        print(f"{a:<22}{b:<22}{c:<30}")


def ciz(xs_dummy, grad):
    # Aktivasyonlar
    x = np.linspace(-4, 4, 400)
    plt.figure(figsize=(5.5, 3.6))
    plt.plot(x, relu(x), label="ReLU")
    plt.plot(x, gelu(x), label="GELU")
    plt.plot(x, silu(x), label="SiLU (SwiGLU içindeki)")
    plt.axhline(0, color="gray", lw=0.5)
    plt.axvline(0, color="gray", lw=0.5)
    plt.legend()
    plt.title("Aktivasyon fonksiyonları")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "aktivasyonlar.png"), dpi=150)
    plt.close()

    # Gradyan normu
    plt.figure(figsize=(5.5, 3.6))
    k = range(1, len(grad["post"]) + 1)
    plt.semilogy(k, grad["post"], "o-", label="Post-LN")
    plt.semilogy(k, grad["pre"], "s-", label="Pre-LN")
    plt.xlabel("Katman (1 = girişe en yakın)")
    plt.ylabel("FFN W2 gradyan normu (başlangıç)")
    plt.title("Katman katman gradyan, 12 katman")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "gradyan_post_pre.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    norm_ornegi()
    xs = aktivasyon_tablosu()
    kodlayici_kodcozucu_farki()
    parametre_hesabi()

    print("\n" + "=" * 60)
    print("POST-LN vs PRE-LN: başlangıçta katman katman gradyan normu (12 katman)")
    g = gradyan_deneyi()
    print(f"{'katman':>7} {'Post-LN':>12} {'Pre-LN':>12}")
    for i, (a, b) in enumerate(zip(g["post"], g["pre"]), 1):
        print(f"{i:>7} {a:>12.3e} {b:>12.3e}")
    print(f"oran (son/ilk)  Post-LN: {g['post'][-1] / g['post'][0]:.2f}   Pre-LN: {g['pre'][-1] / g['pre'][0]:.2f}")

    ciz(xs, g)
    print("\nŞekiller figures/ klasörüne kaydedildi.")
