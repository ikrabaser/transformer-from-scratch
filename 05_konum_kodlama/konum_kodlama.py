"""
05 - Konum kodlama: sinüzoidal gömmeler, RoPE ve ALiBi

Öz-dikkat permütasyon-eşdeğişkendir: sıra bilgisini kendiliğinden taşımaz.
Konum bilgisini üç farklı yoldan veririz:
  1) Sinüzoidal (mutlak) konum gömmesi: girdiye eklenir.
  2) RoPE: q ve k vektörleri konuma bağlı açıyla döndürülür; skor yalnızca
     göreli uzaklığa (m - n) bağlı olur.
  3) ALiBi: gömme yok; skora, uzaklıkla orantılı bir ceza eklenir.

Çalıştırma:  python konum_kodlama.py
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# 1) Sinüzoidal konum gömmesi
#    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
#    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
# ---------------------------------------------------------------------------
def sinuzoidal_pe(n, d):
    pos = np.arange(n)[:, None]
    i = np.arange(d // 2)[None, :]
    aci = pos / (10000 ** (2 * i / d))
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(aci)
    pe[:, 1::2] = np.cos(aci)
    return pe


def sinuzoidal_ornek():
    print("=" * 60)
    print("SİNÜZOİDAL PE  (d = 4, ilk 4 konum)")
    pe = sinuzoidal_pe(4, 4)
    print(pe)
    print("\nEl hesabı, pos = 1: sin(1/10000^0), cos(1/10000^0), sin(1/10000^0.5), cos(1/10000^0.5)")
    print("  =", [round(float(f(a)), 4) for f, a in ((np.sin, 1.0), (np.cos, 1.0), (np.sin, 0.01), (np.cos, 0.01))])

    # Doğrusal ilişki: PE(pos + k) = R(k) PE(pos), R bloklu döndürme matrisi
    d, pos, k = 8, 5, 3
    pe = sinuzoidal_pe(40, d)
    tahmin = np.zeros(d)
    for i in range(d // 2):
        w = 1 / (10000 ** (2 * i / d))
        sin_p, cos_p = pe[pos, 2 * i], pe[pos, 2 * i + 1]
        tahmin[2 * i] = sin_p * np.cos(w * k) + cos_p * np.sin(w * k)
        tahmin[2 * i + 1] = cos_p * np.cos(w * k) - sin_p * np.sin(w * k)
    print(f"\nPE({pos}+{k}) = R({k}) PE({pos}) doğrulaması, en büyük fark:", np.abs(tahmin - pe[pos + k]).max())
    return pe


# ---------------------------------------------------------------------------
# 2) RoPE (Rotary Position Embedding)
#    (x1, x2) çiftleri, m konumunda m*theta_i kadar döndürülür.
#    <R_m q, R_n k> = q^T R_{n-m} k  -> yalnızca göreli uzaklığa bağlı.
# ---------------------------------------------------------------------------
def rope_dondur(x, konum, taban=10000.0):
    """x: (d,) vektörü, konum: skaler. Ardışık çiftler döndürülür."""
    d = x.shape[-1]
    theta = taban ** (-2 * np.arange(d // 2) / d)
    aci = konum * theta
    c, s = np.cos(aci), np.sin(aci)
    x1, x2 = x[0::2], x[1::2]
    y = np.empty_like(x)
    y[0::2] = x1 * c - x2 * s
    y[1::2] = x1 * s + x2 * c
    return y


def rope_ornek():
    print("\n" + "=" * 60)
    print("RoPE: 2 boyutlu el örneği (theta = 1 rad)")
    q = np.array([1.0, 0.0])
    for m in (0, 1, 2):
        # R_m = [[cos m, -sin m], [sin m, cos m]]  ->  R_m q = (cos m, sin m)
        print(f"  konum {m}: R_{m} q =", np.array([np.cos(m), np.sin(m)]))

    rng = np.random.default_rng(0)
    d = 16
    q, k = rng.normal(size=d), rng.normal(size=d)
    print("\nGöreli uzaklık özelliği: <R_m q, R_n k> yalnızca (n-m)'e bağlı")
    print(f"{'(m, n)':>10} {'n-m':>5} {'skor':>10}")
    for m, n in [(0, 3), (5, 8), (20, 23), (100, 103), (0, 10), (7, 17)]:
        print(f"{str((m, n)):>10} {n - m:>5} {rope_dondur(q, m) @ rope_dondur(k, n):>10.6f}")
    return q, k


def rope_skor_egrisi(q, k, m=0, maks=200):
    """Aynı q, k için, sorgu konumu m iken skorun göreli uzaklık r = n - m ile değişimi."""
    return np.array([rope_dondur(q, m) @ rope_dondur(k, m + r) for r in range(maks)])


# ---------------------------------------------------------------------------
# 3) ALiBi
#    skor_{i,j} = q_i k_j / sqrt(d_k) - m_h * (i - j)       (j <= i)
#    m_h = 2^(-8h/H),  h = 1..H   (baş sayısı H = 8 için 1/2, 1/4, ..., 1/256)
# ---------------------------------------------------------------------------
def alibi_egimleri(n_bas):
    return np.array([2 ** (-8 * (h + 1) / n_bas) for h in range(n_bas)])


def alibi_bias(n, egim):
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    mesafe = np.maximum(i - j, 0)
    bias = -egim * mesafe
    return np.where(j <= i, bias, -1e9)  # nedensel maske de dahil


def alibi_ornek():
    print("\n" + "=" * 60)
    print("ALiBi")
    m = alibi_egimleri(8)
    print("8 baş için eğimler m_h:", m)
    print("\nEğim = 0.5 için 5 tokenlik bias matrisi (alt üçgen):")
    b = alibi_bias(5, 0.5)
    print(np.where(b < -1e8, np.nan, b))

    # Aynı skorlar, eğim büyüdükçe dikkat yakına yığılır
    rng = np.random.default_rng(1)
    n = 8
    skor = rng.normal(size=n)  # son token'ın 8 geçmiş token'a ham skoru
    mesafe = (n - 1) - np.arange(n)
    print("\nSon token'ın dikkat ağırlıkları (aynı ham skorlar, farklı eğim):")
    print("mesafe          :", mesafe)
    for egim in (0.0, 0.25, 1.0):
        print(f"eğim {egim:<10}:", softmax(skor - egim * mesafe))


# ---------------------------------------------------------------------------
# Uzun dizilere genelleme: öğrenilen mutlak gömme vs göreli yöntemler
# ---------------------------------------------------------------------------
def genelleme_notu(egitim_uzunlugu=512, test_uzunlugu=2048):
    print("\n" + "=" * 60)
    print(f"GENELLEME: eğitim uzunluğu {egitim_uzunlugu}, test uzunluğu {test_uzunlugu}")
    print(f"  Öğrenilen mutlak gömme : {egitim_uzunlugu}. konumdan sonrası için satır YOK (tanımsız)")
    print(f"  Sinüzoidal             : formül her konum için tanımlı ama model o aralığı hiç görmedi")
    print(f"  RoPE                   : skor yalnızca (n-m)'e bağlı; görülmemiş açılar için yine de tanımlı")
    print(f"  ALiBi                  : bias doğrusal, hiçbir parametre/gömme yok; uzunluk sınırı yok")


def ciz(pe, q, k):
    plt.figure(figsize=(6, 3.8))
    plt.imshow(sinuzoidal_pe(64, 64).T, cmap="RdBu", aspect="auto")
    plt.colorbar()
    plt.xlabel("Konum")
    plt.ylabel("Boyut")
    plt.title("Sinüzoidal konum gömmesi (d = 64)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "sinuzoidal_pe.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6.4, 3.8))
    for m, stil, kalinlik in ((0, "-", 4.0), (37, "--", 2.5), (250, ":", 1.8)):
        plt.plot(rope_skor_egrisi(q, k, m), stil, lw=kalinlik, label=f"sorgu konumu m = {m}")
    plt.legend(fontsize=8)
    plt.xlabel("Göreli uzaklık n − m")
    plt.ylabel("<R_m q, R_n k>")
    plt.title("RoPE: skor yalnızca göreli uzaklığa bağlı")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "rope_skor.png"), dpi=150)
    plt.close()

    n = 32
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))
    vmin = -0.5 * (n - 1)  # en büyük eğimin en büyük cezası: üç panelde ortak renk skalası
    for a, egim in zip(ax, (0.5, 0.0625, 0.0039)):
        b = alibi_bias(n, egim)
        im = a.imshow(np.where(b < -1e8, np.nan, b), cmap="magma", vmin=vmin, vmax=0)
        a.set_title(f"ALiBi bias, eğim = {egim}")
        a.set_xlabel("anahtar j")
        a.set_ylabel("sorgu i")
    fig.colorbar(im, ax=ax, fraction=0.02, label="skora eklenen ceza")
    plt.savefig(os.path.join(FIG_DIR, "alibi_bias.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    pe = sinuzoidal_ornek()
    q, k = rope_ornek()
    alibi_ornek()
    genelleme_notu()
    fark = max(np.abs(rope_skor_egrisi(q, k, 0) - rope_skor_egrisi(q, k, m)).max() for m in (37, 250))
    print(f"\nRoPE: m = 0, 37, 250 için skor eğrileri arasındaki en büyük fark = {fark:.2e}")
    ciz(pe, q, k)
    print("\nŞekiller figures/ klasörüne kaydedildi.")
