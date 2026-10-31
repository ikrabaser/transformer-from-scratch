"""
02 - Ölçeklenmiş nokta çarpımı dikkati (Scaled Dot-Product Attention)

  Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) + M ) V

Bu betik şunları gösterir:
  1) Q, K, V matrislerinin girdi gömmelerinden elde edilmesi
  2) Skor -> ölçekleme -> softmax -> ağırlıklı toplam, adım adım
  3) sqrt(d_k) ile ölçeklemenin gerekçesi (softmax doygunluğu)
  4) Nedensel maske (causal mask)
  5) Dikkat matrisinin ısı haritası

Çalıştırma:  python olceklenmis_dikkat.py
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


def olceklenmis_dikkat(Q, K, V, maske=None, olcekle=True):
    """Dikkat çıktısını, skorları ve ağırlıkları döndürür."""
    d_k = Q.shape[-1]
    skor = Q @ K.T
    if olcekle:
        skor = skor / np.sqrt(d_k)
    if maske is not None:
        skor = np.where(maske, skor, -1e9)  # maskelenen yerler softmax'ta ~0 olur
    agirlik = softmax(skor, axis=-1)
    cikti = agirlik @ V
    return cikti, skor, agirlik


def nedensel_maske(n):
    """Alt üçgen: i. konum yalnızca j <= i konumlarına bakabilir."""
    return np.tril(np.ones((n, n), dtype=bool))


# ---------------------------------------------------------------------------
# Küçük örnek: 4 token, gömme boyutu 4, d_k = d_v = 3
# ---------------------------------------------------------------------------
jetonlar = ["Ben", "kitap", "okumayı", "seviyorum"]

X = np.array(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 2.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ]
)

W_Q = np.array([[1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=float)
W_K = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=float)
W_V = np.array([[0, 2, 0], [0, 3, 0], [1, 0, 3], [1, 1, 0]], dtype=float)


def adim_adim():
    print("=" * 60)
    print("ADIM 1: Q, K, V = X W_Q, X W_K, X W_V")
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    print("Q =\n", Q)
    print("K =\n", K)
    print("V =\n", V)

    d_k = Q.shape[-1]
    print(f"\nADIM 2: ham skorlar S = Q K^T   (d_k = {d_k})")
    S = Q @ K.T
    print(S)

    print(f"\nADIM 3: ölçekleme S / sqrt(d_k) = S / {np.sqrt(d_k):.4f}")
    S_olcekli = S / np.sqrt(d_k)
    print(S_olcekli)

    print("\nADIM 4: satır bazında softmax -> dikkat ağırlıkları A")
    A = softmax(S_olcekli)
    print(A)
    print("satır toplamları:", A.sum(axis=1))

    print("\nADIM 5: çıktı = A V")
    O = A @ V
    print(O)
    return Q, K, V, A


def olcekleme_gerekcesi(boyutlar=(4, 16, 64, 256, 1024), n=8, tekrar=200, seed=0):
    """
    q, k bileşenleri bağımsız N(0,1) iken q·k'nın varyansı d_k olur. Ölçeksiz
    skorların büyüklüğü d_k ile artar ve softmax tek bir elemana yığılır
    (doygunluk). Doygun bölgede gradyan ~ 0'dır. sqrt(d_k) ile bölmek
    varyansı 1'e geri çeker.
    """
    rng = np.random.default_rng(seed)
    sonuc = []
    for d in boyutlar:
        std_o, std_s, max_o, max_s = [], [], [], []
        for _ in range(tekrar):
            q = rng.normal(size=(1, d))
            K = rng.normal(size=(n, d))
            s = (q @ K.T).ravel()
            std_o.append(s.std())
            std_s.append((s / np.sqrt(d)).std())
            max_o.append(softmax(s).max())
            max_s.append(softmax(s / np.sqrt(d)).max())
        sonuc.append((d, np.mean(std_o), np.mean(std_s), np.mean(max_o), np.mean(max_s)))
    return sonuc


def ciz(A_tam, A_maske):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for a, M, b in zip(ax, (A_tam, A_maske), ("Maskesiz (çift yönlü)", "Nedensel maske")):
        im = a.imshow(M, cmap="Blues", vmin=0, vmax=1)
        a.set_title(b)
        a.set_xticks(range(len(jetonlar)))
        a.set_xticklabels(jetonlar, rotation=30)
        a.set_yticks(range(len(jetonlar)))
        a.set_yticklabels(jetonlar)
        a.set_xlabel("Anahtar (bakılan)")
        a.set_ylabel("Sorgu (bakan)")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                       color="w" if M[i, j] > 0.5 else "k", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025)
    plt.savefig(os.path.join(FIG_DIR, "dikkat_matrisi.png"), dpi=150)
    plt.close()


def ciz_doygunluk(sonuc):
    d = [r[0] for r in sonuc]
    plt.figure(figsize=(5.5, 3.6))
    plt.semilogx(d, [r[3] for r in sonuc], "o-", label="ölçeksiz")
    plt.semilogx(d, [r[4] for r in sonuc], "s-", label="√d_k ile ölçekli")
    plt.xlabel("d_k")
    plt.ylabel("softmax'ın en büyük ağırlığı (ort.)")
    plt.title("Ölçeksiz skorlarda softmax doyuyor")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "olcekleme_doygunluk.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    Q, K, V, A = adim_adim()

    print("\n" + "=" * 60)
    print("NEDENSEL MASKE (alt üçgen)")
    M = nedensel_maske(len(jetonlar))
    print(M.astype(int))
    O_m, S_m, A_m = olceklenmis_dikkat(Q, K, V, maske=M)
    print("Maskeli dikkat ağırlıkları:\n", A_m)
    print("Maskeli çıktı:\n", O_m)

    print("\n" + "=" * 60)
    print("ÖLÇEKLEME GEREKÇESİ: skor std'si ve softmax doygunluğu")
    print(f"{'d_k':>6} {'std(ölçeksiz)':>14} {'std(ölçekli)':>13} {'maks(ölçeksiz)':>15} {'maks(ölçekli)':>14}")
    sonuc = olcekleme_gerekcesi()
    for d, so, ss, mo, ms in sonuc:
        print(f"{d:>6} {so:>14.3f} {ss:>13.3f} {mo:>15.3f} {ms:>14.3f}")

    ciz(A, A_m)
    ciz_doygunluk(sonuc)
    print("\nŞekiller figures/ klasörüne kaydedildi.")
