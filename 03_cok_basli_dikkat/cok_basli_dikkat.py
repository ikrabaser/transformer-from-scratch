"""
03 - Çok başlı dikkat, öz-dikkat / çapraz dikkat ve maliyet analizi

  MultiHead(X) = Concat(head_1, ..., head_h) W_O
  head_i       = Attention(X W_Q^i, X W_K^i, X W_V^i)

Bu betik şunları gösterir:
  1) Öz-dikkat (self-attention) ve çapraz dikkat (cross-attention) farkı
  2) Başların bölünmesi ve birleştirilmesi (şekil takibi)
  3) Parametre sayısının baş sayısından bağımsız olması
  4) Dikkatin dizi uzunluğuna göre O(n^2) hesap ve bellek maliyeti
  5) Farklı başların farklı örüntüler öğrenebildiğinin görselleştirilmesi

Çalıştırma:  python cok_basli_dikkat.py
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class CokBasliDikkat:
    def __init__(self, d_model, n_bas, seed=0):
        assert d_model % n_bas == 0, "d_model, baş sayısına tam bölünmeli"
        self.d_model, self.h = d_model, n_bas
        self.d_k = d_model // n_bas
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)
        self.W_Q = rng.normal(scale=s, size=(d_model, d_model))
        self.W_K = rng.normal(scale=s, size=(d_model, d_model))
        self.W_V = rng.normal(scale=s, size=(d_model, d_model))
        self.W_O = rng.normal(scale=s, size=(d_model, d_model))

    def _bol(self, X):
        """(n, d_model) -> (h, n, d_k)"""
        n = X.shape[0]
        return X.reshape(n, self.h, self.d_k).transpose(1, 0, 2)

    def _birlestir(self, X):
        """(h, n, d_k) -> (n, d_model)"""
        h, n, d_k = X.shape
        return X.transpose(1, 0, 2).reshape(n, h * d_k)

    def __call__(self, X_sorgu, X_baglam=None, maske=None):
        """
        Öz-dikkat:      X_baglam verilmez, K ve V de X_sorgu'dan gelir.
        Çapraz dikkat:  Q, X_sorgu'dan (kod çözücü); K ve V X_baglam'dan (kodlayıcı) gelir.
        """
        if X_baglam is None:
            X_baglam = X_sorgu
        Q = self._bol(X_sorgu @ self.W_Q)
        K = self._bol(X_baglam @ self.W_K)
        V = self._bol(X_baglam @ self.W_V)

        skor = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)  # (h, n_q, n_k)
        if maske is not None:
            skor = np.where(maske[None], skor, -1e9)
        A = softmax(skor)
        basliklar = A @ V  # (h, n_q, d_k)
        cikti = self._birlestir(basliklar) @ self.W_O
        return cikti, A

    def parametre_sayisi(self):
        return sum(w.size for w in (self.W_Q, self.W_K, self.W_V, self.W_O))


def sekil_takibi():
    print("=" * 60)
    print("ŞEKİL TAKİBİ  (n=6 token, d_model=8, h=2 baş, d_k=4)")
    mha = CokBasliDikkat(d_model=8, n_bas=2)
    X = np.random.default_rng(1).normal(size=(6, 8))
    print("X                    :", X.shape)
    print("X W_Q                :", (X @ mha.W_Q).shape)
    print("böl  (h, n, d_k)     :", mha._bol(X @ mha.W_Q).shape)
    cikti, A = mha(X)
    print("dikkat A (h, n, n)   :", A.shape)
    print("birleştir (n, d_model):", mha._birlestir(A @ mha._bol(X @ mha.W_V)).shape)
    print("çıktı  (n, d_model)  :", cikti.shape)

    print("\nÇAPRAZ DİKKAT: kod çözücü 3 token, kodlayıcı 6 token")
    Y = np.random.default_rng(2).normal(size=(3, 8))
    cikti_c, A_c = mha(Y, X)
    print("A (h, n_hedef, n_kaynak):", A_c.shape, " çıktı:", cikti_c.shape)


def parametre_baslardan_bagimsiz():
    print("\n" + "=" * 60)
    print("PARAMETRE SAYISI (d_model = 512, bias yok)")
    for h in (1, 2, 8, 16):
        m = CokBasliDikkat(512, h)
        print(f"h = {h:2d}  d_k = {m.d_k:3d}  parametre = {m.parametre_sayisi():,}   (4 d_model^2 = {4*512*512:,})")


def maliyet_tablosu(d_model=512, h=8, bayt=4):
    print("\n" + "=" * 60)
    print(f"DİKKAT MALİYETİ (d_model={d_model}, h={h}, float{bayt*8})")
    print(f"{'n':>8} {'skor matrisi h*n*n':>20} {'bellek (MB)':>13} {'FLOP ~ 4*n^2*d':>18}")
    satirlar = []
    for n in (128, 512, 2048, 8192, 32768):
        eleman = h * n * n
        mb = eleman * bayt / 2**20
        flop = 4 * n * n * d_model
        satirlar.append((n, eleman, mb, flop))
        print(f"{n:>8} {eleman:>20,} {mb:>13,.1f} {flop:>18,.0f}")
    return satirlar


def ciz_basliklar():
    """Elle kurulmuş iki 'baş': biri yerel (bir önceki token), biri anlamsal benzerlik."""
    jetonlar = ["Kedi", "minderin", "üstünde", "uyudu", "çünkü", "yorgundu"]
    n = len(jetonlar)
    yerel = np.zeros((n, n))
    for i in range(n):
        yerel[i, max(i - 1, 0)] = 1.0
    rng = np.random.default_rng(3)
    E = rng.normal(size=(n, 4))
    anlamsal = softmax(E @ E.T * 0.8)

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    for a, M, b in zip(ax, (yerel, anlamsal), ("Baş 1: yerel (önceki token)", "Baş 2: içerik benzerliği")):
        a.imshow(M, cmap="Oranges", vmin=0, vmax=1)
        a.set_title(b, fontsize=10)
        a.set_xticks(range(n))
        a.set_xticklabels(jetonlar, rotation=40)
        a.set_yticks(range(n))
        a.set_yticklabels(jetonlar)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "basliklar.png"), dpi=150)
    plt.close()


def ciz_maliyet(satirlar):
    n = [s[0] for s in satirlar]
    mb = [s[2] for s in satirlar]
    plt.figure(figsize=(5.5, 3.6))
    plt.loglog(n, mb, "o-")
    plt.xlabel("Dizi uzunluğu n")
    plt.ylabel("Skor matrisi belleği (MB, h=8, float32)")
    plt.title("Dikkat belleği n² ile büyür")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "maliyet.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    sekil_takibi()
    parametre_baslardan_bagimsiz()
    satirlar = maliyet_tablosu()
    ciz_basliklar()
    ciz_maliyet(satirlar)
    print("\nŞekiller figures/ klasörüne kaydedildi.")
