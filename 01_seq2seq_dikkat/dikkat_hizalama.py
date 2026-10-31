"""
01 - Seq2seq yapısı ve dikkat mekanizmasının ortaya çıkışı

Bu betik üç şeyi gösterir:
  1) Sabit uzunluklu bağlam vektörünün (kodlayıcının son gizli durumu) uzun
     dizilerde başlangıçtaki bilgiyi nasıl kaybettiği.
  2) Toplamsal (Bahdanau) ve çarpımsal (Luong) dikkat skorlarının küçük bir
     örnek üzerinde adım adım hesaplanması.
  3) Kod çözücünün her adımı için hizalama (alignment) matrisinin çıkarılıp
     ısı haritası olarak çizilmesi.

Çalıştırma:  python dikkat_hizalama.py
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # sayısal kararlılık
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# 1) Sabit uzunluklu bağlam vektörü darboğazı
# ---------------------------------------------------------------------------
def darbogaz_deneyi(uzunluklar=(2, 5, 10, 20, 40, 80), d=16, tekrar=30, seed=0):
    """
    Basit bir tanh RNN kodlayıcı kurulur. Giriş dizisinin yalnızca İLK
    elemanı biraz değiştirilir; kodlayıcının son gizli durumunun ne kadar
    değiştiği ölçülür. Uzun dizilerde bu fark sıfıra yaklaşır: ilk kelimeyle
    ilgili bilgi, tek bir sabit vektörün içinde "silinir".
    """
    rng = np.random.default_rng(seed)
    sonuc = []
    for T in uzunluklar:
        farklar = []
        for _ in range(tekrar):
            W = rng.normal(size=(d, d))
            W *= 0.9 / np.max(np.abs(np.linalg.eigvals(W)))  # spektral yarıçap 0.9
            U = rng.normal(size=(d, d)) * 0.5
            x = rng.normal(size=(T, d))
            x2 = x.copy()
            x2[0] += rng.normal(size=d)  # yalnızca ilk eleman bozuldu

            def kodla(girdi):
                h = np.zeros(d)
                for t in range(T):
                    h = np.tanh(W @ h + U @ girdi[t])
                return h

            farklar.append(np.linalg.norm(kodla(x) - kodla(x2)))
        sonuc.append(np.mean(farklar))
    return list(uzunluklar), sonuc


# ---------------------------------------------------------------------------
# 2) Toplamsal ve çarpımsal dikkat skorları
# ---------------------------------------------------------------------------
def bahdanau_skor(s, H, W_s, W_h, v):
    """e_j = v^T tanh(W_s s + W_h h_j)   (toplamsal dikkat)"""
    return np.array([v @ np.tanh(W_s @ s + W_h @ h) for h in H])


def luong_nokta_skor(s, H):
    """e_j = s^T h_j   (çarpımsal, 'dot')"""
    return H @ s


def luong_genel_skor(s, H, W):
    """e_j = s^T W h_j   (çarpımsal, 'general')"""
    return np.array([s @ W @ h for h in H])


def dikkat_uygula(skorlar, H):
    alfa = softmax(skorlar)
    baglam = alfa @ H  # c = sum_j alfa_j h_j
    return alfa, baglam


# Kodlayıcı gizli durumları: 4 kaynak kelime, her biri 3 boyutlu
H = np.array(
    [
        [1.0, 0.0, 0.5],  # h1
        [0.0, 1.0, 0.5],  # h2
        [1.0, 1.0, 0.0],  # h3
        [0.5, 0.0, 1.0],  # h4
    ]
)
# Kod çözücü durumları: 3 hedef kelime adımı
S = np.array(
    [
        [1.0, 0.0, 0.5],  # s1 -> h1'e benziyor
        [0.0, 1.0, 0.5],  # s2 -> h2'ye benziyor
        [0.5, 0.0, 1.0],  # s3 -> h4'e benziyor
    ]
)

# Bahdanau parametreleri (küçük, elle izlenebilir)
W_s = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])  # (2x3)
W_h = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])  # (2x3)
v = np.array([1.0, 1.0])
# Luong 'general' ağırlık matrisi
W_luong = np.eye(3) * 0.5


def adim_adim_ornek():
    print("=" * 60)
    print("KODLAYICI DURUMLARI H (4 x 3)")
    print(H)
    s = S[0]
    print("\nKod çözücü durumu s1 =", s)

    print("\n--- Sabit bağlam vektörü (yalnızca son durum) ---")
    print("c_sabit = h4 =", H[-1])

    print("\n--- Luong nokta çarpımı ---")
    e = luong_nokta_skor(s, H)
    alfa, c = dikkat_uygula(e, H)
    print("skorlar e =", e)
    print("alfa = softmax(e) =", alfa, " toplam =", alfa.sum())
    print("bağlam c = sum alfa_j h_j =", c)

    print("\n--- Luong genel (s^T W h) ---")
    e = luong_genel_skor(s, H, W_luong)
    alfa, c = dikkat_uygula(e, H)
    print("skorlar e =", e)
    print("alfa =", alfa)
    print("bağlam c =", c)

    print("\n--- Bahdanau toplamsal (v^T tanh(W_s s + W_h h_j)) ---")
    e = bahdanau_skor(s, H, W_s, W_h, v)
    alfa, c = dikkat_uygula(e, H)
    print("skorlar e =", e)
    print("alfa =", alfa)
    print("bağlam c =", c)


def hizalama_matrisleri():
    A_dot = np.vstack([dikkat_uygula(luong_nokta_skor(s, H), H)[0] for s in S])
    A_bah = np.vstack(
        [dikkat_uygula(bahdanau_skor(s, H, W_s, W_h, v), H)[0] for s in S]
    )
    return A_dot, A_bah


def ciz(uzunluklar, farklar, A_dot, A_bah):
    # Şekil 1: darboğaz
    plt.figure(figsize=(5.5, 3.6))
    plt.plot(uzunluklar, farklar, "o-")
    plt.xlabel("Kaynak dizi uzunluğu T")
    plt.ylabel("‖h_T(x) − h_T(x')‖  (ilk eleman değişince)")
    plt.title("Sabit bağlam vektörü ilk kelimeyi unutuyor")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "darbogaz.png"), dpi=150)
    plt.close()

    # Şekil 2: hizalama matrisleri
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, M, baslik in zip(ax, (A_dot, A_bah), ("Luong (nokta)", "Bahdanau (toplamsal)")):
        im = a.imshow(M, cmap="viridis", vmin=0, vmax=1)
        a.set_title(baslik)
        a.set_xticks(range(4))
        a.set_xticklabels([f"h{j+1}" for j in range(4)])
        a.set_yticks(range(3))
        a.set_yticklabels([f"s{i+1}" for i in range(3)])
        a.set_xlabel("Kaynak konumu")
        a.set_ylabel("Hedef adımı")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025)
    plt.savefig(os.path.join(FIG_DIR, "hizalama_matrisi.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    adim_adim_ornek()

    print("\n" + "=" * 60)
    print("HİZALAMA MATRİSLERİ (satır: hedef adımı, sütun: kaynak konumu)")
    A_dot, A_bah = hizalama_matrisleri()
    print("Luong nokta:\n", A_dot)
    print("Bahdanau:\n", A_bah)

    print("\n" + "=" * 60)
    print("DARBOĞAZ DENEYİ")
    T, f = darbogaz_deneyi()
    for t, x in zip(T, f):
        print(f"T={t:3d}  ortalama fark = {x:.6f}")

    ciz(T, f, A_dot, A_bah)
    print("\nŞekiller figures/ klasörüne kaydedildi.")
