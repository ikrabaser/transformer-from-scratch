"""
06 - Metin üretiminde örnekleme stratejileri

Açgözlü (greedy) seçim, sıcaklık (temperature), top-k ve top-p (nucleus)
örneklemesi; önce küçük bir logit vektöründe adım adım, sonra eğitilmiş
mini GPT ile aynı başlangıç metni üzerinde karşılaştırılır.

Önce mini_gpt.py çalıştırılıp gpt.pt üretilmelidir.

Çalıştırma:  python ornekleme.py
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mini_gpt import KarakterVeri, MiniGPT, KLASOR, FIG_DIR

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# Strateji fonksiyonları (logit vektörü -> olasılık dağılımı)
# ---------------------------------------------------------------------------
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sicaklik_uygula(logit, T):
    """p_i ∝ exp(z_i / T).  T<1 dağılımı keskinleştirir, T>1 düzleştirir."""
    return softmax(logit / T)


def top_k_uygula(p, k):
    """En yüksek k olasılığı tut, gerisini 0 yap, yeniden normalize et."""
    p = p.copy()
    if k < len(p):
        esik = np.sort(p)[-k]
        p[p < esik] = 0.0
    return p / p.sum()


def top_p_uygula(p, top_p):
    """Kümülatif olasılığı >= top_p olan en küçük kümeyi tut (eşiği aşan token dahil)."""
    sira = np.argsort(-p)
    kum = np.cumsum(p[sira])
    kes = np.searchsorted(kum, top_p) + 1  # eşiği geçen ilk token da kalır
    tut = sira[:kes]
    q = np.zeros_like(p)
    q[tut] = p[tut]
    return q / q.sum()


def sec(p, rng, greedy=False):
    return int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))


# ---------------------------------------------------------------------------
# 1) Küçük logit örneği
# ---------------------------------------------------------------------------
def logit_ornegi():
    kelimeler = ["güzel", "iyi", "kötü", "büyük", "hızlı", "garip"]
    z = np.array([3.0, 2.5, 1.0, 0.5, 0.0, -1.0])
    print("=" * 66)
    print("LOGİT ÖRNEĞİ   logit z =", z)
    print(f"{'':<22}" + "".join(f"{k:>9}" for k in kelimeler) + f"{'toplam':>9}")

    def satir(ad, p):
        print(f"{ad:<22}" + "".join(f"{x:>9.4f}" for x in p) + f"{p.sum():>9.4f}")

    p1 = softmax(z)
    satir("softmax (T=1)", p1)
    for T in (0.5, 2.0):
        satir(f"sıcaklık T={T}", sicaklik_uygula(z, T))
    satir("top-k  (k=3)", top_k_uygula(p1, 3))
    satir("top-p  (p=0.8)", top_p_uygula(p1, 0.8))
    print("\nAçgözlü seçim:", kelimeler[int(np.argmax(p1))])
    kum = np.cumsum(np.sort(p1)[::-1])
    print("Sıralı kümülatif olasılık:", kum, "-> p=0.8 için ilk", int(np.searchsorted(kum, 0.8)) + 1, "token tutulur")
    return kelimeler, z


# ---------------------------------------------------------------------------
# 2) Eğitilmiş modelle üretim
# ---------------------------------------------------------------------------
@torch.no_grad()
def uret(model, veri, baslangic, uzunluk, strateji, rng, **kw):
    model.eval()
    idx = veri.kodla(baslangic).unsqueeze(0)
    for _ in range(uzunluk):
        girdi = idx[:, -model.blok :]
        logit = model(girdi)[0][0, -1].numpy().astype(np.float64)
        if strateji == "greedy":
            p, g = softmax(logit), True
        else:
            g = False
            p = sicaklik_uygula(logit, kw.get("T", 1.0))
            if "k" in kw:
                p = top_k_uygula(p, kw["k"])
            if "top_p" in kw:
                p = top_p_uygula(p, kw["top_p"])
        idx = torch.cat([idx, torch.tensor([[sec(p, rng, g)]])], dim=1)
    return veri.coz(idx[0])


def benzersiz_ngram_orani(metin, n=4):
    ng = [metin[i : i + n] for i in range(len(metin) - n + 1)]
    return len(set(ng)) / max(len(ng), 1)


def _sozcukler(metin):
    temiz = "".join(c if c.isalpha() or c.isspace() else " " for c in metin.lower())
    return temiz.split()


def gecerli_sozcuk_orani(metin, egitim_metni):
    """Üretilen sözcüklerin kaçı eğitim metninin sözcük dağarcığında geçiyor (kalite göstergesi)."""
    sozluk = set(_sozcukler(egitim_metni))
    s = _sozcukler(metin)[1:-1]  # ilk ve son sözcük kesilmiş olabilir
    return sum(x in sozluk for x in s) / max(len(s), 1)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    kelimeler, z = logit_ornegi()

    yol = os.path.join(KLASOR, "gpt.pt")
    if not os.path.exists(yol):
        raise SystemExit("gpt.pt yok. Önce `python mini_gpt.py` çalıştırın.")
    veri = KarakterVeri(os.path.join(KLASOR, "veri.txt"))
    ck = torch.load(yol)
    model = MiniGPT(ck["vocab"], blok=ck["blok"])
    model.load_state_dict(ck["durum"])

    baslangic = "Kütüphane şehrin"
    stratejiler = [
        ("açgözlü", "greedy", {}),
        ("T=0.5", "ornek", {"T": 0.5}),
        ("T=1.0", "ornek", {"T": 1.0}),
        ("T=1.5", "ornek", {"T": 1.5}),
        ("top-k=5", "ornek", {"k": 5}),
        ("top-p=0.9", "ornek", {"top_p": 0.9}),
    ]

    print("\n" + "=" * 66)
    print(f'ÜRETİM   başlangıç: "{baslangic}"   (aynı tohum, 160 karakter)')
    for ad, tur, kw in stratejiler:
        rng = np.random.default_rng(7)
        metin = uret(model, veri, baslangic, 160, tur, rng, **kw)
        print(f"\n[{ad}]\n{metin.replace(chr(10), ' / ')}")

    print("\n" + "=" * 66)
    print("SAYISAL KARŞILAŞTIRMA (10 örnek x 200 karakter)")
    print(f"{'strateji':<12}{'benzersiz 4-gram':>18}{'bilinen sözcük oranı':>24}")
    olcum = []
    for ad, tur, kw in stratejiler:
        rng = np.random.default_rng(0)
        n_ = 1 if tur == "greedy" else 10  # açgözlü deterministik: tek örnek yeterli
        metinler = [uret(model, veri, baslangic, 200, tur, rng, **kw) for _ in range(n_)]
        bz = float(np.mean([benzersiz_ngram_orani(m) for m in metinler]))
        ez = float(np.mean([gecerli_sozcuk_orani(m, veri.metin) for m in metinler]))
        olcum.append((ad, bz, ez))
        print(f"{ad:<12}{bz:>18.3f}{ez:>24.3f}")

    # Şekil: logit örneğinde stratejiler
    p1 = softmax(z)
    seriler = [("softmax", p1), ("T=0.5", sicaklik_uygula(z, 0.5)), ("T=2", sicaklik_uygula(z, 2.0)),
               ("top-k=3", top_k_uygula(p1, 3)), ("top-p=0.8", top_p_uygula(p1, 0.8))]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    w = 0.16
    for j, (ad, p) in enumerate(seriler):
        ax.bar(np.arange(len(kelimeler)) + (j - 2) * w, p, w, label=ad)
    ax.set_xticks(range(len(kelimeler)))
    ax.set_xticklabels(kelimeler)
    ax.set_ylabel("Olasılık")
    ax.set_title("Örnekleme stratejilerinin aynı logitler üzerindeki etkisi")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ornekleme_stratejileri.png"), dpi=150)
    plt.close()
    print("\nfigures/ornekleme_stratejileri.png kaydedildi.")
