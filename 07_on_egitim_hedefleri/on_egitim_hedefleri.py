"""
07 - Ön eğitim hedefleri: GPT (nedensel LM), BERT (maskeli LM), T5 (metinden metne)

Aynı cümle üzerinde üç hedefin girdi/etiket çiftleri kurulur, kayıp hesabına
hangi konumların girdiği gösterilir ve üç mimarinin dikkat maskeleri çizilir.

  GPT   (yalnız kod çözücü)       : P(x_t | x_<t)             -> nedensel maske
  BERT  (yalnız kodlayıcı)        : P(x_maskeli | x_görünen)  -> tam (çift yönlü) maske
  T5    (kodlayıcı-kod çözücü)    : P(hedef | bozuk girdi)    -> kodlayıcı tam, kod çözücü nedensel + çapraz dikkat

Çalıştırma:  python on_egitim_hedefleri.py
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

CUMLE = "Kedi minderin üstünde uyudu çünkü çok yorgundu".split()
N = len(CUMLE)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def capraz_entropi(logit, etiket, konumlar):
    """Yalnızca `konumlar` üzerindeki ortalama -log p(etiket)."""
    p = softmax(logit)
    return float(np.mean([-np.log(p[i, etiket[i]]) for i in konumlar]))


# Ortak sözlük ve sabit "model çıktısı" (rastgele logitler, tohum sabit)
SOZLUK = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<X>", "<Y>", "<Z>"] + sorted(set(CUMLE))
S2I = {s: i for i, s in enumerate(SOZLUK)}
V = len(SOZLUK)


def rastgele_logit(uzunluk, seed):
    return np.random.default_rng(seed).normal(size=(uzunluk, V))


# ---------------------------------------------------------------------------
# 1) GPT: nedensel dil modelleme
# ---------------------------------------------------------------------------
def gpt_ornegi():
    print("=" * 70)
    print("1) GPT - NEDENSEL DİL MODELLEME")
    girdi = ["<BOS>"] + CUMLE
    hedef = CUMLE + ["<EOS>"]  # girdi bir kaydırılmış
    print(f"{'konum':<7}{'girdi':<12}{'hedef (etiket)':<16}")
    for t, (g, h) in enumerate(zip(girdi, hedef)):
        print(f"{t:<7}{g:<12}{h:<16}")
    etiket = [S2I[h] for h in hedef]
    logit = rastgele_logit(len(girdi), 0)
    kayip = capraz_entropi(logit, etiket, range(len(girdi)))
    print(f"\nKayba giren konum sayısı: {len(girdi)} / {len(girdi)}  (her konum bir etiket üretir)")
    print(f"Rastgele logitlerle ortalama kayıp = {kayip:.4f}   (referans: ln V = {np.log(V):.4f})")
    return len(girdi)


# ---------------------------------------------------------------------------
# 2) BERT: maskeli dil modelleme (%15 seç; %80 <MASK>, %10 rastgele, %10 aynı)
# ---------------------------------------------------------------------------
def bert_maskele(token, oran=0.15, seed=3, en_az_bir=True):
    rng = np.random.default_rng(seed)
    n = len(token)
    secilen = rng.random(n) < oran
    if en_az_bir and not secilen.any():
        secilen[rng.integers(n)] = True
    girdi = list(token)
    for i in np.where(secilen)[0]:
        r = rng.random()
        if r < 0.8:
            girdi[i] = "<MASK>"
        elif r < 0.9:
            girdi[i] = str(rng.choice(sorted(set(CUMLE))))  # rastgele sözcük
        # son %10: değiştirme (aynı bırak)
    return girdi, secilen


def bert_ornegi(oran=0.4, seed=11):
    print("\n" + "=" * 70)
    print(f"2) BERT - MASKELİ DİL MODELLEME  (gösterim için oran = {oran}; gerçekte 0.15)")
    girdi, secilen = bert_maskele(CUMLE, oran=oran, seed=seed)
    print(f"{'konum':<7}{'orijinal':<12}{'girdi':<12}{'kayba girer?':<14}")
    for t, (o, g, s) in enumerate(zip(CUMLE, girdi, secilen)):
        print(f"{t:<7}{o:<12}{g:<12}{'EVET' if s else 'hayır':<14}")
    etiket = [S2I[c] for c in CUMLE]
    logit = rastgele_logit(N, 0)
    konumlar = list(np.where(secilen)[0])
    kayip = capraz_entropi(logit, etiket, konumlar)
    print(f"\nKayba giren konum: {len(konumlar)} / {N}  -> sinyal yalnızca seçilen konumlardan gelir")
    print(f"Kayıp (yalnız seçilenler) = {kayip:.4f}")

    # Gerçek oranla (0.15) beklenen etiket sayısı
    print(f"\n%15 oranında {N} tokenlik cümlede beklenen etiket sayısı ≈ {0.15 * N:.1f}  (GPT'de {N + 1})")
    print("Bu yüzden BERT, aynı veriyi daha çok geçmeyi (epoch) gerektirir: örnek başına daha az öğrenme sinyali.")
    return len(konumlar)


# ---------------------------------------------------------------------------
# 3) T5: aralık bozma (span corruption)
#    Seçilen ardışık aralıklar tek bir sentinel ile değiştirilir; hedef,
#    sentinel + silinen aralık biçimindedir.
# ---------------------------------------------------------------------------
def t5_aralik_boz(token, aralıklar):
    """aralıklar: [(baslangic, bitis_dahil_degil), ...]"""
    sentinel = ["<X>", "<Y>", "<Z>"]
    girdi, hedef, i, k = [], [], 0, 0
    for (b, e) in aralıklar:
        girdi += token[i:b] + [sentinel[k]]
        hedef += [sentinel[k]] + token[b:e]
        i, k = e, k + 1
    girdi += token[i:]
    hedef += [sentinel[k]]  # kapanış sentineli
    return girdi, hedef


def t5_ornegi():
    print("\n" + "=" * 70)
    print("3) T5 - ARALIK BOZMA (span corruption)")
    aralıklar = [(1, 2), (4, 6)]  # "minderin" ve "çünkü çok"
    girdi, hedef = t5_aralik_boz(CUMLE, aralıklar)
    print("Orijinal :", " ".join(CUMLE))
    print("Silinen  :", [" ".join(CUMLE[b:e]) for b, e in aralıklar])
    print("Kodlayıcı girdisi :", " ".join(girdi))
    print("Kod çözücü hedefi :", " ".join(hedef))
    print(f"\nKodlayıcı girdi uzunluğu = {len(girdi)} (orijinal {N}),  hedef uzunluğu = {len(hedef)}")
    print("Kayıp yalnızca kod çözücü hedefi üzerinde. Gösterim için cümlenin 3/7'si bozuldu; gerçek T5'te ~%15 bozulur,")
    print("bu durumda kodlayıcı girdisi ~%85-90, hedef dizisi ise orijinalin ~%15-20'si kadar uzunlukta olur.")
    etiket = [S2I[h] for h in hedef]
    logit = rastgele_logit(len(hedef), 0)
    kayip = capraz_entropi(logit, etiket, range(len(hedef)))
    print(f"Rastgele logitlerle kayıp = {kayip:.4f}")
    return len(hedef)


# ---------------------------------------------------------------------------
# 4) Üç mimarinin dikkat maskeleri ve karşılaştırma
# ---------------------------------------------------------------------------
def maskeler():
    tam = np.ones((N, N), dtype=int)
    nedensel = np.tril(np.ones((N, N), dtype=int))
    return tam, nedensel


def karsilastirma_tablosu():
    print("\n" + "=" * 70)
    print("4) MİMARİ KARŞILAŞTIRMASI")
    satirlar = [
        ("", "GPT", "BERT", "T5"),
        ("Mimari", "yalnız kod çözücü", "yalnız kodlayıcı", "kodlayıcı-kod çözücü"),
        ("Dikkat", "nedensel (sol->sağ)", "çift yönlü", "kodlayıcı çift yönlü,"),
        ("", "", "", "kod çözücü nedensel+çapraz"),
        ("Ön eğitim hedefi", "sonraki token", "maskeli token", "bozuk aralık -> aralık"),
        ("Kayıp konumları", "tüm konumlar", "~%15 konum", "yalnız hedef dizisi"),
        ("Doğal kullanım", "metin üretimi", "sınıflandırma/etiketleme", "çeviri/özetleme/QA"),
        ("Üretim yapabilir mi", "evet (doğal)", "hayır (doğal değil)", "evet"),
    ]
    for s in satirlar:
        print(f"{s[0]:<20}{s[1]:<22}{s[2]:<26}{s[3]:<28}")


def ciz(gpt_n, bert_n, t5_n):
    tam, nedensel = maskeler()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for a, M, b in zip(ax, (nedensel, tam), ("GPT: nedensel maske", "BERT / T5 kodlayıcı: tam maske")):
        a.imshow(M, cmap="Greens", vmin=0, vmax=1.3)
        a.set_title(b, fontsize=10)
        a.set_xticks(range(N))
        a.set_xticklabels(CUMLE, rotation=45, ha="right", fontsize=8)
        a.set_yticks(range(N))
        a.set_yticklabels(CUMLE, fontsize=8)
        a.set_xlabel("bakılan (anahtar)")
        a.set_ylabel("bakan (sorgu)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "dikkat_maskeleri.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(5.8, 3.6))
    etiketler = ["GPT\n(her konum)", "BERT\n(%40 maske, gösterim)", "T5\n(hedef dizisi)"]
    plt.bar(etiketler, [gpt_n, bert_n, t5_n], color=["#4c78a8", "#f58518", "#54a24b"])
    plt.ylabel("Kayba giren etiket sayısı (örnek cümle)")
    plt.title("Aynı cümleden çıkan öğrenme sinyali")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "etiket_sayilari.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    g = gpt_ornegi()
    b = bert_ornegi()
    t = t5_ornegi()
    karsilastirma_tablosu()
    ciz(g, b, t)
    print("\nŞekiller figures/ klasörüne kaydedildi.")
