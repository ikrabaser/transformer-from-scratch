"""
06 - Sıfırdan küçük bir GPT (karakter düzeyinde, yalnız kod çözücü)

Mimari (GPT-2 tarzı, Pre-LN):
    token gömme + öğrenilen konum gömmesi
    N x [ LN -> nedensel çok başlı öz-dikkat -> + ;  LN -> FFN(GELU) -> + ]
    son LN -> doğrusal çıkış (ağırlıklar token gömmesiyle paylaşılır)

Eğitim hedefi: bir sonraki karakteri tahmin et (çapraz entropi).

Çalıştırma:
    python mini_gpt.py                       # eğitir, gpt.pt ve figures/egitim_kaybi.png üretir
    python mini_gpt.py --adim 1500           # adım sayısını değiştir
    python mini_gpt.py --veri buyuk_metin.txt  # kendi metninle eğit (daha büyük metin = daha az ezber)
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KLASOR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(KLASOR, "figures")


# ---------------------------------------------------------------------------
# Veri: karakter sözlüğü
# ---------------------------------------------------------------------------
class KarakterVeri:
    def __init__(self, yol, bolme=0.9):
        with open(yol, encoding="utf-8") as f:
            self.metin = f.read()
        self.karakterler = sorted(set(self.metin))
        self.k2i = {c: i for i, c in enumerate(self.karakterler)}
        self.i2k = {i: c for c, i in self.k2i.items()}
        self.vocab = len(self.karakterler)
        veri = torch.tensor([self.k2i[c] for c in self.metin], dtype=torch.long)
        kes = int(bolme * len(veri))
        self.egitim, self.dogrulama = veri[:kes], veri[kes:]

    def kodla(self, s):
        return torch.tensor([self.k2i[c] for c in s], dtype=torch.long)

    def coz(self, idx):
        return "".join(self.i2k[int(i)] for i in idx)

    def parti(self, bolum, batch, blok, cihaz="cpu"):
        v = self.egitim if bolum == "egitim" else self.dogrulama
        ix = torch.randint(len(v) - blok - 1, (batch,))
        x = torch.stack([v[i : i + blok] for i in ix])
        y = torch.stack([v[i + 1 : i + blok + 1] for i in ix])  # bir kaydırılmış hedef
        return x.to(cihaz), y.to(cihaz)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class NedenselOzDikkat(nn.Module):
    def __init__(self, d, h, blok, dropout):
        super().__init__()
        assert d % h == 0
        self.h, self.d_k = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.cikis = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("maske", torch.tril(torch.ones(blok, blok)).view(1, 1, blok, blok))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.h, self.d_k).transpose(1, 2) for t in (q, k, v))  # (B, h, T, d_k)
        skor = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        skor = skor.masked_fill(self.maske[:, :, :T, :T] == 0, float("-inf"))
        a = self.drop(F.softmax(skor, dim=-1))
        y = (a @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.cikis(y)


class Blok(nn.Module):
    def __init__(self, d, h, blok, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = NedenselOzDikkat(d, h, blok, dropout)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab, blok=64, d=96, h=4, katman=3, dropout=0.1):
        super().__init__()
        self.blok = blok
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(blok, d)
        self.drop = nn.Dropout(dropout)
        self.bloklar = nn.ModuleList([Blok(d, h, blok, dropout) for _ in range(katman)])
        self.ln_f = nn.LayerNorm(d)
        self.kafa = nn.Linear(d, vocab, bias=False)
        self.kafa.weight = self.tok.weight  # ağırlık paylaşımı
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, hedef=None):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.bloklar:
            x = b(x)
        logit = self.kafa(self.ln_f(x))
        kayip = None if hedef is None else F.cross_entropy(logit.view(-1, logit.size(-1)), hedef.view(-1))
        return logit, kayip


@torch.no_grad()
def kayip_olc(model, veri, batch, blok, tekrar=20):
    model.eval()
    sonuc = {}
    for bolum in ("egitim", "dogrulama"):
        ls = [model(*veri.parti(bolum, batch, blok))[1].item() for _ in range(tekrar)]
        sonuc[bolum] = sum(ls) / len(ls)
    model.train()
    return sonuc


def egit(adim=300, batch=32, blok=64, lr=3e-3, degerlendirme=10, seed=1337, veri_yolu=None):
    torch.manual_seed(seed)
    veri = KarakterVeri(veri_yolu or os.path.join(KLASOR, "veri.txt"))
    model = MiniGPT(veri.vocab, blok=blok)
    print(f"karakter sayısı (vocab) = {veri.vocab}, toplam karakter = {len(veri.metin)}")
    print(f"model parametre sayısı = {sum(p.numel() for p in model.parameters()):,}")
    print(f"başlangıç kaybı beklentisi ln(vocab) = {math.log(veri.vocab):.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adim, eta_min=lr / 10)
    kayitlar = []
    t0 = time.time()
    for it in range(adim + 1):
        if it % degerlendirme == 0:
            k = kayip_olc(model, veri, batch, blok)
            kayitlar.append((it, k["egitim"], k["dogrulama"]))
            print(f"adım {it:5d} | eğitim {k['egitim']:.3f} | doğrulama {k['dogrulama']:.3f} | {time.time() - t0:5.1f} s")
        if it == adim:
            break
        x, y = veri.parti("egitim", batch, blok)
        _, kayip = model(x, y)
        opt.zero_grad(set_to_none=True)
        kayip.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    torch.save({"durum": model.state_dict(), "vocab": veri.vocab, "blok": blok}, os.path.join(KLASOR, "gpt.pt"))
    return kayitlar


def ciz(kayitlar):
    os.makedirs(FIG_DIR, exist_ok=True)
    it = [k[0] for k in kayitlar]
    plt.figure(figsize=(6, 3.8))
    plt.plot(it, [k[1] for k in kayitlar], label="eğitim")
    plt.plot(it, [k[2] for k in kayitlar], label="doğrulama")
    plt.xlabel("Adım")
    plt.ylabel("Çapraz entropi kaybı")
    plt.title("Mini GPT eğitim kaybı")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "egitim_kaybi.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adim", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--veri", default=None, help="eğitim metni (varsayılan: veri.txt)")
    args = ap.parse_args()
    ciz(egit(adim=args.adim, batch=args.batch, veri_yolu=args.veri))
    print("\ngpt.pt ve figures/egitim_kaybi.png kaydedildi.")
