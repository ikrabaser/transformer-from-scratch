# Transformer'ı Sıfırdan: Dikkat Mekanizmasından GPT'ye

[Türkçe](README.md) | [English](README.en.md)

Dikkat mekanizmasından Transformer bloğuna, konum kodlamadan küçük bir GPT'nin eğitimine ve ön eğitim hedeflerine kadar uzanan konuların **çalışan kod + adım adım el hesabı** ile anlatıldığı bir repo. Her klasör kendi başına çalıştırılabilir ve kendi `README.md`'sinde hesapların terminal çıktısıyla birebir aynı sayılarını içerir.

## İçindekiler

| Klasör | Konu | Çıktılar |
|---|---|---|
| [`01_seq2seq_dikkat`](01_seq2seq_dikkat) | Seq2seq yapısı, sabit bağlam vektörü darboğazı, toplamsal (Bahdanau) ve çarpımsal (Luong) dikkat, hizalama matrisleri | NumPy, 2 şekil |
| [`02_olceklenmis_nokta_carpimi`](02_olceklenmis_nokta_carpimi) | Ölçeklenmiş nokta çarpımı dikkati (Q, K, V), √d_k gerekçesi ve softmax doygunluğu, nedensel maske | NumPy, 2 şekil |
| [`03_cok_basli_dikkat`](03_cok_basli_dikkat) | Çok başlı öz-dikkat ve çapraz dikkat, `O(n²)` hesap/bellek maliyeti, dikkatin açıklama olarak kullanımı tartışması | NumPy, 2 şekil |
| [`04_transformer_blogu`](04_transformer_blogu) | Artık bağlantılar, Post-LN / Pre-LN, LayerNorm / RMSNorm, ReLU / GELU / SwiGLU, kodlayıcı–kod çözücü bloğu, parametre sayısı | NumPy + PyTorch, 2 şekil |
| [`05_konum_kodlama`](05_konum_kodlama) | Sinüzoidal gömme, RoPE, ALiBi, uzun dizilere genelleme | NumPy, 3 şekil |
| [`06_mini_gpt`](06_mini_gpt) | PyTorch ile karakter düzeyinde küçük GPT, eğitim kaybı, açgözlü / sıcaklık / top-k / top-p örnekleme | PyTorch, 2 şekil |
| [`07_on_egitim_hedefleri`](07_on_egitim_hedefleri) | GPT, BERT ve T5 ön eğitim hedefleri, yalnız kodlayıcı / yalnız kod çözücü / kodlayıcı-kod çözücü, ince ayar paradigması | NumPy, 2 şekil |

Önerilen okuma sırası 01 → 07'dir: her klasör bir öncekinin kavramlarını kullanır.

## Kurulum ve çalıştırma

```bash
git clone <bu-reponun-adresi>
cd transformer-from-scratch
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# her modül kendi klasöründe çalıştırılır:
cd 02_olceklenmis_nokta_carpimi
python olceklenmis_dikkat.py
```

- 01, 02, 03, 05 ve 07 yalnızca `numpy` + `matplotlib` gerektirir.
- 04 ve 06 ayrıca `torch` ister (CPU yeterlidir; 06'da eğitim 2 çekirdekte yaklaşık 30 saniye sürer).
- Şekiller her klasörün `figures/` altına kaydedilir.

## Bu repo neyi gösterir, neyi göstermez?

Sayıların ne anlama geldiği her README'de belirtilir, ama özetle:

- **Analitik / doğrulanmış:** el hesaplarının NumPy çıktısıyla eşleşmesi (02, 04, 05), parametre sayılarının PyTorch katmanlarıyla birebir doğrulanması (04), RoPE'nin göreli uzaklık özelliği ve sinüzoidal gömmenin doğrusal ilişkisi (05).
- **Küçük deneyler:** darboğaz deneyi (01), √d_k ölçeklemesi ve softmax doygunluğu (02), Post-LN / Pre-LN gradyan karşılaştırması (04, **tek tohum, 12 katman**), örnekleme stratejileri (06, **tek eğitim çalıştırması**). Yön gösterir; makalelerin sonuçlarının tekrar üretimi değildir.
- **Elle kurulmuş örnekler:** 01'deki hizalama matrisleri ve 03'teki "başlıklar" şekli öğrenilmiş değil, anlatım için elle/rastgele oluşturulmuştur. README'lerde belirtilmiştir.
- **06 ezberler.** Eğitim metni 2,7 bin karakterdir; doğrulama kaybı ~200. adımdan sonra yükselir. Model kalitesi değil, mimari ve örnekleme davranışı gösterilir. Kendi metninle `--veri` seçeneği kullanılabilir.
- **Ölçülmeyenler:** uzun dizi doğruluğu (05), GPU üzerinde süre/bellek ölçümü (03), ön eğitim hedeflerinin gerçek model kalitesine etkisi (07). Bunlar ilgili makalelere bırakılmıştır.

## Kaynaklar

- Sutskever vd. (2014), *Sequence to Sequence Learning with Neural Networks*, NeurIPS
- Bahdanau vd. (2015), *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR
- Luong vd. (2015), *Effective Approaches to Attention-based Neural Machine Translation*, EMNLP
- Vaswani vd. (2017), *Attention Is All You Need*, NeurIPS
- Alammar (2018), *The Illustrated Transformer* (blog yazısı)
- Rush (2018), *The Annotated Transformer*, NLP-OSS
- Karpathy, *nanoGPT* (GitHub deposu)
- Clark vd. (2019), *What Does BERT Look At? An Analysis of BERT's Attention*, BlackboxNLP
- Jain ve Wallace (2019), *Attention is not Explanation*, NAACL
- Wiegreffe ve Pinter (2019), *Attention is not not Explanation*, EMNLP
- Xiong vd. (2020), *On Layer Normalization in the Transformer Architecture*, ICML
- Zhang ve Sennrich (2019), *Root Mean Square Layer Normalization*, NeurIPS
- Shazeer (2020), *GLU Variants Improve Transformer*, arXiv
- Shaw vd. (2018), *Self-Attention with Relative Position Representations*, NAACL
- Su vd. (2024), *RoFormer: Enhanced Transformer with Rotary Position Embedding*, Neurocomputing
- Press vd. (2022), *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*, ICLR
- Holtzman vd. (2020), *The Curious Case of Neural Text Degeneration*, ICLR
- Radford vd. (2018), *Improving Language Understanding by Generative Pre-Training*, OpenAI teknik raporu
- Devlin vd. (2019), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL
- Raffel vd. (2020), *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, JMLR
