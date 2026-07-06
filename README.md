# SLI — Synthetic Learnability Index: Deney Altyapısı

Protokol v1 (dondurulmuş) uygulaması. Bu repo, makalenin deney hattını
aşamalı olarak inşa eder (protokol §3.6):

1. **Aşama 1 — sentetik omurga üretimi + ölçüm** (GPU gerekmez) ← şu an burası
2. Aşama 2 — forecaster eğitim döngüsü (lokal RTX 3080)
3. Aşama 3 — gerçek alanlar + EPİAŞ vaka çalışması (Colab)
4. Aşama 4 — ablasyonlar / planlı genişletmeler

---

## 1. Sıfırdan ortam kurulumu (lokal makine)

### 1a. Miniconda kur
- Windows: https://docs.conda.io/en/latest/miniconda.html → 64-bit installer
- Linux: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh`

### 1b. Ortam oluştur (Aşama 1 için yeterli)
```bash
conda create -n sli python=3.11 -y
conda activate sli
pip install -r requirements.txt
```

### 1c. Aşama 1'i çalıştır ve kapıyı (gate) kontrol et
```bash
python -m scripts.validate_backbone
```
Beklenen çıktı sonunda:
```
max calibration error : < 0.02   → PASS koşulu
Spearman rho (Ω vs perm): > 0.8  → PASS koşulu
STAGE-1 GATE: PASS
```
Bu komut ayrıca şunları üretir:
- `data/synthetic/*.npy` — 54 kalibre seri (3 Ω × 3 varyant × 6 tohum), float32
- `results/backbone_validation.csv` — spec + ölçümler (achieved_omega,
  pred_perm, mix_weight, concentration, calib_err)
- `results/backbone_validation.png` — hedef-vs-ölçülen tanı grafiği

> Not: Seriler tamamen (target_omega, seed, variant) üçlüsünden yeniden
> üretilebilir; .npy dosyaları hız içindir, kaynak-of-truth spec'tir.

---

## 2. Aşama 2 hazırlığı: GPU / PyTorch (RTX 3080)

3080 için güncel CUDA 12.x tekerlekleri sorunsuz çalışır:
```bash
conda activate sli
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install einops tqdm
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
`True NVIDIA GeForce RTX 3080` görüyorsan hazırsın. (Sürücü eskiyse önce
NVIDIA sürücüsünü güncelle; CUDA toolkit'i ayrıca kurmana gerek yok,
pip tekerleği kendi runtime'ını getirir.)

VRAM notu: protokoldeki forecaster'lar (DLinear, PatchTST, iTransformer,
TimesNet) tek değişkenli 20k-nokta serilerde 10 GB'ın çok altında kalır.

---

## 3. Gerçek veri edinme (Aşama 3'te gerekecek)

| Alan | Kaynak | Not |
|---|---|---|
| ETT (ETTh1/h2, ETTm1/m2) | github.com/zhouhaoyi/ETDataset | CSV, doğrudan |
| Electricity | Autoformer/Informer benchmark paketleri (github.com/thuml/Autoformer README'sindeki Google Drive bağlantısı) | standart split'lerle |
| Weather | aynı benchmark paketi | 21 kanal, 10-dk |
| Traffic / PEMS | aynı paket (veya Caltrans PeMS) | |
| **EPİAŞ fiyat (vaka çalışması)** | seffaflik.epias.com.tr → Şeffaflık API (ücretsiz kayıt) | PTF saatlik; elinde zaten deneyim var |

Aşama 3 script'leri bu dosyaları `data/real/<alan>/` altına bekleyecek
şekilde yazılacak.

---

## 4. Proje düzeni

```
sli/
├── src/
│   ├── measures.py      # Ω_spec (Welch, nperseg=1024 sabit) + H_perm (order=4)
│   └── signals.py       # iki-aşamalı kalibrasyonlu Fourier üreteci
├── scripts/
│   └── validate_backbone.py   # Aşama-1 kapısı
├── data/synthetic/      # üretilen omurga serileri (git'e koyma)
├── results/             # CSV + grafikler
└── requirements.txt
```

## 5. Tasarım kararları (makale Methods ile birebir)

- **Ω tanımı:** 1 − normalize spektral entropi; Welch, `nperseg=1024`
  **sabit** — ölçüm parametresi hedefe göre oynatılmaz, makalede raporlanır.
- **İki-aşamalı kalibrasyon:** (A) çekirdek genlik-konsantrasyonu ile
  çekirdeğin Ω'sı hedef+0.04 üstüne çıkarılır; (B) çekirdek/gürültü karışım
  ağırlığı bisection ile hedefe oturtulur. Tolerans 0.01.
- **İki bağımsız proxy:** Ω_spec ve H_perm; kapı koşulu sıralama tutarlılığı
  (Spearman ρ > 0.8). Bilinen ayrışma: baskın periyot çok yavaşsa H_perm
  lokal gürültüye duyarlıdır (örn. Ω*=0.8/regime/seed=3) — bu, mutlak değil
  *ordinal* kullanım kararımızın gerekçesini destekler; robustness
  bölümünde çoklu-delay H_perm ablasyonu planlıdır.
