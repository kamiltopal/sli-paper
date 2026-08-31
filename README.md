# SLI — Synthetic Learnability Index

*When Does Synthetic Data Help Time-Series Forecasting?
An Entropy-Conditioned, Information-Routing Account.* This repository is the production-training-analysis pipeline for all experiments in the article. Each run (serial, regime, generator, architecture, repeat) is added to CSVs with its key and **continues from where it left off** (Ctrl-C safe).

## 1. Set up

```bash
conda create -n sli python=3.11 -y && conda activate sli
pip install -r requirements.txt                     # Aşama 1 (GPU'suz)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install einops tqdm                             # Aşama 2+ (GPU)
python -c "import torch; print(torch.cuda.is_available())"
```

## 2. Pipeline — ordered

| # | Comment | Produces | Note |
|---|---|---|---|
| 1 | `python -m scripts.validate_backbone` | `data/synthetic/*.npy` (54 seri, (20000,3)), `results/backbone_validation.csv/.png` | Aşama-1 kapısı: kalibrasyon hatası <0.02, Spearman ρ>0.8. GPU gerekmez. |
| 2 | `python -m scripts.smoke_test` | — | 3 mimarinin uçtan uca 2-epoch testi (<1 dk GPU). |
| 3 | `python -m scripts.run_baselines` | `results/baselines.csv` (2.430 satır) | 54×3 rejim×3 mimari×5 tekrar. `--quick` = mini tur. |
| 4 | `python -m scripts.run_augmented --arch itransformer` | `results/augmented.csv` (3.240) | 4 üreteç; havuz hücre-başına bir kez üretilir. Diğer mimariler için `--arch dlinear/patchtst` (H3 pası). |
| 5 | `python -m scripts.compute_fidelity` | `results/fidelity.csv` (648) | Eğitim-öncesi F_probe + F_spec (keşifsel; script başındaki beyana bak). |
| 6 | `python -m scripts.prepare_real` | `data/real/*.npy`, `results/real_domains.csv` | Ham dosyalar §3'teki düzende olmalı. |
| 7 | `python -m scripts.run_real` | `results/real_runs.csv` (144) | 4 alan × 2 rejim × (yok/vae/stl) × 2 mimari × 3 tekrar. |
| 8 | `python -m src.sli` | `results/effects.csv` + H1–H4 özetleri | Ön-kayıtlı analiz; kural eşiği yarı-kalibre/yarı-dondurulmuş. |

## 3. Real Data placement

```
data/real/weather/weather.csv        # Autoformer benchmark paketi
data/real/traffic/traffic.csv        # Autoformer benchmark paketi
data/real/electricity/ETTh1.csv      # github.com/zhouhaoyi/ETDataset
data/real/epias/ptf.csv              # EPİAŞ Şeffaflık, Türkçe format (;)
```
EPİAŞ: `Tarih;Saat;PTF (TL/MWh);...` başlıklı ham export olduğu gibi
bırakılır; parser Türkçe sayı biçimini ve çift timestamp'i kendisi çözer.

## 4. Files and modules

```
src/
  measures.py    Ω_spec (Welch nperseg=1024 SABİT), H_perm (order=4)
  signals.py     iki-aşamalı kalibrasyonlu üreteç + 3-kanal (H3 için)
  data.py        kronolojik split, rejim dilimi, pencereleme (sızıntısız)
  models.py      DLinear, PatchTST-lite [kanal-bağımsız]; iTransformer-lite
                 [kanal-karıştıran]; TimesNet = planlı genişletme
  train.py       sabit bütçe: Adam 1e-3, batch 64, 10 epoch, val-seçimli
  generators.py  bootstrap, vae [O=0]; seasonal(oracle), exog, stl [O=1]
                 — openness etiketleri ÖN-KAYITLI; ayrıca probe/spectral
                 fidelity fonksiyonları
  sli.py         headroom (ordinal, Ω-bandı içi), etki tablosu, karar
                 kuralı ve naif-kural kıyası
scripts/         yukarıdaki tablo; hepsi resume'lu
```

## 5. constants

- Ω ölçümü `nperseg=1024`'te sabittir; hedefe göre oynatılmaz.
- Openness etiketleri ve üreteç tasarımları pilot sonrası **donduruldu**
  (exog-v1 bilinen faz-kırma kusuruyla bilinçli olarak olduğu gibi
  raporlanır; bkz. makale §3.4 "Pilot disclosure").
- Karar kuralı eşiği omurganın yarısında kalibre edilir, kalan yarı +
  gerçek alanlarda dondurulmuş test edilir (`src/sli.py:evaluate_rule`).
- İki makine/GPU karışımı serbesttir; tekrar tohumları cihazlar-arası
  nondeterminizmi soğurur (makale §3.7).

