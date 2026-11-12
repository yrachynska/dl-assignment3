# dl-assignment3

Витягування географічних назв. BIO tagging; BiLSTM/BiGRU; PyTorch Lightning + W&B. **12.11.2026: проміжні результати.**

[Kaggle](https://www.kaggle.com/competitions/kse-ua-location-extraction-2026) · [W&B: dl-3](https://wandb.ai/yrachynska-kyiv-school-of-economics/dl-3)

Ноутбуки: [01 EDA](01_eda.ipynb) → [02 split/metric](02_validation.ipynb) → [03 cache](03_data_prep.ipynb) → [04 experiments](04_experiments.ipynb) → [05 full train](05_big_models.ipynb) → [06 ensembles](06_ensembles.ipynb). Поточний runner: [rnn.ipynb](rnn.ipynb).

## Дані

| Split | Прикладів |
|---|---:|
| Train | 373 790 |
| Validation | 2 000 документів |
| Labeling | 26 документів |
| Test | 477 документів |

- 200k `uk` + 200k `ru`; group holdout ≈5%; **4 223** дублікати вилучено з train.
- Групи: `ru/doc_id`, `uk/100 рядків`; validation — склеєні документи, 90% `uk`-джерела.
- Словники: **179 708 word / 16 000 BPE**. Split: `splits.csv`, `valid_docs.csv`.

## Pipeline

1. `prepared.pkl` → IDs/offsets/labels; preprocessing один раз.
2. `Embedding(200) → BiLSTM/BiGRU(hidden=256/напрямок) → Dropout → Linear(3)`.
3. Weighted CE `[O,B,I]=[1,2,2]`; PAD `0`, IGNORE `-100`; batch **128**, max_len **256**, clipping **1.0**, seed **21**.
4. Loss: перші 256 токенів; F1: усі chunks документа → BIO decoding → назви.
5. Validation-only вибір checkpoint, threshold **0.10…0.90 / 0.05**, постобробки та ансамблю.

Метрика: `comp_metric`, document-level sets → micro F1; legacy normalization збережено.

W&B history: `train_loss`, `val_loss`, `val_f1` **при 0.5**, `grad_norm`. Summary: `val_f1@0.5`, tuned `val_f1`, `lab_f1`, `thr`, фактичний `LB`.

## Завершені результати

CV — F1 після threshold tuning. `—` — невідомий LB або не незалежний lab. `full` = 373 790; `s20/s40` = +20k/+40k synthetic; `l20` = labeling ×20.

| Серія / варіант | Train / епохи | CV F1 | Lab F1 | Public LB |
|---|---|---:|---:|---:|
| 04: word BiLSTM | 150k / 3 | 0.77304 | 0.42667 | — |
| 04: BPE BiLSTM, 2 layers | 150k / 3 | 0.75908 | 0.43590 | — |
| 04: word BiLSTM + s20 | 150k+s20 / 3 | 0.78106 | 0.50000 | — |
| 05: word BiLSTM + s40 | full+s40 / 4 | 0.84764 | 0.56410 | — |
| 05: word BiLSTM + s40+l20 | full+s40+l20 / 4 | 0.85180 | — | 0.55950 |
| 06: `ens_lab` | full+s40+l20 / 4 на модель | 0.85246 | — | **0.56553** |
| 06: `ens_four` | 4 моделі, включно з l20 | **0.85424** | — | — |
| rnn: word BiLSTM baseline | 150k / 3 | 0.77889 | 0.42254 | — |
| rnn: word BiLSTM | full / 4 | 0.83268 | 0.49351 | — |
| rnn: word BiGRU | full / 4 | 0.83100 | 0.43678 | — |
| rnn: word 50/50 + postprocessing | full / 4 на модель | 0.85246 | **0.57895** | 0.55325 |
| rnn: BPE BiLSTM, 1 layer | full / 4 | 0.77230 | 0.46341 | — |
| rnn: BPE BiLSTM, 2 layers | full / 4 | 0.81854 | 0.48193 | — |
| rnn: BPE 2 layers + postprocessing | full / 4 | 0.84282 | 0.53846 | 0.55182 |

Джерела: таблиці `04–06`, `comparison.csv`, Kaggle submissions. `ens_lab`/`ens_four`: labeling у train, незалежного lab score немає. Завершені `rnn` runs: `synth=0`, `use_lab=0`.

| Черга на L4 | Повний час, включно з evaluation |
|---|---:|
| Word LSTM + GRU, 4 епохи | 21 хв 54 с |
| BPE LSTM 1/2 layers, 4 епохи | 12 хв 57 с |

## Висновки та покращення

| Спостереження | Рішення / статус |
|---|---|
| Word → BPE: LB **0.55325 → 0.55182** | У цій серії BPE не покращив score; повернення до word |
| Word UNK: val **4.07%**, test **7.81%**; BPE test **1.32%** | Менше UNK не гарантує кращого NER; одиниці токенізації різні |
| BPE 2L, епохи 3→4: loss **0.05055→0.05191**, F1 **0.79454→0.79461** | Early stopping + вибір серед кількох checkpoints — реалізовано |
| CV **0.85246** проти LB **0.55325** | Наступна діагностика: domain/annotation shift, recurrent adversarial validation |

**Довга серія запущена; фінальних результатів ще немає.** Word BiLSTM, `lr=0.001`, cosine, seed=21, full train; максимум **10 епох/модель**.

| Run | Optimizer | Dropout | Weight decay |
|---|---|---:|---:|
| Control | Adam | 0.3 | 0 |
| Dropout | Adam | 0.4 | 0 |
| AdamW | AdamW | 0.4 | 0.01 |

Мінімум **3** епохи; early stopping: `patience=2`, `min_delta=0.001`. Checkpoints: **top-2 val_f1 + min-val_loss**, вибір за tuned CV; повний resume через `last/recovery`. Labeling поза train/selection.

## Запуск і файли

Новий Runtime: стартова комірка → **1–7 → 8 (login) → 9 → 10**. Запуск: `RUN_BATCH=True`. Resume: `BATCH_ID` + `RESUME_BATCH=True`, незмінний план. API keys — приховане введення.

```text
/content/drive/MyDrive/dl-assignment3/
  cache/v1/{prepared.pkl,bpe.json}
  checkpoints/<run_id>/{last.ckpt,recovery/last.ckpt,...}
  results/batches/<BATCH_ID>/{batch.json,comparison.csv,submission_*.csv}
```

`submission_regularized_candidate.csv` — новий кандидат; `submission_best.csv` може бути старим CV-переможцем. Деталі: [rnn_experiment_notes.md](rnn_experiment_notes.md).
