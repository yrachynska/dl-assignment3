# dl-assignment3

Витягування географічних назв з українських і російських текстів за допомогою recurrent NER-моделей. Найкращий зафіксований **Public LB — 0.56553**.

[Kaggle](https://www.kaggle.com/competitions/kse-ua-location-extraction-2026) · [W&B: dl-3](https://wandb.ai/yrachynska-kyiv-school-of-economics/dl-3)

## Дані та валідація

| Split | Прикладів |
|---|---:|
| Train | 373 790 |
| Validation | 2 000 документів |
| Labeling | 26 документів |
| Test | 477 документів |

- 200k `uk` + 200k `ru`; group holdout ≈5%; **4 223** дублікати вилучено з train.
- Групи: `ru/doc_id`, `uk/100 рядків`; validation — склеєні документи, 90% `uk`-джерела.
- Однаковий split для всіх моделей. Labeling — додаткова перевірка цільового розподілу або навчальні дані в окремо позначених експериментах.

## Ідея рішення

1. **Sequence tagging.** Токени отримують теги `O/B/I`: поза локацією / початок / продовження. Offsets дозволяють відновити назви з оригінального тексту.
2. **Word проти BPE.** Словники **179 708 / 16 000**. Word зберігає слово як одиницю; BPE розкладає рідкісні назви й відмінкові форми на субтокени.
3. **Контекст в обох напрямках.** BiLSTM/BiGRU використовують лівий і правий контекст; порівнюємо **1–2 recurrent layers**.
4. **Дисбаланс класів.** Weighted cross-entropy: ваги `[O,B,I]=[1,2,2]`.
5. **Адаптація даних.** +**20k/40k** синтетичних прикладів з адресами й відмінковими формами; окремо — повторення **26** labeling-документів ×**20**.
6. **Decoding.** Імовірність локації `1 − P(O)` → threshold → об'єднання BIO-тегів → рядки назв. Обробляються всі chunks документа.
7. **Постобробка й ансамбль.** Перевіряємо очищення переносів/emoji та фільтри літер/регістру; усереднюємо ймовірності LSTM/GRU лише з однаковою токенізацією.

Базова архітектура: **Embedding 200 → BiRNN 256/напрямок → Dropout 0.3 → Linear 3**. Batch **128**, довжина **256**, clipping **1.0**, seed **21**; Adam/AdamW, OneCycle/cosine.

Метрика — competition micro F1 множин нормалізованих назв по документах. Threshold **0.10…0.90**, крок **0.05**; параметри decoding та ансамблю обираються на validation. Під час навчання F1 рахується при **0.5**, фінальний CV — після tuning.

## Результати

`full` = **373 790**; `s20/s40` = synthetic; `l20` = labeling ×20. CV — після threshold tuning. `—` — немає підтвердженого LB або незалежної lab-оцінки.

| Модель / варіант | Train / епохи | CV F1 | Lab F1 | Public LB |
|---|---|---:|---:|---:|
| Word BiLSTM | 150k / 3 | 0.77304 | 0.42667 | — |
| BPE BiLSTM, 2 layers | 150k / 3 | 0.75908 | 0.43590 | — |
| Word BiLSTM + s20 | 150k+s20 / 3 | 0.78106 | 0.50000 | — |
| Word BiLSTM | full / 4 | 0.83268 | 0.49351 | — |
| Word BiGRU | full / 4 | 0.83100 | 0.43678 | — |
| Word BiLSTM + s40 | full+s40 / 4 | 0.84764 | 0.56410 | — |
| Word BiLSTM + s40+l20 | full+s40+l20 / 4 | 0.85180 | — | 0.55950 |
| LSTM+GRU, 50/50 + postprocessing | full / 4 на модель | 0.85246 | **0.57895** | 0.55325 |
| LSTM+GRU з labeling (`ens_lab`) | full+s40+l20 / 4 на модель | 0.85246 | — | **0.56553** |
| Ансамбль 4 моделей (`ens_four`) | включно з s40+l20 | **0.85424** | — | — |
| BPE BiLSTM, 1 layer | full / 4 | 0.77230 | 0.46341 | — |
| BPE BiLSTM, 2 layers | full / 4 | 0.81854 | 0.48193 | — |
| BPE 2 layers + postprocessing | full / 4 | 0.84282 | 0.53846 | 0.55182 |

У моделях із `l20`, `ens_lab` і `ens_four` labeling входить у train: їхній lab score не є незалежною оцінкою. Найкращий CV та найкращий Public LB отримані різними ансамблями.

## Висновки та напрями покращення

| Спостереження | Напрям |
|---|---|
| Word → BPE: LB **0.55325 → 0.55182** | У цих конфігураціях BPE не дав приросту; подальші експерименти з word |
| Word UNK: val **4.07%**, test **7.81%**; BPE test **1.32%** | Перевірка pretrained embeddings; менше UNK саме по собі не покращило LB |
| BPE 2L, епохи 3→4: loss **0.05055→0.05191**, F1 **0.79454→0.79461** | Early stopping та вибір checkpoint після threshold tuning |
| CV **0.85246** проти LB **0.55325** | Аналіз domain/annotation shift; recurrent adversarial validation |

Розширена перевірка регуляризації: word BiLSTM, full train, `lr=0.001`, cosine; до **10 епох**. Контрольовані порівняння:

| Варіант | Optimizer | Dropout | Weight decay |
|---|---|---:|---:|
| Control | Adam | 0.3 | 0 |
| Dropout | Adam | 0.4 | 0 |
| AdamW | AdamW | 0.4 | 0.01 |

Early stopping: мінімум **3** епохи, `patience=2`, `min_delta=0.001`. Для фінальної оцінки порівнюються **top-2 за F1 + min-val_loss** checkpoints після threshold tuning. Вибір за validation; labeling поза train і автоматичним підбором.
