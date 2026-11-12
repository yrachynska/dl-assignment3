# RNN experiment notes — 2026-11-12

These notes record technical requirements and measured results. They are not a full assignment report. Source: the user-provided `AI301 Deep Learning - Assignment #3 - 2026.pdf`, pages 2–3. The current task is to improve the RNN pipeline within these technical constraints.

## Technical scope

- Use PyTorch and recurrent-like GRU/LSTM models. Transformers and self-attention are out of scope; Keras/TensorFlow/Chainer are excluded.
- Word/BPE tokenization, bidirectionality, additional recurrent layers, pretrained multilingual embeddings, optimizer/scheduler experiments, threshold and entity-merging experiments are explicitly listed.
- Ensemble GRU/LSTM predictions; logistic-regression stacking is optional, not required.
- Preserve the same splits across approaches. If adversarial validation is performed, the assignment asks for the same model family as the main task. The RandomForest check in `02_validation.ipynb` is therefore not an RNN adversarial-validation experiment.
- Report validation and actual leaderboard scores, optimization diagnostics and unsuccessful experiments. Unknown LB scores must stay unknown.

The current notebook uses existing cache/v1, PyTorch Lightning and W&B. It neither redownloads the competition dataset nor rebuilds preprocessing/tokenizers. Labeling is excluded from training and automatic parameter selection. The legacy competition metric, including its literal regex behavior, remains unchanged for team comparability. A corrected normalization would be a separately named diagnostic, not a silent replacement.

## Observed results

Validation values below use the best checkpoint and threshold tuning. Public scores are known for the selected word and BPE submissions; other individual model scores were not inferred from them.

| Candidate | Validation F1 | Labeling F1 | Public LB |
|---|---:|---:|---:|
| Word BiLSTM, 150k, 3 epochs | 0.77888730 | 0.42253521 | unknown |
| Word BiLSTM, full train, 4 epochs | 0.83267717 | 0.49350649 | unknown |
| Word BiGRU, full train, 4 epochs | 0.83099688 | 0.43678161 | unknown |
| Word ensemble 50/50 + selected postprocessing | 0.85245902 | 0.57894737 | **0.55325** |
| Prior team `ens_lab` | not verified here | not an independent holdout if labeling was in training | **0.56553** |
| BPE BiLSTM, 1 layer, full train, 4 epochs | 0.77230360 | 0.46341463 | unknown |
| BPE BiLSTM, 2 layers, full train, 4 epochs | 0.81854456 | 0.48192771 | unknown |
| Selected BPE candidate: 2-layer model + postprocessing | 0.84281581 | 0.53846154 | **0.55182** |

The new word ensemble is 0.01228 below the prior team's public score. Higher validation F1 has not established better leaderboard performance. The reference batch is `mykyta_20261112T153807Z_bb0085`; original checkpoints, predictions and CSV remain unchanged on Drive.

## Cache/prediction diagnostics

All statistics were computed from the existing trusted prepared.pkl and saved predictions, without training or preprocessing.

| Split | Documents | Word UNK % | BPE UNK % | Word documents >256 tokens | BPE documents >256 tokens |
|---|---:|---:|---:|---:|---:|
| Train | 373790 | 2.263 | 0.000 | 0 | 0 |
| Validation | 2000 | 4.068 | 0.001 | 0 | 18 |
| Labeling | 26 | 7.703 | 1.833 | 0 | 0 |
| Test | 477 | 7.811 | 1.317 | 13 | 35 |

UNK rates have different token-unit denominators for word and BPE; they are not a direct estimate of model error reduction. Word validation contains fewer unknown tokens than labeling/test. This supports testing subwords, but does not prove they will improve LB. Current inference processes every chunk of a document; long-document context boundaries may still matter.

For the selected word ensemble, validation TP/FP/FN = 2158/284/463 (precision 0.884, recall 0.823); labeling TP/FP/FN = 22/22/10 (precision 0.500, recall 0.688). Errors include incomplete multiword names, missed rare locations and extra locations in channel signatures or contexts with different annotation conventions. These examples are diagnostics; no rules are fitted to labeling golds and labeling does not choose the next winner.

## Next controlled study

Use the existing BPE cache, full train, seed 21, Adam + OneCycle, lr 0.002, embedding 200, hidden 256 per direction, dropout 0.3, batch 128, max_len 256, 4 epochs and no synthetic/labeling training data.

1. BPE BiLSTM, one recurrent layer: tokenizer comparison against the completed full-train word BiLSTM.
2. BPE BiLSTM, two recurrent layers: depth comparison against the first BPE run. Inter-layer dropout is active for the two-layer recurrent module, as in the existing architecture.
3. Compare the two models and their 50/50 probability ensemble on validation using the established threshold grid and explicit postprocessing candidates. Average token probabilities only within BPE; never align word/BPE predictions by token index.
4. Compare against the saved word ensemble. Export both `submission_best.csv` (best validation candidate, possibly the old word model) and `submission_bpe_candidate.csv` (the new BPE hypothesis). The latter prevents accidentally re-submitting the old CSV while trying to test BPE generalization.
5. Keep LB empty for new candidates until actually submitted. Record 0.55325 only on the reference word run. Checkpoints and stage state support resuming after runtime loss.

The previous word queue took 21m54s; the duration of the two-layer BPE model has not been measured. This study has an explicit epoch budget, not a guaranteed 30-minute wall-clock limit. No new GPU experiment was launched during preparation.

## Completed BPE study and next hypotheses

The BPE batch `mykyta_20261112T194220Z_164805` completed in 12m57s. The selected BPE candidate uses only the two-layer model (one-layer weight 0), threshold 0.65 and newline/emoji/letters/upper postprocessing. The user submitted `submission_mykyta_bpe_v2.csv` and reported public LB **0.55182** in a Kaggle screenshot. This is 0.00143 below our word ensemble (0.55325) and 0.01371 below the previous team result (0.56553). This BPE experiment did not establish an improvement; reduced UNK alone was insufficient.

For the two-layer model, validation loss increased from 0.05055474 at epoch 3 to 0.05191190 at epoch 4, while fixed-threshold F1 changed from 0.79453534 to 0.79460919 (gain about 0.000074). This supports mild late-stage overfitting/calibration degradation and an F1 plateau, not a conclusion of severe overfitting. The one-layer model still improved F1 from 0.75053227 to 0.75521481 despite a small loss increase. Existing checkpoint selection correctly follows the configured fixed-threshold F1 objective and therefore retained epoch 4.

Pipeline candidates from this review (implementation status updated below):

- Add configurable EarlyStopping on fixed-threshold val_f1 with min_delta=0.001 and patience=1; maintain the existing team metric semantics. Early stopping is a compute control, not a guarantee of better LB. Treat an early-stopped run as completed training in the batch state rather than as a runtime interruption.
- Keep the top two F1 checkpoints and/or the minimum-validation-loss checkpoint, then compare their threshold-tuned validation scores. A tiny fixed-threshold gain need not identify the best calibrated final candidate.
- Return to the stronger word-level architecture for controlled regularization experiments: first dropout 0.3 versus 0.4 with the other settings fixed; then Adam versus AdamW with explicit weight decay, retaining the same split. Do not change every hyperparameter at once.
- Revisit the validation-to-target distribution/annotation mismatch. The current validation/LB gap is larger than the small last-epoch loss change. If formal adversarial validation is added, use a GRU/LSTM as required by the assignment.

The submitted BPE file is `results/batches/mykyta_20261112T194220Z_164805/submission_bpe_candidate.csv`, copied locally as `submission_mykyta_bpe_v2.csv`. In this batch `submission_best.csv` is the unchanged reference word prediction. Both variants have now been submitted; submitting these exact same files again would not test a new hypothesis. Both CSV files have 477 unique IDs and valid JSON location lists.

## Longer word regularization study — prepared, not trained

The user authorized a longer experiment. The current section 9 preset is now `word_regularization`, with a maximum of 10 epochs for each of three full-train word BiLSTM runs:

| Arm | Optimizer | Dropout | Weight decay |
|---|---|---:|---:|
| Long-regime control | Adam | 0.3 | 0 |
| Dropout comparison | Adam | 0.4 | 0 |
| Optimizer/decay comparison | AdamW | 0.4 | 0.01 |

All three share seed 21, lr 0.001, cosine scheduler, embedding 200, hidden 256 per direction, one recurrent layer, batch 128, max_len 256 and the same cached full train/validation split. They use neither labeling nor synthetic training data. The long-regime control is rerun because lr/scheduler/epoch budget differ from the earlier 4-epoch reference; within the new study the first comparison changes dropout, and the second changes optimizer/decay.

Implemented:

- Early stopping monitors history `val_f1` at threshold 0.5, with at least 3 epochs, patience 2 validation checks, and min_delta 0.001. Reaching this stopping criterion is successful completion, so the batch continues to evaluation.
- Top two fixed-threshold F1 snapshots and the minimum-val_loss snapshot are retained. These evaluation-only snapshots omit optimizer states to reduce storage. Separate last/recovery checkpoints retain optimizer, scheduler, callbacks, RNG, config and provenance for training resume.
- Evaluation deduplicates snapshots by global_step, sweeps the established validation thresholds for each, chooses by tuned validation F1, and only then evaluates labeling/test. `checkpoint_scores.csv` and `thresholds_all_checkpoints.csv` document the choice. The model's checkpoint hook preserves run metadata even in lightweight snapshots.
- Individual models, all three pairwise 50/50 ensembles and the equal-weight three-model ensemble are compared with the same validation-only postprocessing/threshold grid. Token probabilities are averaged only within one tokenizer.
- Known word/BPE LB scores are logged to their own existing W&B selection runs and appear only on reference rows. New model LB values remain empty. The batch normally creates four new runs: three training runs and one selection run.
- `submission_regularized_candidate.csv` exports the new study's candidate; `submission_best.csv` remains the best overall validation candidate and may still be the old word prediction. Existing completed word/BPE batches and their checkpoints are not overwritten.

The maximum budget is 30 full-data epochs across the three models. Based on the prior word runs, reaching all limits could take roughly 1–1.5 hours; early stopping may finish sooner. This is an estimate, not a wall-clock guarantee. `RUN_BATCH` remains False until the user explicitly starts section 10.

Validation completed on tiny CPU fixtures: normal early stopping finishes successfully; lightweight snapshots contain metadata and omit optimizer state; full recovery checkpoints retain EarlyStopping state; checkpoint selection can prefer lower F1@0.5 when tuned validation F1 is better; labeling/test are not evaluated during checkpoint selection; an epoch-boundary resume reproduced uninterrupted weights and scheduler state exactly. Mocked orchestration checks covered three-member blending, reference-only LB values, completed-stage skipping and the new CSV export. No new real-data/GPU experiment was launched during preparation.

References: [Lightning EarlyStopping](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html), [Lightning ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html), [PyTorch AdamW](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html).
