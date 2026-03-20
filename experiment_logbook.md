## Experiment 1: Baseline

**Date**: 2026-03-19
**Hypothesis**: The current checked-in Muon convolution setup should land near, but likely below, the 94% TTA validation target after 8 epochs. This establishes the reference point before any tuning.
**Change**: None. Ran the code as-is.

**Command**:
```bash
source scripts/env.sh && python airbench94_conv_muon.py --run-name baseline-2026-03-19
```

**Results**:
- Mean TTA val accuracy: `0.9385`
- TTA val accuracy std: `0.0012`
- Mean val accuracy: `0.9296`
- Mean train accuracy: `0.9667`
- Mean train loss: `inf`
- Epochs: `8`
- Batch size: `2000`
- Measured runs: `5` plus `1` warmup
- Final per-run TTA val accuracies: `0.9387`, `0.9394`, `0.9369`, `0.9374`, `0.9401`
- Typical measured runtime per run: about `4.25s`
- Log artifact: `/home/thibaut.boissin/projects/muon_conv2/logs/65ee1942-2839-4c18-ae74-fc4bea2a97e3/log.pt`
- W&B run: `baseline-2026-03-19` (`7cha8k77`)

**Observations**:
- The current configuration is below the 94% target on mean TTA accuracy, though one run reached `0.9401`.
- Train accuracy is materially higher than validation accuracy, which is consistent with the user’s overfitting hypothesis.
- `train_loss_mean: inf` is a numerical warning worth investigating separately; accuracy remains stable enough that this may be coming from logging overflow in half precision rather than total training failure.
- Best next experiment should stay small and isolated. The strongest candidates from current evidence are stronger augmentation, slightly higher weight decay, or a mild reduction in Muon update aggressiveness.

## Experiment 2: Higher Weight Decay

**Date**: 2026-03-19
**Hypothesis**: The baseline shows a clear train/validation gap, so a modest increase in weight decay should reduce overfitting and improve mean TTA validation accuracy without materially hurting convergence speed.
**Change**: Increased `DEFAULT_OPTIMIZER_CONFIG["weight_decay_scale"]` from `1e-6` to `2e-6` in `airbench94_conv_muon.py`.

**Command**:
```bash
source scripts/env.sh && python airbench94_conv_muon.py --run-name exp2-weight-decay-2e-6-2026-03-19
```

**Results**:
- Mean TTA val accuracy: `0.9395`
- TTA val accuracy std: `0.0018`
- Mean val accuracy: `0.9310`
- Mean train accuracy: `0.9648`
- Mean train loss: `inf`
- Epochs: `8`
- Batch size: `2000`
- Measured runs: `5` plus `1` warmup
- Final per-run TTA val accuracies: `0.9368`, `0.9389`, `0.9412`, `0.9386`, `0.9418`
- Typical measured runtime per run: about `4.27s`
- Log artifact: `/home/thibaut.boissin/projects/muon_conv2/logs/90fd0845-7643-49c2-8bce-0130e7bac269/log.pt`
- W&B run: `exp2-weight-decay-2e-6-2026-03-19` (`ks9vm9bv`)

**Observations**:
- This improved the mean TTA validation accuracy by about `+0.0010` over baseline (`0.9395` vs `0.9385`).
- Mean train accuracy decreased slightly while mean validation accuracy increased, which is the expected direction if the extra regularization is helping.
- The model is still below the 94% target on average, but two runs exceeded `0.94`, so the setting looks directionally useful.
- `train_loss_mean` remains `inf`, so the loss logging issue is independent of this regularization tweak.
- The next best isolated test is stronger augmentation, since H1 and H2 now both have some support: Muon appears to fit fast, and modest extra regularization helped.

## Experiment 3: Muon Weight Decay

**Date**: 2026-03-19
**Hypothesis**: Since Muon-managed convolution weights previously received no explicit decay, adding Muon-side L2 regularization might further reduce overfitting beyond Experiment 2.
**Change**: Added a Muon-specific weight decay term by injecting `weight_decay * p` into the Muon gradient path for conv filters only, while keeping Experiment 2's `weight_decay_scale=2e-6` elsewhere.

**Command**:
```bash
source scripts/env.sh && python airbench94_conv_muon.py --run-name exp3-muon-weight-decay-2026-03-19
```

**Results**:
- Mean TTA val accuracy: `0.9388`
- TTA val accuracy std: `0.0011`
- Mean val accuracy: `0.9320`
- Mean train accuracy: `0.9681`
- Mean train loss: `inf`
- Epochs: `8`
- Batch size: `2000`
- Measured runs: `5` plus `1` warmup
- Final per-run TTA val accuracies: `0.9373`, `0.9379`, `0.9389`, `0.9396`, `0.9402`
- Typical measured runtime per run: about `4.27s`
- Log artifact: `/home/thibaut.boissin/projects/muon_conv2/logs/a395eb65-74d8-4cfc-9264-edab95d494c9/log.pt`
- W&B run: `exp3-muon-weight-decay-2026-03-19` (`seewl49n`)

**Observations**:
- This underperformed Experiment 2 on the main objective (`0.9388` vs `0.9395` mean TTA val accuracy).
- Mean train accuracy increased relative to Experiment 2, so this Muon regularization path did not reduce fitting in the way we wanted.
- The effect is not catastrophic, but it is not a win, so the Muon weight decay implementation was reverted after the run.
- This is evidence against the specific Muon-side L2-gradient implementation tried here, not against all possible Muon regularization schemes.

## Experiment 4: Remove Muon Weight Renormalization

**Date**: 2026-03-19
**Hypothesis**: Removing Muon's per-step weight renormalization without retuning `muon_lr` should degrade generalization, likely by making updates too aggressive and increasing overfitting.
**Change**: Removed the line `p.data.mul_(len(p.data) ** 0.5 / p.data.norm())` from `Muon.step()`, keeping the Experiment 2 configuration otherwise unchanged.

**Command**:
```bash
source scripts/env.sh && python airbench94_conv_muon.py --run-name exp4-no-muon-renorm-2026-03-19
```

**Results**:
- Mean TTA val accuracy: `0.9356`
- TTA val accuracy std: `0.0012`
- Mean val accuracy: `0.9260`
- Mean train accuracy: `0.9843`
- Mean train loss: `inf`
- Epochs: `8`
- Batch size: `2000`
- Measured runs: `5` plus `1` warmup
- Final per-run TTA val accuracies: `0.9376`, `0.9344`, `0.9349`, `0.9348`, `0.9365`
- Typical measured runtime per run: about `4.25s`
- Log artifact: `/home/thibaut.boissin/projects/muon_conv2/logs/ff7829d2-7281-41b2-9bd4-670ffa4b2f21/log.pt`
- W&B run: `exp4-no-muon-renorm-2026-03-19` (`rwgcffa6`)

**Observations**:
- This is a clear regression relative to Experiment 2 (`0.9356` vs `0.9395` mean TTA val accuracy).
- Mean train accuracy increased sharply to `0.9843`, consistent with the expected "too much effective step size / more overfitting" behavior when renormalization is removed without retuning `muon_lr`.
- The degradation is larger than run-to-run noise, so the normalization step appears materially important at the current learning rate.
- The renormalization line was restored after the run to keep the code at the best known setting.

## Experiment 5: Remove Renormalization + Stronger Augmentation

**Date**: 2026-03-19
**Hypothesis**: If the no-renormalization setting mainly fails because it overfits too aggressively, pairing it with stronger augmentation could recover some generalization without changing the Muon learning rate.
**Change**: Removed the Muon per-step weight renormalization line again, and increased training translation augmentation from `2` to `4` via `--aug-translate 4`. All other settings matched Experiment 2.

**Command**:
```bash
source scripts/env.sh && python airbench94_conv_muon.py --run-name exp5-no-renorm-aug4-2026-03-19 --aug-translate 4
```

**Results**:
- Mean TTA val accuracy: `0.9341`
- TTA val accuracy std: `0.0009`
- Mean val accuracy: `0.9247`
- Mean train accuracy: `0.9745`
- Mean train loss: `inf`
- Epochs: `8`
- Batch size: `2000`
- Measured runs: `5` plus `1` warmup
- Final per-run TTA val accuracies: `0.9343`, `0.9334`, `0.9358`, `0.9331`, `0.9340`
- Typical measured runtime per run: about `4.28s`
- Log artifact: `/home/thibaut.boissin/projects/muon_conv2/logs/cc09cdcd-000e-4b2d-9b0c-98e5599cfd1a/log.pt`
- W&B run: `exp5-no-renorm-aug4-2026-03-19` (`esogfozn`)

**Observations**:
- Stronger augmentation did not rescue the no-renormalization setting; it actually underperformed Experiment 4 (`0.9341` vs `0.9356` mean TTA val accuracy).
- Train accuracy was lower than Experiment 4 but still much higher than Experiment 2, so augmentation only partially reduced the overfitting signal.
- The dominant issue still appears to be update scale / optimizer calibration after removing renormalization, not insufficient augmentation alone.
- The renormalization line was restored after the run to keep the code at the best known setting.

## Experiment 6: Remove Renormalization + Stronger Weight Decay

**Date**: 2026-03-19
**Hypothesis**: If no-renormalization mainly fails due to overfitting, increasing the usual weight decay under the original augmentation setting could recover some generalization without changing `muon_lr`.
**Change**: Removed the Muon per-step weight renormalization line and increased `weight_decay_scale` from `2e-6` to `4e-6`, keeping the usual augmentation (`aug_translate=2`).

**Command**:
```bash
source scripts/env.sh && python airbench94_conv_muon.py --run-name exp6-no-renorm-wd4e-6-2026-03-19
```

**Results**:
- Mean TTA val accuracy: `0.9347`
- TTA val accuracy std: `0.0018`
- Mean val accuracy: `0.9252`
- Mean train accuracy: `0.9837`
- Mean train loss: `inf`
- Epochs: `8`
- Batch size: `2000`
- Measured runs: `5` plus `1` warmup
- Final per-run TTA val accuracies: `0.9327`, `0.9341`, `0.9331`, `0.9374`, `0.9363`
- Typical measured runtime per run: about `4.25s`
- Log artifact: `/home/thibaut.boissin/projects/muon_conv2/logs/94360808-1e6b-433b-b459-2af2bde7f845/log.pt`
- W&B run: `exp6-no-renorm-wd4e-6-2026-03-19` (`h0cwrnx8`)

**Observations**:
- This is slightly better than the other no-renormalization variants, but still clearly below Experiment 2 (`0.9347` vs `0.9395` mean TTA val accuracy).
- Mean train accuracy remained extremely high at `0.9837`, so stronger weight decay did not fix the main overfitting / scale-control problem.
- This suggests the renormalization step is providing something stronger than what modest extra regularization can replace at the current learning rate.
- The code was restored after the run to the best known setting with renormalization enabled and `weight_decay_scale=2e-6`.

## Experiment 7: Remove Renormalization + Stronger Weight Decay + Lower Muon LR

**Date**: 2026-03-19
**Hypothesis**: The no-renormalization setting may need a smaller Muon learning rate to show its potential. Lowering `muon_lr` while keeping the stronger weight decay should reduce the scale mismatch seen in Experiments 4-6.
**Change**: Removed the Muon per-step weight renormalization line, kept `weight_decay_scale=4e-6`, and lowered `muon_lr` from `0.52` to `0.35`. Usual augmentation (`aug_translate=2`) remained unchanged.

**Command**:
```bash
source scripts/env.sh && python airbench94_conv_muon.py --run-name exp7-no-renorm-wd4e-6-muonlr0.35-2026-03-19
```

**Results**:
- Mean TTA val accuracy: `0.9334`
- TTA val accuracy std: `0.0014`
- Mean val accuracy: `0.9241`
- Mean train accuracy: `0.9795`
- Mean train loss: `inf`
- Epochs: `8`
- Batch size: `2000`
- Measured runs: `5` plus `1` warmup
- Final per-run TTA val accuracies: `0.9333`, `0.9317`, `0.9321`, `0.9352`, `0.9346`
- Typical measured runtime per run: about `4.27s`
- Log artifact: `/home/thibaut.boissin/projects/muon_conv2/logs/2392e516-5460-4d7e-b38f-8b4a29e72ca3/log.pt`
- W&B run: `exp7-no-renorm-wd4e-6-muonlr0.35-2026-03-19` (`gp912ayf`)

**Observations**:
- This underperformed Experiment 6 (`0.9334` vs `0.9347` mean TTA val accuracy), so `muon_lr=0.35` is too low for this no-renormalization setup.
- Mean train accuracy dropped relative to Experiment 6, which confirms that lowering the learning rate changed the optimization regime, but it did not translate into better validation performance.
- The no-renormalization idea still has not matched the renormalized baseline; if pursued further, the next sensible retune would be between `0.35` and `0.52` rather than lower.
- The code was restored after the run to the best known setting with renormalization enabled, `weight_decay_scale=2e-6`, and `muon_lr=0.52`.

## Experiment 8: Stronger Augmentation on Best Mainline

**Date**: 2026-03-19
**Hypothesis**: Since the best mainline setup still shows a train/validation gap, stronger translation augmentation might improve generalization without changing the optimizer behavior.
**Change**: Kept the best known code state unchanged and increased training translation augmentation from `2` to `4` via `--aug-translate 4`.

**Command**:
```bash
source scripts/env.sh && python airbench94_conv_muon.py --run-name exp8-aug-translate4-2026-03-19 --aug-translate 4
```

**Results**:
- Mean TTA val accuracy: `0.9366`
- TTA val accuracy std: `0.0009`
- Mean val accuracy: `0.9278`
- Mean train accuracy: `0.9555`
- Mean train loss: `inf`
- Epochs: `8`
- Batch size: `2000`
- Measured runs: `5` plus `1` warmup
- Final per-run TTA val accuracies: `0.9373`, `0.9360`, `0.9365`, `0.9379`, `0.9352`
- Typical measured runtime per run: about `4.30s`
- Log artifact: `/home/thibaut.boissin/projects/muon_conv2/logs/d2938314-a2ee-4fb1-a95d-b94bb380cf88/log.pt`
- W&B run: `exp8-aug-translate4-2026-03-19` (`osfkf2h4`)

**Observations**:
- Stronger augmentation reduced mean train accuracy substantially, so it did act as regularization.
- Despite that, it underperformed the current best mainline result from Experiment 2 (`0.9366` vs `0.9395` mean TTA val accuracy).
- This suggests the mainline bottleneck is not simply insufficient augmentation at the current epoch budget.
- The best known configuration remains Experiment 2: renormalization enabled, `weight_decay_scale=2e-6`, `muon_lr=0.52`, `aug_translate=2`.

## Experiment 9: Experiment 2 Replicate

**Date**: 2026-03-19
**Hypothesis**: The original Experiment 2 result (`0.9395` mean TTA val accuracy) may have been a favorable sample, so rerunning the exact same configuration should tell us whether it is stable or an outlier.
**Change**: None relative to Experiment 2. Re-ran the exact same checked-in configuration.

**Command**:
```bash
source scripts/env.sh && python airbench94_conv_muon.py --run-name exp9-exp2-replica-2026-03-19
```

**Results**:
- Mean TTA val accuracy: `0.9390`
- TTA val accuracy std: `0.0013`
- Mean val accuracy: `0.9323`
- Mean train accuracy: `0.9630`
- Mean train loss: `inf`
- Epochs: `8`
- Batch size: `2000`
- Measured runs: `5` plus `1` warmup
- Final per-run TTA val accuracies: `0.9405`, `0.9384`, `0.9385`, `0.9405`, `0.9370`
- Typical measured runtime per run: about `4.28s`
- Log artifact: `/home/thibaut.boissin/projects/muon_conv2/logs/c3345522-697c-4e94-a697-ae08de464ae0/log.pt`
- W&B run: `exp9-exp2-replica-2026-03-19` (`vwemskd2`)

**Observations**:
- This replicate is very close to the original Experiment 2 result (`0.9390` vs `0.9395` mean TTA val accuracy).
- The difference is much smaller than the run-to-run variation already observed in this setup, so Experiment 2 does not look like an outlier.
- The best mainline configuration is therefore reinforced rather than weakened by the rerun.

## Experiment 10: Extend Mainline to 9 Epochs

**Date**: 2026-03-20
**Hypothesis**: The best mainline result may still be slightly undertrained at 8 epochs, since validation accuracy was still rising sharply in the last logged epoch. Increasing the step budget to 9 epochs could be enough to push the mean TTA accuracy above 94%.
**Change**: Kept the Experiment 2 mainline unchanged and increased `--epochs` from `8` to `9`.

**Command**:
```bash
source scripts/env.sh 11.8 8.8.1 && python airbench94_conv_muon.py --run-name exp10-epochs9-2026-03-20 --epochs 9
```

**Results**:
- Mean TTA val accuracy: `0.9391`
- TTA val accuracy std: `0.0013`
- Mean val accuracy: `0.9316`
- Mean train accuracy: `0.9748`
- Mean train loss: `inf`
- Epochs: `9`
- Batch size: `2000`
- Measured runs: `5` plus `1` warmup
- Final per-run TTA val accuracies: `0.9379`, `0.9396`, `0.9374`, `0.9410`, `0.9395`
- Typical measured runtime per run: about `5.12s`
- Log artifact: `/home/thibaut.boissin/projects/muon_conv2/logs/e38ffcc0-4860-4ead-9723-66cb15467f86/log.pt`
- W&B run: `exp10-epochs9-2026-03-20` (`lx0jarg8`)

**Observations**:
- This was effectively neutral relative to the best 8-epoch run (`0.9391` vs `0.9395` mean TTA val accuracy), so the extra epoch did not solve the gap to 94%.
- Mean train accuracy increased sharply (`0.9748`), while mean validation accuracy improved only marginally, which is consistent with added fitting without a matching TTA gain.
- This weakens the idea that the best mainline is simply one epoch short; if step budget is revisited later, it should probably be coupled with another change rather than increased in isolation.
- The best known configuration remained the 8-epoch mainline from Experiment 2.

## Experiment 11: Slightly Higher Non-Muon Weight Decay

**Date**: 2026-03-20
**Hypothesis**: Since `adam_weight_decay_scale=2e-6` beat the `1e-6` baseline and the 9-epoch extension mainly increased fitting, a mild increase to `3e-6` might improve generalization without changing the Muon path.
**Change**: Kept the Experiment 2 mainline unchanged and increased `--adam-weight-decay-scale` from `2e-6` to `3e-6`.

**Command**:
```bash
source scripts/env.sh 11.8 8.8.1 && python airbench94_conv_muon.py --run-name exp11-adamwd3e-6-2026-03-20 --adam-weight-decay-scale 3e-6
```

**Results**:
- Mean TTA val accuracy: `0.9387`
- TTA val accuracy std: `0.0009`
- Mean val accuracy: `0.9301`
- Mean train accuracy: `0.9662`
- Mean train loss: `inf`
- Epochs: `8`
- Batch size: `2000`
- Measured runs: `5` plus `1` warmup
- Final per-run TTA val accuracies: `0.9400`, `0.9396`, `0.9379`, `0.9378`, `0.9380`
- Typical measured runtime per run: about `4.59s`
- Log artifact: `/home/thibaut.boissin/projects/muon_conv2/logs/5c16ea32-be01-487b-9bf8-ce06f089c1bb/log.pt`
- W&B run: `exp11-adamwd3e-6-2026-03-20` (`cmfioa00`)

**Observations**:
- This regressed on the main objective relative to Experiment 2 (`0.9387` vs `0.9395` mean TTA val accuracy).
- The run-to-run variance tightened, but around a worse mean, so the extra regularization improved consistency more than peak or average performance.
- Together with Experiment 10, this suggests the current 8-epoch mainline is sitting in a narrow sweet spot where both extra step budget and extra non-Muon regularization miss the target.
- The best known configuration remained Experiment 2.

## Experiment 12: Softer Late Orthogonalization

**Date**: 2026-03-20
**Hypothesis**: The flat `beta=0.5` Bjork schedule may be pushing the Muon convolution update too far from the raw gradient throughout all iterations. Lowering only the ending beta should preserve early conditioning while making the late-stage update less aggressive.
**Change**: Kept the Experiment 2 mainline unchanged except for `--orthogonalize-beta-end 0.25` while keeping `--orthogonalize-beta-init 0.5`.

**Command**:
```bash
source scripts/env.sh 11.8 8.8.1 && python airbench94_conv_muon.py --run-name exp12-betaend0.25-2026-03-20 --orthogonalize-beta-init 0.5 --orthogonalize-beta-end 0.25
```

**Results**:
- Mean TTA val accuracy: `0.9393`
- TTA val accuracy std: `0.0013`
- Mean val accuracy: `0.9305`
- Mean train accuracy: `0.9733`
- Mean train loss: `inf`
- Epochs: `8`
- Batch size: `2000`
- Measured runs: `5` plus `1` warmup
- Final per-run TTA val accuracies: `0.9400`, `0.9387`, `0.9412`, `0.9374`, `0.9390`
- Typical measured runtime per run: about `4.59s`
- Log artifact: `/home/thibaut.boissin/projects/muon_conv2/logs/32774eeb-2f7a-4861-ac70-4b37e2ab6cff/log.pt`
- W&B run: `exp12-betaend0.25-2026-03-20` (`ucz12qe6`)

**Observations**:
- This was competitive but still slightly below the Experiment 2 best (`0.9393` vs `0.9395` mean TTA val accuracy), so it is not a clear improvement.
- Mean train accuracy increased substantially to `0.9733`, which suggests the softer late orthogonalization made optimization easier but did not improve generalization.
- This weakens the specific H3 variant tested here: simply tapering the late beta is not enough to beat the current mainline.
- The best known configuration remains Experiment 2: 8 epochs, `adam_weight_decay_scale=2e-6`, Muon renormalization enabled, `muon_lr=0.52`, `muon_momentum=0.6`, `orthogonalize_beta_init=orthogonalize_beta_end=0.5`, and `aug_translate=2`.
