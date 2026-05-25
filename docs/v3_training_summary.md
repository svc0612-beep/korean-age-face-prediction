# V3 EfficientNet Training Summary

## Purpose

V3 training was started because the previous MobileNetV3 regression model produced unrealistically young predictions on webcam images. The new training approach keeps exact age prediction, but also adds age-group classification so the model learns broader age bands at the same time.

## Model

- Backbone: EfficientNet-B0
- Task 1: age regression
- Task 2: age-group classification
- Input: cached 224x224 face images
- Loss: SmoothL1 regression loss + weighted age-group classification loss
- Extra focus: adult and older-age samples receive stronger training weight

## Training Stop Point

Training was stopped manually on 2026-05-26 during epoch 13. The best checkpoint saved before stopping was from epoch 12.

## Best Saved Result

- Best epoch: 12
- Validation MAE: 4.0719 years
- Adult MAE: 5.4462 years
- Best score: 4.5529
- Validation group accuracy: 0.640

## Age-Group Validation Summary At Best Epoch

| True group | Count | True mean | Pred mean | MAE | Bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0s | 999 | 4.93 | 6.73 | 2.30 | +1.81 |
| 10s | 1438 | 15.14 | 17.78 | 3.79 | +2.64 |
| 20s | 1264 | 23.77 | 25.84 | 4.33 | +2.07 |
| 30s | 624 | 34.11 | 33.92 | 4.48 | -0.19 |
| 40s | 486 | 43.62 | 39.80 | 5.06 | -3.82 |
| 50s | 189 | 54.20 | 47.48 | 7.07 | -6.71 |
| 60plus | 50 | 64.28 | 49.19 | 15.09 | -15.09 |

## Notes

- V3 greatly reduced the problem where 30s faces were predicted as late teens or early 20s in validation-style evaluation.
- The 40s, 50s, and 60plus groups are still under-predicted.
- The next step is to update web inference to load `best_age_efficientnet_v3_multitask.pt`.
- For real webcam accuracy, additional webcam-domain fine-tuning data is still recommended.

## Saved Artifacts

- `data/aging_face/best_age_efficientnet_v3_multitask.pt`
- `data/aging_face/age_efficientnet_v3_history.csv`
- `data/aging_face/age_efficientnet_v3_val_predictions.csv`
