# 데이터 품질 및 모델 진단 리포트

## raw_train
- rows: 40150
- duplicate_rows: 0
- age_summary: `{'min': 0.0, 'q1': 11.0, 'median': 19.0, 'mean': 22.184433374844335, 'q3': 31.0, 'max': 82.0, 'std': 14.601689184073928}`
- age_invalid_count: 0
- age_group_counts: `{'0대': 8574, '10대': 12252, '20대': 8228, '30대': 5359, '40대': 2805, '50대': 2688, '60대이상': 244}`
- gender_counts: `{'female': 22450, 'male': 17700}`
- image_path: exists=40150, missing=0
- json_path: exists=40150, missing=0

## raw_val
- rows: 5050
- duplicate_rows: 0
- age_summary: `{'min': 0.0, 'q1': 12.0, 'median': 20.0, 'mean': 22.313465346534652, 'q3': 31.0, 'max': 76.0, 'std': 14.11620830708413}`
- age_invalid_count: 0
- age_group_counts: `{'0대': 999, '10대': 1438, '20대': 1264, '30대': 624, '40대': 486, '50대': 189, '60대이상': 50}`
- gender_counts: `{'female': 2750, 'male': 2300}`
- image_path: exists=5050, missing=0
- json_path: exists=5050, missing=0

## cache_train
- rows: 13546
- duplicate_rows: 0
- age_summary: `{'min': 0.0, 'q1': 9.0, 'median': 18.0, 'mean': 20.781485309316402, 'q3': 29.0, 'max': 75.0, 'std': 14.508976095495203}`
- age_invalid_count: 0
- age_group_counts: `{'0대': 3398, '10대': 4276, '20대': 2552, '30대': 1643, '40대': 748, '50대': 832, '60대이상': 97}`
- gender_counts: `{'female': 7846, 'male': 5700}`
- image_path: exists=13546, missing=0
- json_path: exists=13546, missing=0
- cache_path: exists=13546, missing=0

## cache_val
- rows: 5050
- duplicate_rows: 0
- age_summary: `{'min': 0.0, 'q1': 12.0, 'median': 20.0, 'mean': 22.313465346534652, 'q3': 31.0, 'max': 76.0, 'std': 14.11620830708413}`
- age_invalid_count: 0
- age_group_counts: `{'0대': 999, '10대': 1438, '20대': 1264, '30대': 624, '40대': 486, '50대': 189, '60대이상': 50}`
- gender_counts: `{'female': 2750, 'male': 2300}`
- image_path: exists=5050, missing=0
- json_path: exists=5050, missing=0
- cache_path: exists=5050, missing=0

## 학습 히스토리
- best_epoch: 54
- best_val_mae: 4.647110369842832
- best_train_mae: 1.8786535185618607
- generalization_gap: 2.7684568512809715
- overfit_signal: True

## 권장 조치
- age_past를 회귀 정답으로 유지한다.
- 0세와 70세 이상은 이상치가 아니라 실제 라벨일 수 있으므로 제거하지 않고 구간별 MAE로 감시한다.
- 과적합 신호가 보이면 dropout, weight decay, augmentation, early stopping, 더 많은 train cache를 사용한다.
- 웹앱 배포 전에는 validation 전체에 대한 구간별 MAE와 예측 샘플을 확인한다.