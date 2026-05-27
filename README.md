# Korean Age Face Prediction

Korean face age prediction web project using the AI Hub facial aging image dataset.

## Project Overview

This project predicts an estimated age from a webcam face image without using an external API or LLM. The current web inference path uses the trained EfficientNet-B0 v3 multitask model.

## Current Model

- Model file: `data/aging_face/best_age_efficientnet_v3_multitask.pt`
- Backbone: EfficientNet-B0
- Task 1: exact age regression
- Task 2: age-group classification
- Input size: 224x224 face crop
- Validation MAE: about 4.07 years
- Adult MAE: about 5.45 years

The web app blends the regression age output with the age-group head output so that the prediction is based on the latest v3 training artifact.

## Main Files

- `index.html`: prediction page
- `style.css`: web page styling
- `app.js`: webcam capture and frontend prediction logic
- `server.py`: local web server and upload API
- `model_inference.py`: local PyTorch model inference logic
- `visualizations.html`: visualization page
- `terms.html`: glossary page
- `files.html`: file upload and download page
- `train_age_regression_efficientnet_v3.py`: v3 training script
- `docs/v3_training_summary.md`: v3 training result summary

## Run Locally

```powershell
cd C:\Users\svc06\OneDrive\Desktop\한국인_이미지_프로젝트
python server.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Notice

Original AI Hub dataset files, extracted images, user-uploaded files, and private webcam captures are not included because of privacy, license, and file size issues.
