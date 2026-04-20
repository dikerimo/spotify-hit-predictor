# Hit Detector

Hit Detector is a local ML project for estimating the `hit-likeness` of a song. I built it both as an AI portfolio project and as a practical support tool for my own music workflow as a cover artist on Spotify. My main use case is to analyze existing songs and decide whether they look like strong candidates to cover, record, and release.

This kind of decision support matters in a real release workflow: my artist account currently averages around 400k monthly listeners and over 2 million monthly streams, so choosing what to cover is not just a hobby decision but part of an active music business. This tool is one of the signals I can use in that process.

The current version trains on the Kaggle dataset `maharshipandya/-spotify-tracks-dataset`, which already contains Spotify-style audio features and popularity labels, then extracts local audio features from a provided song file and scores it with a trained model.

## Overview

The project is designed as a decision-support tool built around a practical question:

`Does this song look like a strong cover candidate, based on how similar its audio profile is to tracks that historically performed well?`

I originally approached this from the cover-song angle: if I am choosing between several songs to cover, can audio-based modeling help identify which ones look more promising?

The same idea can also be useful for less niche purposes, the main beeing screening songs before spending time on production (so just in general trying to determine whether an original song can potentially bring in streams)

This detector should be treated as an assistant, not as a fully deterministic tool. In my case, it helps answer `should I consider covering this song?`, not `how popular will my final release definitely become?` Real performance still depends on factors like audience, marketing, timing, platform exposure, visuals, branding, and the quality of the final cover, not just the audio features themselves.

## Original Workflow Idea

The original plan was to build the dataset directly from Spotify's API, but Spotify's current API restrictions (since April) blocked key parts of the workflow.

## Current Workflow

Instead of collecting training data live from Spotify every time, the project now:

1. downloads the Kaggle dataset `maharshipandya/-spotify-tracks-dataset`, which already contains Spotify-style feature columns
2. trains models locally on the CSV data
3. extracts local proxy audio features from your uploaded song
4. scores the song with the trained model

A different dataset was tested during development, but `maharshipandya/-spotify-tracks-dataset` became the main training source because it produced more usable class balance, better model behavior and the data is general was collected later, which makes it more valuable for determining something as fast-changing as music trends.

The dataset itself is not committed to this repository. To run the project, download it locally through the Kaggle API and keep it in your local `data` folder.

## What This Project Does

1. Download a Spotify-style dataset from Kaggle.
2. Train a classifier that learns a `hit` vs `non-hit` label from a chosen popularity threshold.
3. Compare multiple models.
4. Tune the strongest models.
5. Extract local proxy audio features from your uploaded song.
6. Predict a `hit-likeness` score for the uploaded song.

## Repository Structure

- `audio_features.py`
  Extracts local audio features from a song file using `librosa`.

- `train_model.py`
  Trains the current default Random Forest model from a CSV dataset.

- `compare_models.py`
  Compares Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost on the same dataset split.

- `tune_models.py`
  Searches a small set of parameter combinations for Random Forest and XGBoost.

- `predict_cover.py`
  Loads a trained model, extracts features from an audio file, and outputs a prediction JSON.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Kaggle API Setup

Put your Kaggle credentials in `.env`:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_TOKEN=your_kaggle_api_token
```

Load them into the current PowerShell session:

```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
```

## Download The Dataset

Primary dataset used in this project:

```powershell
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset
Expand-Archive -Path .\-spotify-tracks-dataset.zip -DestinationPath .\data2 -Force
Get-ChildItem .\data2
```

## Train A Model

Train the current default Random Forest model:

```powershell
python train_model.py --data .\data2\dataset.csv --output models\hit_model_data2_65.joblib --threshold 65
```

What `threshold 65` means:

- `1` = popularity >= 65
- `0` = popularity < 65

The current default training configuration in `train_model.py` uses the best Random Forest found during the focused second-stage tuning pass:

- `n_estimators=300`
- `max_depth=None`
- `min_samples_leaf=6`
- `min_samples_split=6`
- `max_features="sqrt"`

## Compare Models

Compare several models on the same dataset and threshold:

```powershell
python compare_models.py --data .\data2\dataset.csv --threshold 65 --output models\compare_data2_65.json
```

Current comparison set:

- Logistic Regression
- Random Forest
- HistGradientBoosting
- XGBoost

## Tune Models

Tune Random Forest and XGBoost with a small parameter search:

```powershell
python tune_models.py --data .\data2\dataset.csv --threshold 65 --output models\tuning_data2_65.json
```

## Predict On A Song

After training a model, score an audio file:

```powershell
python predict_cover.py --audio path\to\your_song.wav --model models\hit_model_data2_65.joblib
```

The prediction script writes a JSON result under `predictions/`.

## Evaluation Metrics

The most important metrics for this project are:

- `hit_precision`
  When the model predicts `hit`, how often it is right.

- `hit_recall`
  Of all real hits, how many the model catches.

- `hit_f1`
  The best single summary of hit-class quality because it balances precision and recall.

- `roc_auc`
  How well the model separates hits from non-hits overall.

For this project, `hit_f1` and `roc_auc` matter more than raw accuracy.

### Baseline Comparison Results

Summary of the first comparison pass:

- Logistic Regression was a useful baseline, but weaker overall: `roc_auc = 0.6794`, `hit_f1 = 0.2130`
- Random Forest gave the best practical balance and became the main tuning candidate: `roc_auc = 0.7702`, `hit_f1 = 0.2580`
- HistGradientBoosting and XGBoost achieved stronger `roc_auc`, but predicted too few hits at the default `0.5` cutoff, so they were less useful for the main task

## Best Configuration

After a broader tuning pass and a focused second-stage Random Forest search, the final selected configuration was:

- `n_estimators=300`
- `max_depth=None`
- `min_samples_leaf=6`
- `min_samples_split=6`
- `max_features="sqrt"`

Best tuned Random Forest metrics after analysis:

- `roc_auc = 0.8999`
- `hit_precision = 0.7385`
- `hit_recall = 0.5931`
- `hit_f1 = 0.6579`

The committed summary of the selected model is stored in `results/final_model_summary.json`.

XGBoost showed some promise during comparison, but under the current setup it still predicted too few hits to beat the tuned Random Forest for the actual use case.

## Limitations

This project predicts `hit-likeness`, not guaranteed popularity.

Why:

- Spotify popularity is influenced by marketing, audience size, timing, and platform effects
- local feature extraction only approximates Spotify's original feature definitions
- real-world performance depends on factors outside the audio itself

That is acceptable for an MVP. A stronger future version would add my own release outcomes as labels.

## Dataset Limitations And Future Work

The current training pipeline uses a public Kaggle dataset that I did not create myself. That makes it useful for prototyping, but it also creates some clear limitations.

The most important limitations are:

- the dataset is fixed at the time it was published, so it does not reflect newer songs released after that point
- this matters especially for newer viral or TikTok-driven songs, which are relevant for cover selection and would be useful to have in the training data
- the labels and feature definitions come from a third-party dataset rather than from a dataset built directly around my own workflow

In future work, I would like to build a more custom dataset for this project. That could include:

- newer songs that are more relevant to current cover trends
- tracks chosen specifically for cover-selection decisions rather than generic popularity modeling
- additional metadata or business-side signals that may help the model beyond audio alone
- eventually, my own release outcomes as more task-specific labels
