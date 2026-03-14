# Touch Classifier Training

This project now exports confirmed interaction windows to `Videos/classifier_dataset`.

## 1. Label the exported clips

Open each exported payload json and set:

- `"label": "valid_touch"` for real contact
- `"label": "no_touch"` for close-but-not-contact

Recommended first scope:

- focus on `CONFIRMED_RAIDER_DEFENDER_CONTACT`
- ignore HLI for the first model unless you explicitly want it

## 2. Create train/val splits

Create two text files in `Videos/classifier_dataset`:

- `train_clips.txt`
- `val_clips.txt`

Each line should contain one `clip_id` from `manifest.csv`.

## 3. Train the model

Run from the repo root:

```powershell
python Kabaddi_video_processing\train_touch_classifier.py --dataset-root Videos/classifier_dataset --output-dir Kabaddi_video_processing\models\touch_classifier
```

Optional flags:

```powershell
python Kabaddi_video_processing\train_touch_classifier.py --epochs 25 --batch-size 8 --num-frames 12
```

## 4. Output

Training writes:

- `Kabaddi_video_processing/models/touch_classifier/best_model.pt`
- `Kabaddi_video_processing/models/touch_classifier/training_history.json`

## 5. Inference

Use `touch_classifier_inference.py` to load the checkpoint and score a clip.

The first baseline model is:

- `ResNet18` frame encoder
- temporal average pooling
- binary output: `no_touch` / `valid_touch`
