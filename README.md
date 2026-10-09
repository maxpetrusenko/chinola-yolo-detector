# chinola-yolo-detector

A YOLO11n detector for passion fruit (chinola, maracuya), plus the pipeline used to build it: Gemini pre-labeling, human review, YOLO11n training, and inference on video frames. It was built as an unpaid project for a passion-fruit farm in the Dominican Republic, as a first step toward counting fruit and spotting problems from phone and drone footage.

This repository holds code, one small trained checkpoint, and the training log. It does not contain any images, video, or farm data.

## Problem

Farm staff walk rows to check fruit by eye. The goal was an automatic fruit detector that could run on frames from phone or drone video, as a base for counting and later for flower and disease detection. There was no labeled passion-fruit detection data to start from, so the labeling pipeline is a large part of the project.

## Pipeline

1. Collect images and frames. Public passion-fruit images from a Kaggle classification dataset (`roynjuguna/passion-fruit-disease-detection`, a balanced sample of healthy, brown spot and woodiness classes) and 1 fps frames from video (`scripts/extract_video_frames.py`).
2. Pre-label with Gemini. `scripts/gemini_annotate_frames.py` asks a Gemini vision model for fruit bounding boxes with confidences and writes a CSV.
3. Human review. `scripts/annotation_server.py` is a small local UI to accept, delete or add boxes. In the model reported below, every Gemini draft box was reviewed and accepted.
4. Build the YOLO dataset. `scripts/build_yolo_from_gemini.py` converts the reviewed CSV into a one-class YOLO dataset with a train/val split.
5. Train YOLO11n. `scripts/train_yolo.py` wraps Ultralytics training.
6. Video inference. `scripts/predict_video_frames.py` runs the detector over extracted frames and exports a detections CSV and annotated frames.

## Dataset (reported model)

| Item | Value |
|---|---|
| Source images | 75 sampled from the public Kaggle set (25 healthy, 25 brown spot, 25 woodiness); 74 had at least one fruit box |
| Classes | 1 (`passion_fruit`) |
| Train | 60 images, 128 boxes |
| Validation | 14 images, 29 boxes |
| Labels | Gemini draft boxes, all accepted after human review |

The images are not redistributed here. Download them from Kaggle with `scripts/download_kaggle_passionfruit.py` (needs the Kaggle CLI and your own credentials) and check the dataset's license before reuse.

## Metrics

Validation of `models/passion-fruit-yolo11n.pt` on the 14-image validation split, re-run on 2026-10-09 to confirm (see `reports/val_metrics.txt`):

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.946 | 0.759 | 0.825 | 0.682 |

Training setup: `yolo11n.pt` start, 30 epochs, image size 640, batch 4, CPU. The per-epoch log is `models/passion-fruit-yolo11n-train-log.csv`. During training the best epoch (16) logged precision 0.922 and mAP50 0.874, and the last epoch logged mAP50 0.841; the table above is the separate final validation of the saved checkpoint. With only 29 validation boxes, differences of a few points between these numbers are noise.

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# frames from a video at 1 fps
python scripts/extract_video_frames.py --source path/to/videos --out data/frames --fps 1

# detect fruit
python scripts/predict_video_frames.py --model models/passion-fruit-yolo11n.pt \
  --source data/frames --out reports/detections.csv --conf 0.25

# pre-label with Gemini (set GEMINI_API_KEY first), review, then build and train
python scripts/gemini_annotate_frames.py --source data/frames --out reports/gemini-detections.csv --limit 100
python scripts/annotation_server.py --draft data/processed/draft --reviewed data/processed/reviewed
python scripts/build_yolo_from_gemini.py --detections reports/gemini-detections.csv --images data/frames
python scripts/train_yolo.py --data data/data.yaml --model yolo11n.pt --epochs 30 --imgsz 640 --batch 4
```

Run scripts from the repository root. Check each script's `--help` for the exact arguments; the annotation server writes reviewed YOLO labels that you then point `data/data.yaml` at.

## Limitations

- Very small validation set: 14 images, 29 boxes. The metrics show the pipeline works, not that the detector generalizes.
- The checkpoint was trained on clean close-up fruit photos. On real video frames it produced false positives on people, hands, buildings, text, leaves and other round objects. It is a draft pre-labeler, not a counting model.
- Later versions added reviewed video frames and hard negatives and improved recall at the cost of more false positives. Those datasets include private farm footage, so those models and their numbers are not published here.
- One class only. Flower and disease detectors were prototyped but are not good enough to publish: the flower model rests on 30 boxes and the disease labels are image-level, not fruit-level truth.
- Gemini labels reduce annotation time but inherit its mistakes. Accepting every draft box in the first model was a shortcut and a source of label noise.
- No tracking, so per-frame box counts must not be summed into a fruit count. Far-away drone imagery would need a different pipeline (orthomosaic and row geometry), not this close-range detector.

## License

MIT. See `LICENSE`. Ultralytics YOLO is AGPL-3.0; check its terms if you build on this code commercially.
