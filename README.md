# Fashion-MNIST ANN Pipeline

End-to-end ML versioning project using Git, DVC and TensorFlow.
## Project structure

| Path | Purpose |
|------|---------|
| `src/prepare.py` | Downloads Fashion-MNIST and saves raw arrays to `data/raw/` |
| `src/preprocess.py` | Normalizes images, splits train/val and saves to `data/processed/` |
| `src/train.py` | Builds and trains the ANN, saves `models/model.h5` and `models/history.csv` |
| `src/evaluate.py` | Evaluates the model, writes `metrics.json` and `confusion_matrix.png` |
| `params.yaml` | All hyperparameters |
| `dvc.yaml` | Pipeline stages |

## Reproduce

```bash
pip install -r requirements.txt
dvc pull
dvc repro
```