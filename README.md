# Radar Occupancy Grid Learning

Implementation of "Road Scene Understanding by Occupancy Grid Learning from Sparse Radar Clusters".

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Dummy Training (Verification)
To verify the pipeline works without NuScenes data:
```bash
python train.py --dummy --epochs 5
```

### Full Training (NuScenes)
(Requires NuScenes dataset)
```bash
python train.py --dataroot /path/to/nuscenes
```

## Structure
- `dataset.py`: Data loading and aggregation (k=20 frames).
- `model.py`: U-Net architecture.
- `losses.py`: Lovasz-Softmax loss.
- `train.py`: Training loop.
