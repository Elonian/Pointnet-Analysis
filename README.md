PointNet Shape Analysis
=======================

This project contains the completed PointNet assignment code in the current
repository structure.

Project layout
--------------

```text
Pointnet-Shape-Analysis/
├── configs/
│   ├── environment.yml
│   ├── evaluation.yml
│   ├── training.yml
│   └── visualization.yml
├── data/
│   ├── modelnet40_ply_hdf5_2048/
│   └── shapenet_part_seg_hdf5_data/
├── execution_scripts/
│   ├── evaluate_pointnet.sh
│   ├── run_pointnet.sh
│   └── visualize_pointnet.sh
├── evaluation/
│   └── evaluation.py
├── logs/
├── models/
│   └── pointnet_models.py
├── modules/
│   ├── checkpoints.py
│   ├── datasets.py
│   ├── evaluators.py
│   ├── log_parsing.py
│   ├── plotting.py
│   ├── reporting.py
│   └── runtime.py
├── sub-modules/
│   └── pointnet/
│       ├── model.py
│       ├── train_cls.py
│       ├── train_seg.py
│       ├── dataloaders/
│       ├── utils/
│       └── data/ -> links to ../../data datasets
├── requirements.txt
├── notebook/
│   └── run_colab.ipynb
├── utils/
│   ├── common.py
│   └── logging_utils.py
├── visualization/
│   └── visualize.py
└── src/
    └── run_training.py
```

Completed tasks
---------------

The assignment TODOs are implemented in:

- `sub-modules/pointnet/model.py`
- `sub-modules/pointnet/train_cls.py`
- `sub-modules/pointnet/train_seg.py`

Implemented parts:

- PointNet global feature encoder with optional input and feature transforms
- ModelNet40 classification head and loss
- ShapeNet part segmentation head and loss
- Feature-transform orthogonality regularization
- CPU/GPU selection with `--gpu`; use `--gpu -1` for CPU

Install
-------

Pip install:

```bash
cd /mntdatalora/src/Pointnet-Shape-Analysis
python3 -m pip install -r requirements.txt
```

Conda environment:

```bash
cd /mntdatalora/src/Pointnet-Shape-Analysis
conda env create -f configs/environment.yml
conda activate pointnet-shape-analysis
```

Data
----

The required datasets have already been downloaded and extracted:

```text
/mntdatalora/src/Pointnet-Shape-Analysis/data/modelnet40_ply_hdf5_2048
/mntdatalora/src/Pointnet-Shape-Analysis/data/shapenet_part_seg_hdf5_data
```

The training code sees them through:

```text
/mntdatalora/src/Pointnet-Shape-Analysis/sub-modules/pointnet/data
```

Run both tasks
--------------

The shell script reads `configs/training.yml`, then runs classification and
segmentation through `src/run_training.py`. New run logs are written under
`logs/`.

```bash
cd /mntdatalora/src/Pointnet-Shape-Analysis
./execution_scripts/run_pointnet.sh
```

Optional runtime settings can be passed as command arguments:

```bash
./execution_scripts/run_pointnet.sh --epochs 20 --batch_size 128 --lr 0.001 --gpu 0 --amp
```

Use CPU:

```bash
./execution_scripts/run_pointnet.sh --gpu -1 --no-amp
```

Evaluate checkpoints
--------------------

Evaluation is YAML-driven through `configs/evaluation.yml`. By default it finds
the best saved classification and segmentation checkpoints, evaluates them on
the test splits, writes JSON/Markdown reports, and saves plots under
`analysis_outputs/evaluation`.

```bash
cd /mntdatalora/src/Pointnet-Shape-Analysis
./execution_scripts/evaluate_pointnet.sh
```

Override the runtime without editing YAML:

```bash
./execution_scripts/evaluate_pointnet.sh --task classification --batch_size 128 --gpu 0
./execution_scripts/evaluate_pointnet.sh --gpu -1 --no-plots
```

Visualize and analyze
---------------------

Visualization is YAML-driven through `configs/visualization.yml`. It can parse
training logs, summarize saved checkpoints, generate training curves, produce a
classification confusion matrix, plot segmentation class mIoU, and render
segmentation prediction samples.

```bash
cd /mntdatalora/src/Pointnet-Shape-Analysis
./execution_scripts/visualize_pointnet.sh
```

For fast log/checkpoint-only plots without running model evaluation:

```bash
./execution_scripts/visualize_pointnet.sh --skip-eval
```

Run tasks separately
--------------------

Classification:

```bash
cd /mntdatalora/src/Pointnet-Shape-Analysis/sub-modules/pointnet
python3 train_cls.py --epochs 20 --batch_size 128 --lr 0.001 --gpu 0 --amp
```

Segmentation:

```bash
cd /mntdatalora/src/Pointnet-Shape-Analysis/sub-modules/pointnet
python3 train_seg.py --epochs 20 --batch_size 128 --lr 0.001 --gpu 0 --amp
```

Colab
-----

Open `notebook/run_colab.ipynb`, mount Google Drive, change `PROJECT_DIR` to the
folder containing this project, install requirements, then run the two training
cells.
