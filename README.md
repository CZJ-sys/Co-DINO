# Co-DINO

Co-DINO is the independent forecasting implementation extracted from the
ECMENO research repository. It contains the selected radial-angular spectral
mixer, local depthwise convolution, and sparse selected-expert communication
with two shared experts, eight routed experts, and top-4 routing. Full-model
pretraining and joint finetuning use per-sample Relative-L1.

No sibling repository is imported at runtime. The existing `spectra_repro`
Conda environment may be used for dependencies. Install missing packages with
`python -m pip install -r requirements.txt`.

## Model variants

All four variants use stage depths `[2, 2, 2, 2]`. Only the base width changes.

| Variant | Base width | Stage widths | Parameters |
| --- | ---: | --- | ---: |
| T | 24 | 24/48/96/192 | 12,117,648 |
| S | 32 | 32/64/128/256 | 21,508,352 |
| B | 38 | 38/76/152/304 | 30,307,484 |
| L | 48 | 48/96/192/384 | 48,317,664 |

Parameter counts are scalar counts, independent of FP32/BF16 storage.

```bash
python -c "from model import *; [print(s, count_parameters(CoDINOForForecasting(get_model_config(s)))) for s in 'TSBL']"
```

## Environment and data

Run the following commands from the Co-DINO project root. The examples use the
original Poseidon `.nc` data directory. Replace the absolute data path with
your own directory if necessary. For one visible GPU, set
`CUDA_VISIBLE_DEVICES=0` in your shell before launching Python; the training
entrypoints also support `--device cpu --amp none` for small checks.

```bash
source /data1_hdd/chenzhengjie/miniconda3/etc/profile.d/conda.sh
conda activate spectra_repro
python -m pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0
```

Training uses `configs/pretrain.yaml` and `configs/finetune.yaml`. The
command-line options below override corresponding YAML values. `--seed`,
`--config`, `--data-path`, and `--output-dir` are required. The selected
variant is `B` (30.3M) in these examples; use `T`, `S`, or `L` as needed.

## Full-model pretraining

Train on all six datasets for 50 epochs:

```bash
python -u train/pretrain_full_model.py \
  --config configs/pretrain.yaml \
  --data-path /data1_hdd/chenzhengjie/Poseiden \
  --output-dir checkpoint/B_seed0/pretrain \
  --model-size B --seed 0 \
  --pretrain-datasets NS-Gauss NS-Sines CE-RP CE-CRP CE-Gauss CE-KH \
  --num_epochs 50 --lr 0.0001 --num_trajectories 200 \
  --batch_size 16 --num_workers 8 --device cuda
```

Resume from the saved model, optimizer, scheduler, and RNG state:

```bash
python -u train/pretrain_full_model.py \
  --config configs/pretrain.yaml \
  --data-path /data1_hdd/chenzhengjie/Poseiden \
  --output-dir checkpoint/B_seed0/pretrain \
  --model-size B --seed 0 \
  --resume checkpoint/B_seed0/pretrain/last.pt \
  --device cuda
```

On resume the checkpoint configuration is authoritative. Keep the data path
and output directory the same as in the original run.

## Joint finetuning

Finetune the complete pretrained model on one PDE for 150 epochs:

```bash
python -u train/finetune.py \
  --config configs/finetune.yaml \
  --data-path /data1_hdd/chenzhengjie/Poseiden \
  --output-dir checkpoint/B_seed0/finetune/CE-Gauss_joint \
  --model-size B --dataset CE-Gauss --seed 0 \
  --full-checkpoint checkpoint/B_seed0/pretrain/best.pt \
  --num_epochs 150 --lr 0.0001 --num_trajectories 200 \
  --batch_size 16 --num_workers 8 --device cuda
```

Resume the same finetuning run:

```bash
python -u train/finetune.py \
  --config configs/finetune.yaml \
  --data-path /data1_hdd/chenzhengjie/Poseiden \
  --output-dir checkpoint/B_seed0/finetune/CE-Gauss_joint \
  --model-size B --dataset CE-Gauss --seed 0 \
  --resume checkpoint/B_seed0/finetune/CE-Gauss_joint/last.pt \
  --device cuda
```

For all six targets, run the finetuning command separately with
`--dataset NS-Gauss`, `NS-Sines`, `CE-RP`, `CE-CRP`, `CE-Gauss`, and
`CE-KH`; give each run its own output directory. The same pretrained
`best.pt` can initialize all six runs.

## Evaluation

The default test sample count is 200 trajectories per dataset. On
Navier–Stokes data, channels 0 and 3 are excluded from error metrics.

```bash
python -u train/evaluate.py \
  --checkpoint checkpoint/B_seed0/finetune/CE-Gauss_joint/best.pt \
  --data-path /data1_hdd/chenzhengjie/Poseiden \
  --datasets CE-Gauss --split test --samples 200 \
  --first-target-time 1 --rollout-steps 20 \
  --evaluation-resolution 128 --metric-space normalized \
  --batch-size 8 --num-workers 4 --device cuda \
  --save-detailed-metrics --save-ground-truth-and-forecast 8 \
  --generate graph
```

The graph shows steps 1, 10, and 20 on a fixed 0–110% axis with each
bar's error annotated.

## Visual analysis

Raw-state directional spectra:

```bash
python -u train/analyze_directional_spectra.py \
  --data-path /data1_hdd/chenzhengjie/Poseiden \
  --datasets NS-Gauss NS-Sines CE-RP CE-CRP CE-Gauss CE-KH \
  --trajectories-per-dataset 10 --samples-per-trajectory 2 \
  --output-dir analysis/raw_state_directional
```

Routed-expert activations and heatmaps:

```bash
python -u train/analyze_experts.py \
  --checkpoint checkpoint/B_seed0/pretrain/best.pt \
  --data-path /data1_hdd/chenzhengjie/Poseiden \
  --datasets NS-Gauss NS-Sines CE-RP CE-CRP CE-Gauss CE-KH \
  --samples 200 --output-dir checkpoint/B_seed0/pretrain/expert_analysis \
  --device cuda
```

To regenerate the aggregate activation heatmap from its CSV:

```bash
python -u train/plot_expert_activation_heatmap.py \
  --input-csv checkpoint/B_seed0/pretrain/expert_analysis/expert_activation.csv \
  --output-dir checkpoint/B_seed0/pretrain/expert_analysis
```

Encoder first/last communication matrices:

```bash
python -u train/plot_encoder_communication_matrix.py \
  --checkpoint checkpoint/B_seed0/pretrain/best.pt \
  --output-dir checkpoint/B_seed0/pretrain/expert_analysis/communication_matrices
```

For an existing detailed evaluation report, regenerate the rollout chart:

```bash
python -u train/plot_rollout_errors.py \
  --input checkpoint/B_seed0/finetune/CE-Gauss_joint/evaluation/eval_CE-Gauss_test_n200_t1_t20_128_normalized_detailed.json
```

Run `python -m unittest discover -s tests -v` for local checks. See
[docs/architecture.md](docs/architecture.md) for tensor shapes and operators.
