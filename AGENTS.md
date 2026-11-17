# Co-DINO project rules

- Co-DINO is independent: do not import implementation code from sibling projects.
- Keep only the selected radial-angular Sparse-Communication-MoE architecture.
- T/S/B/L may differ only through `configs/model_sizes.json` base width/depth presets.
- Do not add ablation, baseline, representation-pretraining or mechanism-bypass switches.
- Full-model pretraining and joint encoder-decoder finetuning use Relative-L1.
- Preserve the four-channel Poseidon contract and the NS channel-0/channel-3 metric mask.
- Expert hidden width is always equal to its stage channel width.
