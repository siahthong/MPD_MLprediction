# Machine-learning Prediction of Macrostate Probability Distributions

Data and analysis code supporting the manuscript on predicting single-component adsorption macrostate probability distributions for CO₂, CH₄, and N₂ in metal–organic frameworks using PCA+MLP and DeepONet.

This package contains the exact final 2,095-structure inputs, original scientific scripts, selected fitted artifacts, test predictions, and numerical manuscript tables. Raw simulation trees, discarded runs, logs, temporary files, and the separate forensic audit are excluded.

## Data and array definitions

Each `data/GAS directory` preserves the final row order. `mof_ids.csv` contains MOF ID, fixed split, source row, and retained Nmax. `feature_values.npz` contains unscaled features and ordered feature_names. `mpd_values.npz` contains normalized reference lnpi and absolute integer n_axis. `mpd_masks.npz` contains valid_mask, true on the contiguous domain N=0..Nmax. splits.json contains fixed ID lists. `parent_groups_and_splits.csv` records parent-group assignments and the fixed split, containing 1,676 training, 210 validation, and 209 test structures. Structures belonging to the same parent group are assigned to the same split.

`reweighting_energy_GAS.npz` is a lossless aligned packaging of the original term_1 energy columns in kelvin units. Its arrays are mof_ids, energy_K, valid_mask, and source_relative_paths.

Verified definitions: Nmax_planned=floor(1.1*Nsat). The full-sampling correction applied to planned Nmax<4: 65 CH4 and 54 N2 systems used N=0,1,2,3,4. The ML Nmax feature and mask endpoint use final retained support.

## Execution order

Run commands from the package root. Install the dependencies first:

```bash
python -m pip install -r requirements.txt
```

1. Fixed grouped split provenance: python code/split_dataset.py --help
2. The released data are already assembled. `code/prepare_stage04_data.py` records historical feature-assembly provenance and requires original source tables; it is not a standalone package step.
3. Rank-30 fit: python code/fit_masked_low_rank.py --data data/CO2 --ranks 30 --smoothness 10 --ridge 1e-6 --max-iterations 3000 --tolerance 1e-6 --output _outputs/low_rank/CO2
4. PCA+MLP: python code/train_masked_pca_mlp.py --data data/CO2 --low-rank-model results/models/indirect/low_rank/CO2/masked_low_rank_model.npz --coefficients results/models/indirect/low_rank/CO2/latent_coefficients.csv --output _outputs/pca_mlp/CO2 --hidden 256 128 64 --dropout 0.1 --lr 0.001 --epochs 1000 --patience 200 --batch-size 64 --seed 42
5. DeepONet: python code/train_deeponet_absolute_n.py --data data/CO2 --output _outputs/deeponet/CO2 --epochs 1000 --patience 200 --batch-size 64 --lr 0.001 --p 64 --hidden-dim 128 --seed 42
6. Prediction is performed by the final sections of the training scripts after best-validation-checkpoint reload.
7. Final manuscript evaluation: `python code/evaluate_manuscript_298K.py`. This evaluates the six released test prediction arrays at 298 K on 100 logarithmically spaced pressures from 1e-5 to 100 bar and writes new outputs under `_outputs/evaluation_298K/`. It verifies TV and RMSRE against the released 298 K structure and parent-group tables.
8. Rank reconstruction evaluation remains at 273 K: `python code/evaluate_rank_selection_273K.py`. The package includes rank-30 fitted artifacts; therefore the default evaluates rank 30 on 1,886 training+validation structures. Other ranks require their corresponding historical fitted artifacts, supplied with `--fits-root` and `--ranks`. The released rank count table remains the historical authoritative table; the included CO2 rank-30 fit is the later 3,000-iteration fit, so a new reconstruction run is not assumed to reproduce the historical contingency counts.
9. Optional 273 K test evaluation: `python code/evaluate_test_tv_rmsre_273K.py`. Its output is supplementary to the manuscript's 298 K evaluation.
10. Checkpoint and data check: `python code/smoke_test.py` (requires PyTorch).

## Result conditions and verification

`test_structure_metrics.csv`, `test_parent_group_metrics.csv`, and the feature-ablation tables report the manuscript's 298 K evaluations. `stage04_test_metrics_by_mof.csv`, `stage04_test_summary.csv`, and `group_macro_test_summary.csv` are legacy 273 K analyses. `evaluate_group_macro_metrics.py` and `build_qualitative_comparison.py` are historical project-dependent analysis scripts, not standalone release commands.

The final evaluator uses mean molecule number per simulation supercell; relative isotherm error is unchanged by the same mol/kg conversion applied to reference and prediction. It does not reproduce absolute mol/kg uptake without the original mass and supercell metadata. At 298 K the energy and temperature correction terms cancel. The optional 273 K evaluation uses the released aligned energy arrays and preserves the original pressure-ratio and energy formula.

The model paths use `results/models/indirect/low_rank`, `results/models/indirect/pca_mlp`, and `results/models/direct/deeponet`. 
