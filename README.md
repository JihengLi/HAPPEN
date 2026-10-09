# HAPPEN: Human-in-the-Loop Auditing Pipeline for Exact and Near Duplicates in MRI Repositories

HAPPEN audits exact and near duplicates in T1-weighted brain MRI repositories and produces reviewer-confirmed subject groups.

[Paper (arXiv preprint)](https://arxiv.org/abs/2610.09614) · [Container release](https://doi.org/10.5281/zenodo.21986366) · [License](LICENSE.pdf)

## Access the container image

HAPPEN is distributed through [Zenodo](https://doi.org/10.5281/zenodo.21986366) as a prebuilt Singularity/Apptainer image with dependencies and pretrained models. For download, configuration, and execution instructions, see [zenodo/README.md](zenodo/README.md).

## Run the current source

**Requirements:** Linux x86_64, Git, and Conda. GPU execution requires an NVIDIA GPU with a CUDA 12.8-compatible driver. Read [LICENSE.pdf](LICENSE.pdf) before use.

### 1. Get the code and install dependencies

```bash
git clone https://github.com/MASILab/HAPPEN.git
cd HAPPEN
conda create -n happen-source --override-channels -c conda-forge \
  python=3.11 ants=2.6.2 pip -y
conda activate happen-source
python -m pip install uv
uv sync --frozen
command -v N4BiasFieldCorrection antsRegistrationSyN.sh antsRegistration antsApplyTransforms
```

Run the remaining commands from the HAPPEN directory with `happen-source` active.

### 2. Prepare the pretrained model

Place the pretrained HAPPEN checkpoint at `resources/model.pth`. It is not included in Git. If you have the [released SIF](https://doi.org/10.5281/zenodo.21986366), you can extract its checkpoint with Apptainer or Singularity:

```bash
apptainer exec --bind "$PWD/resources:/host-resources" \
  /absolute/path/to/happen-v0.1.3-cu128.sif \
  cp /app/resources/model.pth /host-resources/model.pth
```

Replace `apptainer` with `singularity` if needed. The atlas is already in `resources/`.

### 3. Set the input and output

Copy the configuration template, then set an output directory and either a BIDS directory or scan CSV as described in [Configuration](#configuration):

```bash
cp configs/config_usr.toml my_config.toml
```

### 4. Run the pipeline

```bash
uv run --frozen python -m happen.pipeline my_config.toml
```

### 5. Review and generate confirmed groups

```bash
uv run --frozen python -m happen.review my_config.toml
```

Open **http://127.0.0.1:8000**, review candidate pairs, and save your decisions. Stop the server with `Ctrl+C`, then run:

```bash
uv run --frozen python -m happen.review my_config.toml finalize
```

> [!IMPORTANT]
> Use the same configuration and output directory for pipeline, review, and finalize.

## Configuration

User configuration: [configs/config_usr.toml](configs/config_usr.toml).

```toml
  [run]
  out = ""
  stages = ["exact", "near"]

  profile_runtime = true
  modality = "T1w"

  [exact]
  root = ""
  candidates_csv = ""
  valid_csv = ""
  thread_workers = 32
  process_workers = 16

  [near]
  preprocess_workers = 16
  device = "auto"
  min_similarity = 0.92
  review_mode = "lazy" # off | lazy | precompute
  review_workers = 16

  batch_size = 16
  use_amp = true
  overwrite_embeddings = false
  fail_fast = false
  topk = 500
  retrieval_batch_size = 2048
  use_ivf = true
  nlist = 4096
  nprobe = 64
  train_size = 200000
  seed = 0
  overwrite_retrieval = false
  overwrite_review_assets = false
  max_scan_candidates_per_query = 100

  [review]
  port = 8000

  autosave_every = 1
  host = "0.0.0.0"
  preview_max_width = 1200
  debug = false
```

### Important configuration guide

#### 1. Output directory

- `[run].out`: output directory for all generated artifacts, reports, logs, and review files.

#### 2. Pipeline stages

- `[run].stages`: stages to execute. In most cases, use `["exact", "near"]`.

#### 3. Exact-stage input

> [!IMPORTANT]
> Set only **one** of the following stage input properties.

- `[exact].root`: root directory of a BIDS-style filesystem. When `root` is provided, HAPPEN automatically searches the repository and discovers all eligible T1-weighted scans.
- `[exact].candidates_csv`: path to a CSV with columns `dataset,subject_id,session_id,candidate`, one scan per row. `candidate` accepts absolute or relative NIfTI paths and file symlinks. HAPPEN checks the input paths and records their resolved targets in `valid.csv`.
- `[exact].valid_csv`: path to a validated CSV with the columns `dataset,subject_id,session_id,candidate,resolved_path`. This option is intended for cases where validation and path resolution have already been completed. For an exact+near run, the CSV must also be available at `[run].out/exact/valid.csv`.

```csv
dataset,subject_id,session_id,candidate
DatasetA,sub-001,ses-01,data/scan001.nii.gz
```

`dataset` and `subject_id` must be provided; `session_id` may be empty. Relative paths use the launch directory for source commands, or the configuration directory for `run_happen.sh`. Omit leading `./` and repeated `/` in CSV paths to preserve metadata.

The exact stage also has two important runtime parameters:

- `[exact].thread_workers`: number of thread-based workers used by the exact stage.
- `[exact].process_workers`: number of process-based workers used by the exact stage.

#### 4. Near-stage runtime settings

- `preprocess_workers`: number of workers used during preprocessing. In our reference runs, we used preprocess_workers = 32 on a system with 48 GB GPU memory.
- `device`: compute device used by the HAPPEN embedding model.
- `min_similarity`: similarity threshold for retaining near-duplicate candidates. Default is 0.92.

#### 5. Review-asset preparation

- `review_mode`: controls how review PNGs are generated.
  - `"off"`: do not prepare review assets
  - `"lazy"`: generate PNGs only for near-duplicate scans; assets (difference and checkerboard images) are generated on demand from the frontend workflow
  - `"precompute"`: generate all review PNGs in advance on the backend before review begins

- `review_workers`: number of workers used for review-asset generation.

#### 6. Review server settings

- `port`: port used by the review server

### Other parameters

#### `[run]`

- `profile_runtime`: if `true`, HAPPEN records runtime profiling information.
- `modality`: MRI modality to process. Default is `"T1w"`.

#### `[near]`

- `batch_size`: number of preprocessed scans encoded together in each inference batch.
- `use_amp`: if `true`, use automatic mixed precision when supported.
- `fail_fast`: if `true`, stop the embedding processing loop after preprocessing or tensor-loading errors.
- `topk`: number of nearest neighbors retrieved per query using FAISS before thresholding.
- `retrieval_batch_size`: batch size used during FAISS retrieval.
- `use_ivf`: whether to use IVF-based approximate nearest-neighbor retrieval.
- `nlist`: number of IVF coarse clusters.
- `nprobe`: number of IVF clusters probed during search.
- `train_size`: number of samples used to train the IVF index.
- `seed`: random seed used for retrieval-related steps.
- `overwrite_embeddings`: if `true`, recompute embeddings even if cached outputs already exist.
- `overwrite_retrieval`: if `true`, recompute retrieval outputs even if cached outputs already exist.
- `overwrite_review_assets`: if `true`, regenerate review assets even if they already exist.
- `max_scan_candidates_per_query`: maximum number of candidate scans per query in the review table.

#### `[review]`

- `autosave_every`: save review decisions after this many updates.
- `host`: host address used by the review server.
- `preview_max_width`: downsamples preview images to reduce browser memory usage.
- `debug`: enables debug mode if set to `true`.

## Results

All outputs are written under `[run].out`. The two main output folders are:

- `exact/`
- `near/`

#### Most important outputs

- `exact/duplicates.csv`: Exact deduplication result.
- `exact/figures/files_matrix_T1w.html`: Clear visualization of the exact duplicates.
- `near/scan_candidates.csv`: Near-duplicate candidates for human review.
- `near/review/review_decisions.csv` (after review): Human-in-the-loop auditing result.
- `near/subject_groups.csv` (after finalize): Confirmed near-duplicate subject groups.
- `near/figures/files_matrix_T1w.html` (after finalize): Clear visualization of the confirmed near duplicates.

### Complete list of outputs

#### `exact/`

- `candidates.csv`: candidate scans that searched from `[exact].root` or from input.
- `valid.csv`: all validated T1w MRI scans that passed input checking or from input.
- `invalid_missing_paths_discovery.csv`: scans removed because the input path could not be found.
- `invalid_not_nifti_discovery.csv`: scans removed because the resolved target lacks a NIfTI extension.
- `invalid_permission_denied_paths_discovery.csv`: scans removed because the file could not be accessed due to permission errors.
- `invalid_unreadable_discovery.csv`: reserved readability-check output; currently empty. Image-loading failures are recorded in `hash_errors.csv`.
- `invalid_derivatives.csv`: scans removed because they were recognized as derivatives rather than source scans.
- `T1w_hashes.csv`: SHA-256 voxel-array hashes for successfully hashed scans.
- `hash_errors.csv`: scans that failed during hashing.
- `duplicates.csv`: the main exact-duplicate table; all scans involved in exact duplicates, grouped by duplicate group.

- `by_category/`: exact duplicates reorganized by duplicate type.
  - `within_sessions.csv`: duplicates within the same session.
  - `within_subjects.csv`: duplicates across sessions within the same recorded subject.
  - `within_datasets.csv`: duplicates across different recorded subjects within the same dataset.
  - `across_datasets.csv`: duplicates across different datasets.

- `by_category_stats/`: summary statistics for the four duplicate categories above.

- `by_dataset/`: exact-duplicate outputs reorganized by dataset.

- `figures/files_matrix_T1w.html`: recommended summary view; an interactive lower-triangular matrix showing exact duplicates between dataset pairs. Open in a browser.

#### `near/`

- `no_exact_dup.csv`: near-stage input, retaining one representative per exact-duplicate group and excluding hash failures.
- `exact_removed.csv`: scans excluded from the near stage as redundant exact duplicates or because hashing failed.
- `embedding_manifest.csv`: manifest of scans used for embedding inference.
- `embeddings.npy`: computed embeddings for the scans in `embedding_manifest.csv`.
- `embedding_failures.csv`: scans that failed during preprocessing or embedding inference.
- `raw_neighbors.csv`: raw FAISS retrieval results; for each scan, the top retrieved similar scans before final filtering.
- `scan_candidates.csv`: the main near-duplicate output; final scan-level near-duplicate candidates after filtering.
- `subject_edges.csv` (after finalize): subject-level links derived from positive human review decisions.
- `subject_groups.csv` (after finalize): subject-level connected groups derived from those confirmed links.

- `review/`: outputs used by the review tool.
  - `review_candidates.csv`: candidate pairs presented to the review interface.
  - `review_decisions.csv`: saved human review decisions.
  - `scan_asset_failures.csv`: currently an empty compatibility table; scan-asset errors stop preparation.
  - `pair_asset_failures.csv`: currently an empty compatibility table; pair-asset errors stop preparation.
  - `assets/png/`: PNG renderings of scans involved in review.
  - `assets/diff/`: difference images for near-duplicate scan pairs (`precompute` mode).
  - `assets/checkerboard/`: checkerboard comparison images for near-duplicate scan pairs (`precompute` mode).

- `by_category/`: finalized near-duplicate outputs reorganized by duplicate type after human review.
  - `within_datasets.csv`: confirmed near duplicates involving different recorded subjects within the same dataset.
  - `across_datasets.csv`: confirmed near duplicates spanning different datasets.

- `by_category_stats/`: summary statistics for the two finalized near-duplicate categories above.

- `by_dataset/`: finalized near-duplicate outputs reorganized by dataset. Each file contains all confirmed near-duplicate subjects that form duplicate groups within that dataset scope.

- `figures/`: subject-level summary figures generated from the finalized confirmed near-duplicate subject groups.

## Review interface

HAPPEN also provides a built-in review interface for human inspection of near-duplicate candidates.

The review interface is **scan-level**. For each query scan, the interface displays all candidate scans that satisfy the filtering criteria. For each scan pair, HAPPEN reports the corresponding:

- dataset
- subject
- session
- similarity score

The reviewer then decides whether the pair is a true near duplicate.

The interface contains three views:

### 1. Dataset view (Entrypoint of the app)

<img src="docs/images/review_dataset_view.png" alt="Dataset view" width="700">

### 2. Subject view

<img src="docs/images/review_subject_view.png" alt="Subject view" width="700">

### 3. Review view

<img src="docs/images/review_review_view.png" alt="Review view" width="700">

### Keyboard navigation

Use the arrow keys to change scans in `3. Review view`:

- Left / Right arrow keys: switch between candidate scans for the same query scan
- Up / Down arrow keys: switch between query scans or subjects within the same dataset

## Citation

If you use HAPPEN in research, please cite the [arXiv preprint](https://arxiv.org/abs/2610.09614) for the methodology and the [HAPPEN software release](https://doi.org/10.5281/zenodo.21986366) for the version used. `CITATION.cff` contains both records; GitHub's **Cite this repository** shows the paper as the preferred citation.

```bibtex
@article{li2026happen,
  title = {Identity-Duplication Auditing in National-Scale Neuroimaging Repositories},
  author = {Li, Jiheng and
            Kim, Michael E. and
            Schwartz, Trent M. and
            Cui, Yuhan and
            Rudravaram, Gaurav and
            Archer, Derek B. and
            Hohman, Timothy J. and
            Beason-Held, Lori L. and
            Morgan, Victoria L. and
            Englot, Dario J. and
            Jefferson, Angela L. and
            {for the Alzheimer's Disease Neuroimaging Initiative} and
            {for the BIOCARD Study team} and
            {for the Health and Aging Brain Study: Health Disparities (HABS-HD) Study Team} and
            Zuo, Lianrui and
            Erus, Guray and
            Davatzikos, Christos and
            Landman, Bennett A.},
  journal = {arXiv preprint arXiv:2610.09614},
  year = {2026},
  doi = {10.48550/arXiv.2610.09614},
  url = {https://arxiv.org/abs/2610.09614}
}
```

## License

HAPPEN is provided under Vanderbilt University's [Non-Exclusive Non-Commercial Academic Software License Agreement](LICENSE.pdf). Copyright © 2026 Vanderbilt University. Read the complete agreement before accessing, downloading, or using HAPPEN. It is available only to non-profit academic and/or research institutions for internal non-commercial research purposes. Diagnostic or treatment use and redistribution by licensees are not permitted. For commercial licensing, contact cttc@vanderbilt.edu.

## Third-party licenses

Third-party licenses for redistributed resources and dependencies are provided in `resources/licenses/`.
