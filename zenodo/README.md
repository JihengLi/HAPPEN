# HAPPEN 0.1.3

Audit exact and near duplicates in T1-weighted brain MRI using a BIDS directory or a scan CSV.

**Requirements:** Linux x86_64, Bash 4.3+, and Apptainer or Singularity. GPU execution requires an NVIDIA GPU with a CUDA 12.8-compatible driver. The SIF includes dependencies and models. Read `LICENSE.pdf` before use.

## 1. Download and prepare

Download `happen-v0.1.3-cu128.sif`, `happen-0.1.3-launcher.zip`, `README.md`, `LICENSE.pdf`, and `SHA256SUMS.txt` from the same Zenodo record into one directory.

```bash
sha256sum -c SHA256SUMS.txt
export HAPPEN_IMAGE="$PWD/happen-v0.1.3-cu128.sif"
unzip happen-0.1.3-launcher.zip
cd happen-0.1.3
chmod +x run_happen.sh
cp configs/config_usr.toml my_config.toml
```

## 2. Set the output and input

Edit these entries in `my_config.toml`; keep the remaining settings:

```toml
[run]
out = "/absolute/path/to/results"
stages = ["exact", "near"]

[exact]
root = "/absolute/path/to/BIDS"
candidates_csv = ""
valid_csv = ""

[near]
review_mode = "lazy"

[review]
host = "127.0.0.1"
port = 8000
```

For CSV input, set `root = ""` and `candidates_csv = "/absolute/path/to/scans.csv"`; keep `valid_csv` empty. Use this format, one scan per row:

```csv
dataset,subject_id,session_id,candidate
DatasetA,sub-001,ses-01,/data/DatasetA/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
```

Absolute paths are recommended. `candidate` may be a relative or absolute path, or a symlink to a T1w `.nii` or `.nii.gz` file. Relative paths resolve from the configuration directory, not the CSV directory. Omit leading `./` and repeated `/` in CSV paths to preserve metadata. Provide nonempty `dataset` and `subject_id`; `session_id` may be empty.

## 3. Run the pipeline

For BIDS input:

```bash
./run_happen.sh pipeline my_config.toml
```

For CSV input, mount the MRI data directory:

```bash
./run_happen.sh pipeline my_config.toml --bind /data
```

Replace `/data` with the directory containing your MRI files. Repeat `--bind` for other directories or external symlink targets. Match worker counts and batch size to your CPU/GPU resources.

## 4. Review candidates

```bash
./run_happen.sh review my_config.toml
```

Open **http://127.0.0.1:8000**, review candidate pairs, and save your decisions.

## 5. Generate confirmed groups

Stop review with `Ctrl+C`, then run:

```bash
./run_happen.sh review my_config.toml finalize
```

Keep the same configuration and output directory throughout. Results are under `[run].out`: `exact/duplicates.csv` contains exact groups; `near/subject_groups.csv` contains confirmed near-duplicate groups after finalization. Decisions are in `near/review/review_decisions.csv`.

## Citation and license

Cite the [arXiv preprint](https://arxiv.org/abs/2610.09614) and the [software release](https://doi.org/10.5281/zenodo.21986366). Both records are in `CITATION.cff`.

HAPPEN uses Vanderbilt's Non-Exclusive Non-Commercial Academic Software License Agreement (`LICENSE.pdf`). Use is limited to non-profit academic/research institutions for internal non-commercial research. Diagnostic/treatment use and redistribution by licensees are prohibited. Commercial licensing: cttc@vanderbilt.edu. Third-party notices: `resources/licenses/`.

Source and full input/configuration reference: [MASILab/HAPPEN](https://github.com/MASILab/HAPPEN#configuration).
