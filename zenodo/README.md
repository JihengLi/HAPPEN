# HAPPEN 0.1.3 — Quick start

HAPPEN audits identity duplication in T1-weighted brain MRI repositories using exact-duplicate detection, pretrained similarity retrieval, and human review. Finalization generates reviewer-confirmed subject groups.

**Requirements:** Linux x86_64, Bash 4.3+, and Apptainer or Singularity. The GPU pipeline needs an NVIDIA GPU with a driver compatible with CUDA 12.8. Application dependencies and models are inside the SIF.

## 1. Download and prepare

Download these five files from the same Zenodo record into one directory:

- `happen-v0.1.3-cu128.sif`
- `happen-0.1.3-launcher.zip`
- `LICENSE.pdf`
- `README.md`
- `SHA256SUMS.txt`

```bash
sha256sum -c SHA256SUMS.txt
export HAPPEN_IMAGE="$PWD/happen-v0.1.3-cu128.sif"
unzip happen-0.1.3-launcher.zip
cd happen-0.1.3
chmod +x run_happen.sh
cp configs/config_usr.toml my_config.toml
```

## 2. Configure

Edit these entries in `my_config.toml`; keep the remaining template settings:

```toml
[run]
out = "/absolute/path/to/happen-output"
stages = ["exact", "near"]

[exact]
root = "/absolute/path/to/bids-repository"
candidates_csv = ""
valid_csv = ""

[near]
review_mode = "lazy"

[review]
host = "127.0.0.1"
port = 8000
```

Use absolute paths. Choose exactly one input: a BIDS-style `root`, `candidates_csv`, or `valid_csv`.

CSV headers:

- `candidates_csv`: `dataset,subject_id,session_id,candidate`
- `valid_csv`: `dataset,subject_id,session_id,candidate,resolved_path`

MRI paths in CSVs must be accessible inside the container. Add their data directories with `--bind`, including external symlink targets. Match worker counts and batch size to your CPU/GPU allocation.

## 3. Run, review, and finalize

```bash
./run_happen.sh pipeline my_config.toml
# For CSV input or additional data directories, use:
# ./run_happen.sh pipeline my_config.toml --bind /absolute/path/to/mri-data

./run_happen.sh review my_config.toml
```

Open **http://127.0.0.1:8000** and review candidate pairs. For a remote server, run this on your own computer:

```bash
ssh -N -L 8000:127.0.0.1:8000 username@server
```

For a cluster job, the tunnel must reach the compute node running review. Adjust both ports if you changed the configuration.

After saving your decisions, stop the review server with `Ctrl+C` and run:

```bash
./run_happen.sh review my_config.toml finalize
```

Outputs are under `[run].out`: exact groups in `exact/duplicates.csv`, review decisions in `near/review/review_decisions.csv`, and confirmed subject groups in `near/subject_groups.csv` after finalization. Keep the same configuration/output directory for all three commands. Review and finalization do not request GPU passthrough.

## Citation and licenses

Cite [HAPPEN 0.1.3](https://doi.org/10.5281/zenodo.21986366); metadata is in `CITATION.cff`. Before downloading or using HAPPEN, read Vanderbilt's [Non-Exclusive Non-Commercial Academic Software License Agreement](https://github.com/JihengLi/HAPPEN/blob/main/LICENSE.pdf), attached as `LICENSE.pdf` and included in the launcher ZIP. It is available only to non-profit academic and/or research institutions for internal non-commercial research. Diagnostic/treatment use and redistribution by licensees are prohibited. Commercial licensing: cttc@vanderbilt.edu. Third-party notices remain in `resources/licenses/`.

Source and extended documentation: [github.com/JihengLi/HAPPEN](https://github.com/JihengLi/HAPPEN).
