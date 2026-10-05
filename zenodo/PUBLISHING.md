# Preparing the HAPPEN 0.1.3 Zenodo release

All Zenodo documents, launcher files, and configuration/license copies are kept in `zenodo/`. The original repository layout, Dockerfile, launcher, README, and ignore rules are preserved.

```text
zenodo/
├── README.md
├── PUBLISHING.md
├── CITATION.cff
├── LICENSE
├── run_happen.sh
├── configs/config_usr.toml
├── resources/licenses/
├── package_release.py
├── .gitignore
├── happen-v0.1.3-cu128.sif        # generated on Linux; ignored by Git
└── dist/zenodo-0.1.3/            # generated final upload files; ignored by Git
```

The copied CFF, configuration, and licenses are release snapshots. If they change in the main repository, refresh the corresponding copies here before preparing another release. The Zenodo copy of `run_happen.sh` additionally has working help, checked configuration-parser errors, and a default review port of 8000; the original launcher is preserved.

## Metadata

- Version: `0.1.3`, matching the copied `CITATION.cff` and the current application version.
- Image: `happen-v0.1.3-cu128.sif`, CUDA 12.8, Linux x86_64.
- DOI: `10.5281/zenodo.21986366`, provided this is reserved for this exact release record. It becomes resolvable after publication.
- Release date: `2026-10-05`, if this is the actual publication date; keep the copied CFF and Zenodo date consistent.
- Repository URL: `https://github.com/MASI/HAPPEN`. Confirm this is the intended public location; the local checkout currently has a different origin.
- Resource type: Software. Source license: MIT. Preserve the third-party notices in the launcher package and image.

The CFF key is `orcid`. `cff-version: 1.2.0` describes the citation format, while `version: 0.1.3` describes the software. For a manual Zenodo upload, fill in the deposit form in addition to supplying the citation file.

If a published release is changing substantively, create a new software/Zenodo version. For an unpublished 0.1.3 draft, build and verify the final image before publishing.

## 1. Build the image using the existing Dockerfile

On a Linux x86_64 machine with Docker and Apptainer, pull the intended source checkout. Place the correct pretrained checkpoint in the main repository at `resources/model.pth`. This file is ignored by Git, so transfer it to the Linux machine separately.

From the repository root:

```bash
test -s resources/model.pth
test -s resources/mni_1mm3_t1_brain_atlas.nii.gz
git status --short
```

Record the release commit before building. The original Dockerfile copies `resources/` into `/app/resources/`; the near-stage loader expects `/app/resources/model.pth`. Its existing import checks do not prove that the checkpoint is usable, so complete the image checks in the next section.

The following command supplies the files required by the existing Dockerfile as the build context. It leaves `zenodo/` and its generated downloads outside that context, including during later rebuilds.

```bash
set -o pipefail
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
  -cf - Dockerfile.gpu .dockerignore pyproject.toml uv.lock README.md LICENSE \
  src resources docker/entrypoint.sh | \
docker build --platform linux/amd64 -f Dockerfile.gpu \
  --build-arg HAPPEN_VERSION=0.1.3 \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t happen:v0.1.3-cu128 -

apptainer build zenodo/happen-v0.1.3-cu128.sif \
  docker-daemon:happen:v0.1.3-cu128
```

See the official [Docker tar-context guide](https://docs.docker.com/build/concepts/context/#local-tarballs) and [Apptainer Docker conversion guide](https://apptainer.org/docs/user/latest/docker_and_oci.html#containers-cached-by-the-docker-daemon).

Use this newly built SIF. An older image can carry the same version string while containing different source code.

## 2. Verify the actual SIF and standalone launcher

From the repository root:

```bash
export HAPPEN_IMAGE="$(pwd)/zenodo/happen-v0.1.3-cu128.sif"
apptainer inspect --labels "$HAPPEN_IMAGE"
apptainer exec "$HAPPEN_IMAGE" python -c \
  "from importlib.metadata import version; assert version('happen') == '0.1.3'; print('HAPPEN version OK')"
apptainer exec "$HAPPEN_IMAGE" python -c \
  "import torch; from happen.near.model import load_model; load_model(device=torch.device('cpu')); print('HAPPEN checkpoint load OK')"
apptainer exec "$HAPPEN_IMAGE" test -s /app/resources/mni_1mm3_t1_brain_atlas.nii.gz
apptainer exec "$HAPPEN_IMAGE" antsRegistration --version
apptainer exec "$HAPPEN_IMAGE" N4BiasFieldCorrection --version
apptainer exec --nv "$HAPPEN_IMAGE" python -c \
  "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Confirm version 0.1.3, the intended source revision, and x86_64/amd64 architecture in the SIF metadata. Check that model loading produces no missing or unexpected model-key warnings.

Use a small local MRI input for the complete documented pipeline, browser review, and finalization workflow. Run the launcher copy from `zenodo/`, not the original repository launcher:

```bash
cd zenodo
cp configs/config_usr.toml my_config.toml
# Edit input/output paths and review settings in my_config.toml.
./run_happen.sh pipeline my_config.toml
./run_happen.sh review my_config.toml
# Stop the review server after saving the decisions.
./run_happen.sh review my_config.toml finalize
```

Follow `README.md` for configuration, CSV mounts, and browser access. This run checks ANTs, DeepBet, embedding inference, assets, and the entrypoint together.

## 3. Collect the final upload files

From the repository root, using Python 3.11 or newer:

```bash
python3 zenodo/package_release.py
```

The tool reads only the release copies inside `zenodo/` and automatically uses `zenodo/happen-v0.1.3-cu128.sif` when present. It creates a new `zenodo/dist/zenodo-0.1.3/` directory containing all four upload files:

```text
README.md
happen-0.1.3-launcher.zip
happen-v0.1.3-cu128.sif
SHA256SUMS.txt
```

Upload these four files together to the same Zenodo record. The SIF is copied into the output directory, so allow disk space for that copy. To use a newly tested SIF from a different location, run:

```bash
python3 zenodo/package_release.py --sif /absolute/path/happen-v0.1.3-cu128.sif
```

Existing output directories are preserved; choose a new `--output-dir` for another packaging run. If no SIF is available, the tool creates a launcher preview only, without final image checksums. It never substitutes a preview for a verified image.

The ZIP contains exactly seven files: the launcher, configuration template, CFF, HAPPEN license, and three third-party license notices. The README is a separate upload and is limited to 500 words. Build instructions, packaging tools, Git metadata, and duplicate documentation stay out of the ZIP. The application and pretrained models are inside the SIF; users need neither a Git clone nor registry authentication.

Copy the four files into a fresh directory and follow the packaged README using only those downloads. Verify the checksums and repeat pipeline, review, and finalization before publishing.

## Suggested Zenodo description

HAPPEN is a containerized framework for auditing identity duplication in large T1-weighted brain MRI repositories. It detects exact and near duplicates, supports human adjudication of candidate links, and produces reviewer-confirmed subject groups.

This release includes a Singularity/Apptainer SIF image and a standalone launcher package with a configuration template, citation metadata, and license notices. Download the SIF image, launcher ZIP, README, and checksum file from this record, then follow the README to run the pipeline, review candidates, and generate confirmed reports. A GitHub clone or container-registry login is not required for this workflow.

Software version: 0.1.3. Image variant: CUDA 12.8, Linux x86_64. Source code: https://github.com/MASI/HAPPEN.
