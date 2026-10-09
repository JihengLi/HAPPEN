# Publish HAPPEN 0.1.3 to Zenodo

Run the commands in one Bash session from the HAPPEN repository root on Linux, with Docker, Apptainer, Python 3.11+, and curl. Put the pretrained checkpoint at `resources/model.pth` first.

## 1. Build Docker and SIF

```bash
set -euo pipefail
test -s resources/model.pth
test -s resources/mni_1mm3_t1_brain_atlas.nii.gz

tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
  -cf - Dockerfile.gpu .dockerignore pyproject.toml uv.lock README.md LICENSE.pdf \
  src resources docker/entrypoint.sh | \
sudo docker build --platform linux/amd64 -f Dockerfile.gpu \
  --build-arg HAPPEN_VERSION=0.1.3 \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t happen:v0.1.3-cu128 -

sudo apptainer build --force zenodo/happen-v0.1.3-cu128.sif \
  docker-daemon:happen:v0.1.3-cu128
```

This replaces the local SIF and includes the current code, README, license, and model.

## 2. Check the checkpoint

```bash
apptainer exec zenodo/happen-v0.1.3-cu128.sif python -c \
  "import torch; from importlib.metadata import version; from happen.near.model import load_model; assert version('happen') == '0.1.3'; load_model(device=torch.device('cpu')); print('Version and checkpoint OK')"
```

Model loading must finish without missing/unexpected-key warnings.

## 3. Package all upload files

```bash
RELEASE_DIR="zenodo/dist/zenodo-0.1.3-$(date +%Y%m%d-%H%M%S)"
python3 zenodo/package_release.py \
  --sif zenodo/happen-v0.1.3-cu128.sif --output-dir "$RELEASE_DIR"
(cd "$RELEASE_DIR" && sha256sum -c SHA256SUMS.txt)
```

Upload all five files from that directory:

- `README.md`
- `happen-0.1.3-launcher.zip`
- `LICENSE.pdf`
- `happen-v0.1.3-cu128.sif`
- `SHA256SUMS.txt`

## 4. Replace files in the existing Zenodo record

Open the original record → **Edit** → **Edit files** → **Edit published files**. This option allows minor corrections within 30 days of publication; republishing keeps the DOI. [Zenodo instructions](https://help.zenodo.org/docs/deposit/manage-files/#modify-files-after-publication).

Use a token with `deposit:write` from [Settings → Applications](https://zenodo.org/account/settings/applications/). After opening file editing, [upload](https://github.com/zenodo/zenodo-rdm/blob/master/site/zenodo_rdm/legacy/resources.py) all five files:

```bash
set -euo pipefail
ZENODO_UPLOAD_DIR="$PWD/$RELEASE_DIR"
ZENODO_DEPOSIT_ID=21986366
ZENODO_FILES=(
  README.md LICENSE.pdf happen-0.1.3-launcher.zip
  happen-v0.1.3-cu128.sif SHA256SUMS.txt
)
for filename in "${ZENODO_FILES[@]}"; do
  test -r "$ZENODO_UPLOAD_DIR/$filename"
done

read -rsp "Zenodo token: " ZENODO_TOKEN
printf '\n'

ZENODO_BUCKET=$(
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $ZENODO_TOKEN" \
    "https://zenodo.org/api/deposit/depositions/$ZENODO_DEPOSIT_ID" | \
  python3 -c 'import json, sys
d = json.load(sys.stdin)
assert str(d["id"]) == "21986366", "Wrong Zenodo record"
assert d["doi"] == "10.5281/zenodo.21986366", "Wrong software DOI"
assert d["submitted"] is True and d["state"] == "inprogress", "Open Edit published files first"
print(d["links"]["bucket"])'
)

for filename in "${ZENODO_FILES[@]}"; do
  printf 'Uploading %s\n' "$filename"
  curl --fail --show-error --progress-bar \
    --retry 3 --retry-delay 5 --connect-timeout 30 \
    -H "Authorization: Bearer $ZENODO_TOKEN" \
    --upload-file "$ZENODO_UPLOAD_DIR/$filename" \
    --output /dev/null "${ZENODO_BUCKET%/}/$filename"
done
unset ZENODO_TOKEN
```
