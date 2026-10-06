# Publish HAPPEN 0.1.3 to Zenodo

Run these commands in Bash from the repository root on Linux x86_64 with Docker, Apptainer, Python 3.11+, and an NVIDIA GPU. Put the correct pretrained weights at `resources/model.pth` first; this file is not supplied by Git.

## 1. Build Docker and SIF

```bash
set -euo pipefail
test -s LICENSE.pdf
cmp LICENSE.pdf zenodo/LICENSE.pdf
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

The [tar context](https://docs.docker.com/build/concepts/context/#local-tarballs) keeps generated downloads out of the Docker build. [Apptainer's `--force`](https://apptainer.org/docs/user/latest/cli/apptainer_build.html) overwrites an existing SIF with the same filename.

## 2. Check the SIF

```bash
export HAPPEN_IMAGE="$PWD/zenodo/happen-v0.1.3-cu128.sif"
apptainer inspect --labels "$HAPPEN_IMAGE"
sha256sum LICENSE.pdf
apptainer exec "$HAPPEN_IMAGE" sha256sum /app/LICENSE.pdf
apptainer exec "$HAPPEN_IMAGE" python -c \
  "import torch; from importlib.metadata import version; from happen.near.model import load_model; assert version('happen') == '0.1.3'; load_model(device=torch.device('cpu')); print('Version and checkpoint OK')"
apptainer exec --nv "$HAPPEN_IMAGE" python -c \
  "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

The two PDF hashes must match, and the license label must be `LicenseRef-HAPPEN-NonCommercial-Academic`. Model loading must finish without missing/unexpected-key warnings. Follow `zenodo/README.md` to test pipeline, review, and finalization on a small MRI dataset before uploading.

## 3. Package the release

```bash
python3 zenodo/package_release.py --sif zenodo/happen-v0.1.3-cu128.sif
(cd zenodo/dist/zenodo-0.1.3 && sha256sum -c SHA256SUMS.txt)
```

Packaging copies the SIF, so allow space for one extra image copy. If the output directory already exists, choose a new one:

```bash
python3 zenodo/package_release.py --sif zenodo/happen-v0.1.3-cu128.sif \
  --output-dir zenodo/dist/zenodo-0.1.3-r2
```

## 4. Upload to Zenodo

Create or open your unpublished Zenodo draft. Under **Settings → Applications → Personal access tokens**, create a token with `deposit:write`. Copy the numeric draft ID from its upload/edit URL; use the existing draft associated with your reserved DOI.

Run this from the repository root in Bash with `curl` and Python 3 (Linux or Mac). Change `ZENODO_UPLOAD_DIR` if you used a different output directory:

```bash
set -euo pipefail
ZENODO_UPLOAD_DIR="$PWD/zenodo/dist/zenodo-0.1.3"
ZENODO_FILES=(
  README.md LICENSE.pdf happen-0.1.3-launcher.zip
  happen-v0.1.3-cu128.sif SHA256SUMS.txt
)
for filename in "${ZENODO_FILES[@]}"; do
  test -r "$ZENODO_UPLOAD_DIR/$filename"
done

read -rp "Zenodo draft ID: " ZENODO_DEPOSIT_ID
read -rsp "Zenodo token: " ZENODO_TOKEN
printf '\n'

ZENODO_BUCKET=$(
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $ZENODO_TOKEN" \
    "https://zenodo.org/api/deposit/depositions/$ZENODO_DEPOSIT_ID" | \
  python3 -c 'import json, sys; d = json.load(sys.stdin); assert d["submitted"] is False, "Use an unpublished draft"; print(d["links"]["bucket"])'
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

This uses Zenodo's [large-file upload API](https://developers.zenodo.org/#quickstart-upload). Same-name uploads replace the current draft file. Retries restart the complete file; HTTP PUT uploads do not support [curl resume](https://curl.se/docs/manpage.html#-C). Stop an active browser upload of the same file first.

After upload, check all five files in the Zenodo draft. Choose **Software**, use version `0.1.3`, and copy the citation/date/DOI information from `zenodo/CITATION.cff`. Set the license to **Add custom**:

- Title: **Non-Exclusive Non-Commercial Academic Software License Agreement**
- URL: `https://github.com/JihengLi/HAPPEN/blob/main/LICENSE.pdf`

Keep the official PDF unchanged. Review the metadata and click **Publish** in Zenodo; the commands above only upload files. After publishing, send CTTC the GitHub and Zenodo record links.
