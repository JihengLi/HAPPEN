#!/usr/bin/env python3
"""Prepare Zenodo launcher files with Python 3.11; do not rebuild or validate SIFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path


def cff_scalar(text: str, key: str, *, required: bool = True) -> str | None:
    """Read an unindented, simple scalar without accepting arbitrary YAML."""
    values = re.findall(rf"^{re.escape(key)}:[ \t]*(.*)$", text, re.MULTILINE)
    if not values and not required:
        return None
    if len(values) != 1:
        raise ValueError(f"CITATION.cff must have exactly one top-level {key} field")
    value = values[0].strip()
    if value.startswith('"'):
        try:
            parsed, end = json.JSONDecoder().raw_decode(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported quoted {key} in CITATION.cff") from exc
        if not isinstance(parsed, str) or value[end:].strip():
            raise ValueError(f"CITATION.cff {key} must be a simple string")
        return parsed
    if value.startswith("'"):
        if len(value) < 2 or not value.endswith("'"):
            raise ValueError(f"CITATION.cff {key} must be a simple string")
        return value[1:-1].replace("''", "'")
    if not value or re.search(r"[\s\[\]{}#]", value):
        raise ValueError(f"CITATION.cff {key} must be a simple string")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_contents(archive: Path, directory: Path, prefix: str, sif_name: str | None) -> None:
    expected_members = {
        f"{prefix}/run_happen.sh",
        f"{prefix}/configs/config_usr.toml",
        f"{prefix}/CITATION.cff",
        f"{prefix}/LICENSE",
        f"{prefix}/resources/licenses/ANTs_LICENSE.txt",
        f"{prefix}/resources/licenses/DeepBet_LICENSE.txt",
        f"{prefix}/resources/licenses/MNI_ICBM_2009c_LICENSE.txt",
    }
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(expected_members) or set(names) != expected_members:
            raise ValueError("Launcher ZIP contents do not match the seven required files")
    expected_uploads = {"README.md", archive.name}
    if sif_name is not None:
        expected_uploads.update({sif_name, "SHA256SUMS.txt"})
    uploads = list(directory.iterdir())
    if {path.name for path in uploads} != expected_uploads or any(not path.is_file() for path in uploads):
        raise ValueError("Upload directory contents do not match the required files")


def prepare(bundle_dir: Path, output_arg: Path | None, sif_arg: Path | None) -> tuple[Path, list[Path], Path | None]:
    bundle_dir = bundle_dir.expanduser().resolve()
    citation = bundle_dir / "CITATION.cff"
    required = [
        citation,
        bundle_dir / "README.md",
        bundle_dir / "run_happen.sh",
        bundle_dir / "configs" / "config_usr.toml",
        bundle_dir / "LICENSE",
    ]
    missing = [str(path.relative_to(bundle_dir)) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Missing required file(s): " + ", ".join(missing))
    readme_words = len((bundle_dir / "README.md").read_text(encoding="utf-8").split())
    if readme_words > 500:
        raise ValueError(f"README.md must be at most 500 words (found {readme_words})")
    cff = citation.read_text(encoding="utf-8")
    version = cff_scalar(cff, "version")
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]*", version):
        raise ValueError("CITATION.cff must contain a safe, static version")
    doi = cff_scalar(cff, "doi")
    if doi is None or not re.fullmatch(r"10\.[0-9]{4,9}/[^\s]+", doi):
        raise ValueError("CITATION.cff doi must be a DOI without a URL prefix")
    url = cff_scalar(cff, "url", required=False)
    if url and url.startswith("https://doi.org/") and url != f"https://doi.org/{doi}":
        raise ValueError("CITATION.cff DOI and DOI URL disagree")

    license_names = ("ANTs_LICENSE.txt", "DeepBet_LICENSE.txt", "MNI_ICBM_2009c_LICENSE.txt")
    licenses = [bundle_dir / "resources" / "licenses" / name for name in license_names]
    missing_licenses = [str(path.relative_to(bundle_dir)) for path in licenses if not path.is_file()]
    if missing_licenses:
        raise ValueError("Missing required license(s): " + ", ".join(missing_licenses))
    expected_sif = f"happen-v{version}-cu128.sif"
    sif = None
    if sif_arg is None and (bundle_dir / expected_sif).exists():
        sif_arg = bundle_dir / expected_sif
    if sif_arg is not None:
        sif = sif_arg.expanduser().resolve()
        if not sif.is_file():
            raise ValueError("--sif must point to an existing regular file")
        if sif.suffix != ".sif" or sif.name != expected_sif:
            raise ValueError(f"The SIF must be named {expected_sif} to match the README")

    output = (output_arg.expanduser().resolve() if output_arg else bundle_dir / "dist" / f"zenodo-{version}")
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}; choose a new --output-dir")
    prefix = f"happen-{version}"
    members = [
        (bundle_dir / "run_happen.sh", "run_happen.sh"),
        (bundle_dir / "configs" / "config_usr.toml", "configs/config_usr.toml"),
        (citation, "CITATION.cff"),
        (bundle_dir / "LICENSE", "LICENSE"),
        *[(path, f"resources/licenses/{path.name}") for path in licenses],
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build in a temporary sibling; no partial release directory on failure.
    with tempfile.TemporaryDirectory(prefix=".happen-zenodo-", dir=output.parent) as temporary:
        temporary_dir = Path(temporary)
        readme = temporary_dir / "README.md"
        readme.write_bytes((bundle_dir / "README.md").read_bytes())
        archive = temporary_dir / f"{prefix}-launcher.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for source, relative in members:
                info = zipfile.ZipInfo.from_file(source, arcname=f"{prefix}/{relative}")
                info.create_system = 3
                mode = 0o755 if relative == "run_happen.sh" else 0o644
                info.external_attr = (stat.S_IFREG | mode) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(info, source.read_bytes())
        if sif is not None:
            # Keep every final upload file together, including images passed elsewhere.
            copied_sif = temporary_dir / expected_sif
            shutil.copyfile(sif, copied_sif)
            checksum_inputs = [readme, archive, copied_sif]
            checksum_text = "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_inputs)
            (temporary_dir / "SHA256SUMS.txt").write_text(checksum_text, encoding="utf-8")
        check_contents(archive, temporary_dir, prefix, expected_sif if sif is not None else None)
        # mkdir fails if another process has created the requested destination.
        output.mkdir()
        for path in sorted(temporary_dir.iterdir()):
            path.rename(output / path.name)
    files = [output / "README.md", output / archive.name]
    if sif is not None:
        files.extend([output / expected_sif, output / "SHA256SUMS.txt"])
    return output, files, sif


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Self-contained zenodo directory; defaults to this script's directory")
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing directories are refused")
    parser.add_argument("--sif", type=Path,
                        help="Final tested happen-vVERSION-cu128.sif; defaults to image in bundle directory; copied to output")
    args = parser.parse_args()
    try:
        output, files, sif = prepare(args.bundle_dir, args.output_dir, args.sif)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Prepared: {output}")
    print("Upload files:")
    for path in files:
        print(f"  {path}")
    if sif is None:
        print("Pending: final tested SIF. Host package only; SHA256SUMS.txt was not generated.")
    else:
        print("SIF was copied and checksummed in the output directory; runtime validation must be completed separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
