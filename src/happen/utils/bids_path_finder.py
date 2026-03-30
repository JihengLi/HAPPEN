"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from pathlib import Path
from typing import Dict, Union, Tuple
import re

VALID_ENTS = {"sub", "ses", "task", "acq", "ce", "rec", "run", "echo", "part", "chunk"}
ENT_PAIR_RE = re.compile(r"([a-z]+-[^_]+)")
NII_GZ_RE = re.compile(r"\.nii(\.gz)?$", re.IGNORECASE)


def _ensure_str(x) -> str:
    if x is None:
        raise TypeError("subject_id/stem is None")
    if isinstance(x, (str, bytes)):
        return x.decode() if isinstance(x, bytes) else x
    return str(x)


def _basename_no_ext(p: str) -> str:
    name = Path(p).name
    return NII_GZ_RE.sub("", name)


def parse_entities(stem: Union[str, Path]) -> Dict[str, str]:
    s = _ensure_str(stem)
    s = _basename_no_ext(s)
    return {
        k: v
        for k, v in (pair.split("-", 1) for pair in ENT_PAIR_RE.findall(s))
        if k in VALID_ENTS
    }


def _subject_id_and_suffix_from_t1(t1_path: Path) -> Tuple[str, str]:
    base = _basename_no_ext(str(t1_path))
    parts = base.split("_")
    suffix = parts[-1] if len(parts) >= 2 else "T1w"
    subject_id = "_".join(parts[:-1]) if len(parts) >= 2 else base
    return subject_id, suffix


def _dataset_root_and_sub(t1_path: Path) -> Tuple[Path, str]:
    for anc in t1_path.parents:
        if anc.name.startswith("sub-"):
            return anc.parent, anc.name
    raise ValueError(f"Could not infer dataset root/sub-xxx from: {t1_path}")


def find_t1w_addr_subjectid(
    subject_id: Union[str, Path], t1_root: Union[str, Path]
) -> Path:
    t1_root = Path(t1_root)
    ents = parse_entities(subject_id)
    if "sub" not in ents:
        raise ValueError(f"No 'sub-' entity found in subject_id: {subject_id}")
    sub_dir = t1_root / f"sub-{ents['sub']}"
    ses_dir = sub_dir / f"ses-{ents['ses']}" if "ses" in ents else sub_dir
    anat_dir = ses_dir / "anat"
    if not anat_dir.exists():
        raise FileNotFoundError(f"anat dir not found: {anat_dir}")
    target_name = f"{_ensure_str(subject_id)}_T1w.nii.gz"
    target_path = anat_dir / target_name
    if target_path.is_file():
        return target_path
    hits = list(anat_dir.glob("*_T1w.nii.gz"))
    if not hits:
        raise FileNotFoundError(
            f"{target_name} not found for {subject_id} in {anat_dir}"
        )
    if len(hits) > 1:
        raise RuntimeError(f"Multiple T1w files for {subject_id}: {hits}")
    return hits[0]


def find_slant_addr_subjectid(
    subject_id: Union[str, Path], slant_root: Union[str, Path]
) -> Path:
    root = Path(slant_root)
    ents = parse_entities(subject_id)
    if "sub" not in ents:
        raise ValueError(f"No 'sub-' entity found in subject_id: {subject_id}")
    sub_dir = root / f"sub-{ents['sub']}"
    level_dir = sub_dir / f"ses-{ents['ses']}" if "ses" in ents else sub_dir
    if not level_dir.exists():
        raise FileNotFoundError(f"Level dir not found: {level_dir}")
    slant_dir = None
    for d in level_dir.iterdir():
        if d.is_dir() and d.name.startswith("SLANT-TICVv1.2"):
            slant_dir = d
            break
    if slant_dir is None:
        raise FileNotFoundError(
            f"No SLANT dir with prefix 'SLANT-TICVv1.2' under {level_dir}"
        )
    target_name = f"{_ensure_str(subject_id)}_T1w_seg.nii.gz"
    cand = slant_dir / "post" / "FinalResult" / target_name
    if not cand.is_file():
        raise FileNotFoundError(f"{cand} not found")
    return cand


def find_slant_addr(t1_path: Union[str, Path]) -> Path:
    p = Path(t1_path)
    if not p.exists():
        raise FileNotFoundError(f"T1 path not found: {p}")
    dataset_root, sub_dir = _dataset_root_and_sub(p)
    ents = parse_entities(p.name)
    derivatives_root = dataset_root / "derivatives" / sub_dir
    level_dir = (
        derivatives_root / f"ses-{ents['ses']}" if "ses" in ents else derivatives_root
    )
    if not level_dir.exists():
        raise FileNotFoundError(f"SLANT level dir not found: {level_dir}")
    slant_dir = None
    for d in level_dir.iterdir():
        if d.is_dir() and d.name.startswith("SLANT-TICVv1.2"):
            slant_dir = d
            break
    if slant_dir is None:
        raise FileNotFoundError(
            f"No SLANT dir with prefix 'SLANT-TICVv1.2' under {level_dir}"
        )
    subj_id, suffix = _subject_id_and_suffix_from_t1(p)
    cand = slant_dir / "post" / "FinalResult" / f"{subj_id}_{suffix}_seg.nii.gz"
    if not cand.is_file():
        raise FileNotFoundError(f"{cand} not found")
    return cand
