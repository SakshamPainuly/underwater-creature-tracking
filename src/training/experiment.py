"""Experiment configs: parsing, validation, Ultralytics arguments and dataset checks (stdlib only).

Configs live in configs/experiments/E###_<slug>.yaml (CLAUDE.md section 5). They are YAML, written
in a deliberately small subset that this module parses without PyYAML, so the local PC (standard
library only) can validate them. On Colab train.py also parses each config with PyYAML and stops if
the two parses differ, so what was validated locally is exactly what trains.

Supported subset: `key: scalar` lines, and one level of nesting (`key:` followed by lines indented
by exactly two spaces). Scalars: int, decimal float, true/false, null, a "double"- or
'single'-quoted string without escapes, or a plain string starting with a letter. Anything that
YAML 1.1 could read differently from this parser (yes/no/on/off, 1e-3, 010, 1_000, ...) is
rejected with a request to quote or rewrite it.

Test-split guard: configs cannot name a data file or a split; the runner builds a data file that
has no `test` entry at all (write_trainval_yaml) and passes it to Ultralytics.
"""

import csv
import json
import re
from pathlib import Path

from dataset_audit.config import DEFAULT_IMAGE_SOURCE, MISSING_LABEL_POLICY, PROJECT_ROOT
from dataset_audit import tritonia_dataset as T
from dataset_audit.versioning import (PROCESSED_ROOT, VersionError, image_source_of, is_frozen, read_version_meta,
                                      sha256_file, verify_frozen, version_dir)
from yolo_prep import mixed as M
from yolo_prep.build import (CLASS_NAMES, INCLUDED_STATUSES, SPLITS, TARGET_ULTRALYTICS_VERSION, BuildError,
                             dataset_fingerprint)

EXPERIMENTS_CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"
TRAIN_SPLITS = ("train", "val")  # the only splits a training run can see
VAL_SPLIT = "val"
SMOKE_FRACTION = 0.1

REQUIRED_KEYS = ("id", "slug", "parent", "changed", "dataset_version", "dataset_fingerprint",
                 "model", "ultralytics_version", "seed", "train_args")
OPTIONAL_KEYS = ("notes",)
REQUIRED_TRAIN_ARGS = ("imgsz", "epochs")
# Set by the runner only, never by a config: data/split decide which images are used (test guard),
# project/name/exist_ok/resume decide where checkpoints go, seed/model/fraction come from the
# config's own fields or --smoke, and `cfg` would let an Ultralytics file override all of them.
RUNNER_KEYS = frozenset({"data", "split", "project", "name", "exist_ok", "resume", "model", "seed",
                         "fraction", "val", "save", "mode", "task", "source", "cfg"})

_ID_RE = re.compile(r"^E\d{3}$")
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_MODEL_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*\.pt$")  # an official checkpoint name, never a path
# Checkpoint name prefix -> Ultralytics model family. RT-DETR needs its own model class (trainer,
# NMS-free validator, predictor); loading an RT-DETR checkpoint with YOLO() would use the YOLO ones.
MODEL_FAMILY_PREFIXES = (("rtdetr-", "rtdetr"), ("yolo", "yolo"))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TEST_RE = re.compile(r"(^|[\\/])test([\\/.]|$)", re.IGNORECASE)

_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(?:[ ]+(.*))?$")
_INT_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_FLOAT_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]+$")
_PLAIN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-/+ ]*$")
_QUOTED_RE = re.compile(r"""^(?:"([^"\\]*)"|'([^']*)')(?:[ ]+#.*)?$""")
_YAML11_WORDS = {"yes", "no", "on", "off", "y", "n", "true", "false", "null", "nan", "inf"}


class ConfigError(ValueError):
    pass


# ---------------------------------------------------------------- parsing (restricted YAML subset)

def _parse_scalar(raw: str, where: str):
    m = _QUOTED_RE.match(raw)
    if m:
        return m.group(1) if m.group(1) is not None else m.group(2)
    value = raw.split(" #", 1)[0].rstrip()
    if value == "":
        raise ConfigError(f"{where}: empty value")
    if value in ("true", "false"):
        return value == "true"
    if value == "null":
        return None
    if _INT_RE.match(value):
        return int(value)
    if _FLOAT_RE.match(value):
        return float(value)
    if value.lower() in _YAML11_WORDS or not _PLAIN_RE.match(value):
        raise ConfigError(f"{where}: {value!r} is ambiguous in YAML; quote it or use a decimal "
                          "number / lowercase true, false, null")
    return value


def parse_config_text(text: str) -> dict:
    """Parse the restricted YAML subset described in the module docstring."""
    result: dict = {}
    block = None  # name of the open nested mapping
    for n, line in enumerate(text.splitlines(), 1):
        where = f"line {n}"
        if "\t" in line:
            raise ConfigError(f"{where}: tabs are not allowed")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        m = _KEY_RE.match(line.strip())
        if indent not in (0, 2) or not m:
            raise ConfigError(f"{where}: expected 'key: value' at indent 0 or 2, got {line!r}")
        key, raw = m.group(1), m.group(2)
        if indent == 0:
            if block is not None and not result[block]:
                raise ConfigError(f"'{block}:' has no entries")
            if key in result:
                raise ConfigError(f"{where}: duplicate key {key!r}")
            if raw is None or raw.startswith("#"):
                result[key], block = {}, key
            else:
                result[key], block = _parse_scalar(raw, where), None
        else:
            if block is None:
                raise ConfigError(f"{where}: indented line outside a mapping")
            if raw is None or raw.startswith("#"):
                raise ConfigError(f"{where}: only one level of nesting is supported")
            if key in result[block]:
                raise ConfigError(f"{where}: duplicate key {block}.{key}")
            result[block][key] = _parse_scalar(raw, where)
    if block is not None and not result[block]:
        raise ConfigError(f"'{block}:' has no entries")
    return result


def load_config(path: Path) -> dict:
    return parse_config_text(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- validation

def mentions_test_split(value) -> bool:
    """True for 'test', 'images/test', 'x/test/y', 'test.yaml', ... (not for e.g. 'latest')."""
    return isinstance(value, str) and bool(_TEST_RE.search(value.strip()))


def _test_problems(mapping: dict, prefix: str = "") -> list:
    problems = []
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if key.lower() in ("test", "split"):
            problems.append(f"{name}: configs may not name a split (the test split is never used for training)")
        if isinstance(value, dict):
            problems += _test_problems(value, f"{name}.")
        elif mentions_test_split(value):
            problems.append(f"{name}: value {value!r} refers to the test split")
    return problems


def model_family(model_name: str) -> str | None:
    """'yolo' or 'rtdetr' from the checkpoint name; None for a name of no supported family."""
    for prefix, family in MODEL_FAMILY_PREFIXES:
        if model_name.startswith(prefix):
            return family
    return None


def validate_config(cfg: dict, config_file: str | None = None, processed_root: Path = PROCESSED_ROOT,
                    experiments_dir: Path = EXPERIMENTS_CONFIG_DIR) -> dict:
    """Raise ConfigError listing every problem; return cfg unchanged when it is valid."""
    if not isinstance(cfg, dict):
        raise ConfigError("config must be a mapping")
    problems = []
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    unknown = sorted(set(cfg) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
    if missing:
        problems.append(f"missing keys: {missing}")
    if unknown:
        problems.append(f"unknown keys: {unknown}")

    def check(key, ok, msg):
        if key in cfg and not ok(cfg[key]):
            problems.append(f"{key}: {msg} (got {cfg[key]!r})")

    is_str = lambda v: isinstance(v, str)
    check("id", lambda v: is_str(v) and _ID_RE.match(v), "must look like E001")
    check("slug", lambda v: is_str(v) and _SLUG_RE.match(v), "lowercase words joined by _")
    check("parent", lambda v: is_str(v) and (v == "none" or _ID_RE.match(v)), "'none' or an experiment id")
    check("changed", lambda v: is_str(v) and v.strip(), "describe the single variable changed")
    check("dataset_fingerprint", lambda v: is_str(v) and _SHA256_RE.match(v), "64 lowercase hex characters")
    check("model", lambda v: is_str(v) and _MODEL_RE.match(v) and model_family(v) is not None,
          "an official checkpoint name like yolo26s.pt or rtdetr-l.pt")
    check("ultralytics_version", lambda v: v == TARGET_ULTRALYTICS_VERSION,
          f"must equal the pinned version {TARGET_ULTRALYTICS_VERSION!r}")
    check("seed", lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0, "a non-negative integer")
    check("dataset_version", lambda v: is_str(v) and is_frozen(v, processed_root),
          "must be a frozen dataset version (data/processed/FROZEN.json)")

    if isinstance(cfg.get("id"), str) and cfg.get("parent") == cfg["id"]:
        problems.append("parent: an experiment cannot be its own parent")
    if isinstance(cfg.get("parent"), str) and _ID_RE.match(cfg["parent"]):
        if not list(Path(experiments_dir).glob(f"{cfg['parent']}_*.yaml")):
            problems.append(f"parent: no config {cfg['parent']}_*.yaml in {experiments_dir}")
    if config_file is not None and is_str(cfg.get("id")) and is_str(cfg.get("slug")):
        expected = f"{cfg['id']}_{cfg['slug']}.yaml"
        if Path(config_file).name != expected:
            problems.append(f"file name must be {expected} (got {Path(config_file).name})")

    args = cfg.get("train_args")
    if "train_args" in cfg and not isinstance(args, dict):
        problems.append("train_args: must be a mapping")
    elif isinstance(args, dict):
        for key in REQUIRED_TRAIN_ARGS:
            v = args.get(key)
            if not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
                problems.append(f"train_args.{key}: required positive integer (got {v!r})")
        forbidden = sorted(RUNNER_KEYS & set(args))
        if forbidden:
            problems.append(f"train_args: {forbidden} are set by the runner, not by configs")
        for key, v in args.items():
            if not (v is None or isinstance(v, (bool, int, float, str))):
                problems.append(f"train_args.{key}: only scalar values are allowed")
    problems += _test_problems(cfg)

    if problems:
        raise ConfigError("invalid experiment config:\n  - " + "\n  - ".join(problems))
    return cfg


def load_and_validate(path: Path, **kwargs) -> dict:
    return validate_config(load_config(path), config_file=Path(path).name, **kwargs)


# ---------------------------------------------------------------- Ultralytics arguments / data file

def data_yaml_keys(path: Path) -> set:
    """Top-level keys of a generated data yaml (column-0 `key:` lines)."""
    keys = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):", line)
        if m:
            keys.add(m.group(1))
    return keys


def assert_trainval_only(data_yaml: Path) -> None:
    keys = data_yaml_keys(data_yaml)
    if keys != {"path", "train", "val", "names"}:
        raise ConfigError(f"{data_yaml} must define exactly path/train/val/names (found {sorted(keys)}); "
                          "a training data file never lists the test split")


def dataset_kind(version: str, processed_root: Path = PROCESSED_ROOT) -> str:
    """'lobster' for the lobster versions (no dataset_kind in VERSION.json), else the recorded kind."""
    return (read_version_meta(version, processed_root) or {}).get("dataset_kind", "lobster")


def default_dataset_dir(version: str, processed_root: Path = PROCESSED_ROOT) -> Path:
    """data/yolo/lobster_<version> or data/yolo/tritonia_<version> (D-030)."""
    kind = dataset_kind(version, processed_root)
    return PROJECT_ROOT / "data" / "yolo" / f"{kind}_{version}"


def write_trainval_yaml(dataset_dir: Path, out_path: Path, class_names: list | None = None) -> Path:
    """Ultralytics data file for training: the same dataset view as data.yaml, without `test`."""
    root = Path(dataset_dir).resolve().as_posix().replace("'", "''")
    lines = [
        "# Generated by src/training/experiment.py for one training run: train + val only.",
        "# The test split is deliberately absent (CLAUDE.md rule 7).",
        f"path: '{root}'",
        "train: images/train",
        "val: images/val",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(class_names or CLASS_NAMES)],
        "",
    ]
    out_path = Path(out_path)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    assert_trainval_only(out_path)
    return out_path


def assert_no_test_access(args: dict) -> None:
    """Final guard on the exact kwargs passed to Ultralytics model.train()."""
    if "split" in args:
        raise ConfigError("training arguments may not contain 'split'")
    bad = [k for k, v in args.items() if k != "data" and mentions_test_split(v)]
    if bad:
        raise ConfigError(f"training arguments refer to the test split: {bad}")
    assert_trainval_only(Path(args["data"]))


def build_train_args(cfg: dict, data_yaml: Path, project: Path, name: str, smoke: bool) -> dict:
    """kwargs for Ultralytics model.train(); the config supplies hyperparameters only."""
    args = dict(cfg["train_args"])
    args.update(data=str(data_yaml), project=str(project), name=name, seed=cfg["seed"],
                exist_ok=False, val=True, save=True)
    if smoke:
        args.update(epochs=1, fraction=SMOKE_FRACTION)
    assert_no_test_access(args)
    return args


def check_output_location(path: Path, repo_root: Path, raw_root) -> Path:
    """Run folders and downloaded weights must live outside the git checkout and the raw data.

    `raw_root` is one read-only root or several (every image source, D-017).
    """
    p = Path(path)
    if not p.is_absolute():
        raise ConfigError(f"{path}: use an absolute path (e.g. /content/drive/MyDrive/uwct/runs)")
    p = p.resolve()
    raw_roots = (raw_root,) if isinstance(raw_root, (str, Path)) else tuple(raw_root)
    for root, what in ((repo_root, "the git repository"), *((r, "the read-only raw data") for r in raw_roots)):
        r = Path(root).resolve()
        if p == r or r in p.parents:
            raise ConfigError(f"{p} is inside {what} ({r}); runs and weights must stay outside it")
    return p


# ---------------------------------------------------------------- built dataset verification

def _read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_built_dataset(dataset_dir: Path, cfg: dict, processed_root: Path = PROCESSED_ROOT) -> dict:
    """Check a data/yolo/<name> view against the frozen version the config names. Read-only.

    Verifies: frozen manifest hashes (FROZEN.json), the manifests the view was built from, the
    recomputed dataset fingerprint (== build_summary.json == config), class names, that every
    built image/label belongs to an included label status and split, that no source with a
    'pending' label policy (zExtra) is present, and that the image/label folders hold exactly the
    built files. Returns the facts for the run record.

    Tritonia versions (dataset_kind 'tritonia', D-030) are checked by tritonia_dataset.verify_view
    with the same guarantees; their result also carries the dataset's class_names. Model A mixed
    versions (dataset_kind 'lobster_mixed', v4) are checked by yolo_prep.mixed.verify_view.
    """
    if dataset_kind(cfg["dataset_version"], processed_root) == T.DATASET_KIND:
        try:
            return T.verify_view(dataset_dir, cfg, processed_root)
        except (T.TritoniaError, VersionError) as e:
            raise ConfigError(str(e)) from None
    if dataset_kind(cfg["dataset_version"], processed_root) == M.DATASET_KIND:
        try:  # Model A mixed views (v4, D-031/D-032): the same guarantees, from mixed_manifest.csv
            return M.verify_view(dataset_dir, cfg, processed_root)
        except (BuildError, VersionError) as e:
            raise ConfigError(str(e)) from None
    d = Path(dataset_dir)
    summary_path = d / "build_summary.json"
    if not summary_path.is_file():
        raise ConfigError(f"{summary_path} not found; run scripts/prepare_yolo_dataset.py --version "
                          f"{cfg['dataset_version']} first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    version = cfg["dataset_version"]
    problems = []

    if summary.get("dataset_version") != version:
        problems.append(f"built version {summary.get('dataset_version')!r} != config {version!r}")
    if summary.get("version_status") != "frozen":
        problems.append(f"built from a {summary.get('version_status')!r} version; training uses frozen versions only")
    try:
        verify_frozen(version, processed_root)
    except VersionError as e:
        raise ConfigError(str(e))
    vdir = version_dir(version, processed_root)
    manifest_hashes = {
        "image_manifest_sha256": sha256_file(vdir / "image_manifest.csv"),
        "split_manifest_sha256": sha256_file(vdir / "split_manifest.csv"),
    }
    for key, value in manifest_hashes.items():
        if summary.get("inputs", {}).get(key) != value:
            problems.append(f"{key}: view was built from different manifests; rebuild it")
    fingerprint = dataset_fingerprint(d)
    if fingerprint != summary.get("dataset_fingerprint"):
        problems.append("labels or build_manifest.csv changed since the build (fingerprint mismatch)")
    if fingerprint != cfg["dataset_fingerprint"]:
        problems.append(f"dataset fingerprint {fingerprint} != config dataset_fingerprint {cfg['dataset_fingerprint']}")
    if summary.get("class_names") != {str(i): n for i, n in enumerate(CLASS_NAMES)}:
        problems.append(f"class names {summary.get('class_names')} != {CLASS_NAMES}")
    if summary.get("target_ultralytics_version") != cfg["ultralytics_version"]:
        problems.append("view was built for a different Ultralytics version")
    image_source = image_source_of(version, processed_root)
    if summary.get("image_source", DEFAULT_IMAGE_SOURCE) != image_source:
        problems.append(f"view was built from image source {summary.get('image_source')!r}, "
                        f"but {version} uses {image_source!r}")

    manifest = {r["filename"]: r for r in _read_csv(vdir / "image_manifest.csv")}
    built = _read_csv(d / "build_manifest.csv")
    counts = {s: {"images": 0, "positive": 0, "negative": 0, "instances": 0} for s in SPLITS}
    for r in built:
        m = manifest.get(r["filename"])
        if m is None:
            problems.append(f"{r['filename']}: not in the {version} image manifest")
            continue
        if MISSING_LABEL_POLICY.get(m["source_set"]) == "pending":
            problems.append(f"{r['filename']}: source {m['source_set']} is excluded (pending labels)")
        if m["label_status"] not in INCLUDED_STATUSES or r["label_status"] != m["label_status"]:
            problems.append(f"{r['filename']}: label status {r['label_status']} not allowed / not as in manifest")
        if r["split"] not in SPLITS or r["split"] != m["split"]:
            problems.append(f"{r['filename']}: split {r['split']} differs from the manifest ({m['split']})")
            continue
        if "image_sha256" in r:  # D-017: the image bytes are part of the dataset identity
            img = d / "images" / r["split"] / r["filename"]
            if not img.is_file() or sha256_file(img) != r["image_sha256"]:
                problems.append(f"{r['filename']}: image content differs from the build (sha256)")
        c = counts[r["split"]]
        c["images"] += 1
        c["positive" if m["label_status"] == "positive" else "negative"] += 1
        c["instances"] += int(m["n_instances"])
    for split in SPLITS:
        if {k: summary["splits"].get(split, {}).get(k, 0) for k in counts[split]} != counts[split]:
            problems.append(f"{split}: build_summary counts differ from build_manifest.csv")
        names = {r["filename"] for r in built if r["split"] == split}
        on_disk = {p.name for p in (d / "images" / split).iterdir()} if (d / "images" / split).is_dir() else set()
        labels = {p.name for p in (d / "labels" / split).glob("*.txt")}
        if on_disk != names:
            problems.append(f"images/{split}: {len(on_disk ^ names)} files differ from build_manifest.csv")
        if labels != {Path(n).stem + ".txt" for n in names}:
            problems.append(f"labels/{split}: label files differ from build_manifest.csv")

    if problems:
        raise ConfigError(f"dataset view {d} failed verification:\n  - " + "\n  - ".join(problems[:20]))
    return {
        "version": version,
        "status": "frozen",
        "frozen_registry_verified": True,
        "fingerprint": fingerprint,
        "image_source": image_source,
        **manifest_hashes,
        "dataset_dir": str(d.resolve()),
        "splits_used_for_training": list(TRAIN_SPLITS),
        "counts": {s: counts[s] for s in TRAIN_SPLITS},
    }
