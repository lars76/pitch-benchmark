import io
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from constants import (
    AUDIO_SUBTYPE,
    CAL_DIR,
    CAL_MIN_GROUPS,
    CAL_SHARE,
    CARRIER,
    CLIPS_NAME,
    CONSENSUS,
    DATASET_NAME,
    DEGRADATION,
    DESIGN_SEED,
    FACTORS,
    HOP_SIZE,
    IDENTITY_PANEL,
    MAX_CLIPS,
    MAX_SECONDS,
    OUTPUT,
    PANEL_STAGES,
    POOLS,
    POOLS_NAME,
    RAW,
    RENDER_SEED,
    RENDERS_NAME,
    ROOMS,
    SALT_SAMPLING,
    SAMPLE_RATE,
    SCORED_PANELS,
    TEST_DIR,
    VOICED_THRESHOLD,
)
from corpora import EVAL, TRAIN, get_pitch_dataset
from corpora.base import base_index
from corpora.laryngograph import LaryngographSpeechDataset
from corpora.select import build_eval_source
from degradation import build_pool, load_rooms, load_scene
from stages import render_clip


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


def calibration_groups(groups):
    order = sorted({str(g) for g in groups})
    return set() if len(order) < CAL_MIN_GROUPS else set(order[::CAL_SHARE])


def pool_entry(entry, keys):
    return {k: entry[k] for k in keys if k in entry}


def _as_array(value):
    return value.detach().cpu().numpy().reshape(-1)


def _clip_audio(sample):
    return _as_array(sample["audio"]).astype(np.float32)


def audio_dir(out_dir, panel=None):
    out = Path(out_dir) / "audio"
    return out if panel is None else out / panel


def role_root(out_dir, role=None):
    return Path(out_dir) if role is None else Path(out_dir) / role


def as_stored(audio, sample_rate):
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="FLAC", subtype=AUDIO_SUBTYPE)
    buf.seek(0)
    decoded, _ = sf.read(buf, dtype="float32")
    return decoded


def panels_for(role):
    return SCORED_PANELS if role == CAL_DIR else tuple(PANEL_STAGES)


def write_clip_audio(directory, index, audio, sample_rate):
    sf.write(str(directory / f"{index:07d}.flac"), audio, sample_rate,
             format="FLAC", subtype=AUDIO_SUBTYPE)


def write_clips(ds, out_dir, corpus, split=False, master_seed=RENDER_SEED,
                scene=None, rooms=None):
    out = Path(out_dir)
    entry = to_json_safe({"fmin": ds.fmin, "fmax": ds.fmax})
    groups = [str(ds.get_group(i)) for i in range(len(ds))]
    cal = calibration_groups(groups) if split else set()
    if split:
        entry["cal_groups"] = len(cal)
    roles = [(CAL_DIR if g in cal else TEST_DIR) if split else None for g in groups]
    renders = {}
    for role in dict.fromkeys(roles):
        root = role_root(out, role)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        for panel in (panels_for(role) if split else (None,)):
            audio_dir(root, panel).mkdir(parents=True, exist_ok=True)
    for i, role in enumerate(roles):
        sample = ds[i]
        audio = _clip_audio(sample)
        pitch = _as_array(sample["pitch"]).astype(np.float32)
        conf = sample.get("pitch_conf")
        trusted = (_as_array(conf) >= VOICED_THRESHOLD if conf is not None
                   else np.ones(pitch.shape, dtype=bool))
        index = base_index(ds, i)
        root = role_root(out, role)
        np.savez_compressed(str(root / "labels" / f"{index:07d}.npz"),
                            pitch=pitch, trusted=trusted)
        if not split:
            write_clip_audio(audio_dir(root), index, audio, ds.sample_rate)
            continue
        audio = as_stored(audio, ds.sample_rate)
        for panel in panels_for(role):
            rendered, record = render_clip(
                audio, ds.sample_rate, pitch > 0, ds.hop_size, panel=panel,
                corpus=str(corpus), clip_idx=index, master_seed=int(master_seed),
                scene=scene, rooms=rooms)
            write_clip_audio(audio_dir(root, panel), index, rendered, ds.sample_rate)
            renders.setdefault(f"{index:07d}", {})[panel] = record
    if split:
        write_json_doc(out / RENDERS_NAME, renders)
        write_json_doc(out / CLIPS_NAME,
                       {f"{base_index(ds, i):07d}": {"group": groups[i], "role": roles[i]}
                        for i in range(len(ds))})
    n_cal = sum(role == CAL_DIR for role in roles)
    print(f"{corpus}: {len(groups)} clips ({n_cal} held out)", flush=True)
    return entry


def write_json_doc(path, doc):
    Path(path).write_text(json.dumps(to_json_safe(doc), indent=2) + "\n")


def write_dataset_json(split_dir, corpora):
    write_json_doc(Path(split_dir) / DATASET_NAME,
                   {"hop_size": HOP_SIZE,
                    "sample_rate": SAMPLE_RATE,
                    "build": {"max_clips": MAX_CLIPS, "max_seconds": MAX_SECONDS,
                              "voiced_threshold": VOICED_THRESHOLD,
                              "cal_share": CAL_SHARE, "cal_min_groups": CAL_MIN_GROUPS,
                              "seeds": {"render": RENDER_SEED, "design": DESIGN_SEED,
                                        "sampling_salt": SALT_SAMPLING}},
                    "design": {"identity": IDENTITY_PANEL,
                               "factors": list(FACTORS),
                               "carrier": list(CARRIER),
                               "panels": {p: list(v) for p, v in PANEL_STAGES.items()}},
                    "corpora": corpora})


def corpus_roots(raw):
    roots = {name: raw / name for name in (*TRAIN, *EVAL)}
    roots["SpeechSynth"] = raw / "SpeechSynth" / "speechsynth.pt"
    return roots


def loader_kwargs(name, consensus):
    if issubclass(get_pitch_dataset(name), LaryngographSpeechDataset):
        return {"consensus_dir": consensus}
    return {}


def build_train(roots, output, consensus, raw):
    for name, options in TRAIN.items():
        source = get_pitch_dataset(name)(root_dir=str(roots[name]), hop_size=HOP_SIZE,
                                         sample_rate=SAMPLE_RATE,
                                         **loader_kwargs(name, consensus), **options)
        write_clips(source, output / "train" / name, name)
    augmentation = output / "train" / "augmentation"
    augmentation.mkdir(parents=True, exist_ok=True)
    for entry in (*POOLS, *ROOMS):
        if entry["kind"] != "colored":
            build_pool(entry, augmentation, raw)
    write_json_doc(augmentation / POOLS_NAME,
                   {"pools": [pool_entry(e, ("name", "kind", "point", "label_safe"))
                              for e in POOLS],
                    "rooms": [pool_entry(e, ("name",)) for e in ROOMS]})


def build_eval(roots, output, consensus, raw):
    scene, rooms = load_scene(raw), load_rooms(raw)
    windows = {}
    for name in EVAL:
        source = build_eval_source(name, str(roots[name]),
                                   max_clips=MAX_CLIPS, max_seconds=MAX_SECONDS,
                                   **loader_kwargs(name, consensus))
        windows[name] = write_clips(source, output / "eval" / name, name, split=True,
                                    master_seed=RENDER_SEED, scene=scene, rooms=rooms)
    write_dataset_json(output / "eval", windows)


def main():
    if len(sys.argv) != 1:
        raise SystemExit("build.py takes no arguments; see README.md")
    roots = corpus_roots(RAW)
    required = list(roots.values()) + [RAW / name for name in DEGRADATION] + [CONSENSUS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required inputs:\n" + "\n".join(missing))
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    build_train(roots, OUTPUT, CONSENSUS, RAW)
    build_eval(roots, OUTPUT, CONSENSUS, RAW)


if __name__ == "__main__":
    main()
