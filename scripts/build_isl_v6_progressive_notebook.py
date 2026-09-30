from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "isl_v6_progressive_decoder.py"
OUTPUT = ROOT / "hackcessible-isl-text-to-pose-v6-progressive.ipynb"
MODULE_SOURCE = MODULE.read_text(encoding="utf-8").split('\nif __name__ == "__main__":')[0].rstrip()
MODULE_SHA256 = hashlib.sha256(MODULE.read_bytes()).hexdigest().upper()


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def check_canonicalization_fixture() -> None:
    tree = ast.parse(LOAD_V5_AND_POSES)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "canonicalize_v5")
    fixture_module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "np": np,
        "TMAX": 48,
        "POINT_INDICES": np.asarray(list(range(33)) + list(range(501, 543)), dtype=np.int64),
        "MIN_ANCHOR_CONF": 0.15,
    }
    exec(compile(ast.fix_missing_locations(fixture_module), "V6 canonicalization fixture", "exec"), namespace)
    canonicalize = namespace["canonicalize_v5"]
    pose = np.zeros((48, 576, 3), dtype=np.float32)
    pose[:, 11] = (-1.0, 0.0, 0.0)
    pose[:, 12] = (1.0, 0.0, 0.0)
    pose[:, 0] = (0.0, 2.0, 0.0)
    pose[:, 501] = (2.0, 4.0, 0.8)
    confidence = np.ones((48, 576), dtype=np.float32)
    confidence[10:15, [0, 11, 12]] = 0.0
    output, output_confidence, audit = canonicalize("fixture", pose.reshape(48, 1728), confidence)
    assert output.shape == (48, 75, 3) and output_confidence.shape == (48, 75)
    assert np.isfinite(output).all() and np.isfinite(output_confidence).all()
    assert np.allclose(output[:, 0], (0.0, 1.0, 0.0), atol=1e-6)
    assert np.allclose(output[:, 33], (1.0, 2.0, 0.4), atol=1e-6)
    assert audit == {'direct_anchor_frames': 43, 'interpolated_anchor_frames': 5,
                     'unresolved_anchor_frames': 0, 'resampled_frame_count': 48}
    assert np.allclose(output_confidence[10:15, 1], 0.5)
    try:
        canonicalize("invalid-fixture", pose.reshape(48, 1728), np.zeros_like(confidence))
    except RuntimeError as exc:
        assert "invalid-fixture" in str(exc) and "reliable canonical anchors" in str(exc)
    else:
        raise AssertionError("Canonicalization accepted a clip with no confidence-valid anchors.")


SETUP = r'''# Environment only: this experiment consumes mounted artifacts and cached poses; it never downloads pose data.
import gc, hashlib, json, math, os, random, shutil, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers'])
    from transformers import AutoModel, AutoTokenizer

SEED = 17
TMAX, POINTS, COORDS = 48, 75, 3
POSE_DIM = POINTS * COORDS
LMAX_TOKENS = 128
BATCH_SIZE, FREE_RUN_BATCH = 16, 8
MAX_EPOCHS, PATIENCE, VELOCITY_WEIGHT = 30, 5, 0.05
MIN_ANCHOR_CONF = 0.15
GROUPS = {'body': (0, 33, 0.20), 'left_hand': (33, 54, 0.40), 'right_hand': (54, 75, 0.40)}
TEXT_ENCODER_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
INPUT_ROOT = Path('/kaggle/input')
WORK_ROOT = Path('/kaggle/working')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, 'Select Kaggle GPU T4 x2 before running.'
GPU_NAMES = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
device = torch.device('cuda:0')
print('V6 environment:', torch.__version__, '| GPUs:', GPU_NAMES, '| device:', device)
print('No Hugging Face token or pose-network access is used; only public MiniLM weights may be downloaded.')
'''


LOAD_V5_AND_POSES = r'''# Load the exact V5 cohort/checkpoint and rebuild only from attached raw caches.
V5_MANIFEST_NAME = 'manifest_minilm_position_motion_v5.json'
manifest_paths = sorted(INPUT_ROOT.rglob(V5_MANIFEST_NAME))
if len(manifest_paths) != 1:
    raise RuntimeError(
        f'Expected exactly one attached V5 artifact manifest named {V5_MANIFEST_NAME}; '
        f'found {len(manifest_paths)}. Attach the completed V5 artifact dataset.'
    )
V5_MANIFEST_PATH = manifest_paths[0]
V5_ROOT = V5_MANIFEST_PATH.parent
manifest = json.loads(V5_MANIFEST_PATH.read_text(encoding='utf-8'))
assert manifest.get('model_arm') == 'minilm_position_motion_v5', f'Unexpected V5 model arm: {manifest.get("model_arm")}'
assert manifest.get('evaluation_status') and manifest.get('comparison_design'), 'V5 evaluation provenance is missing.'
print('V6 treats the inherited V5 cohort as development evidence, regardless of the V5 label wording.')
assert manifest.get('coordinate_space', '').startswith('canonical nose/shoulder-centered pose')
assert manifest.get('topology', {}).get('components') == [
    {'name': 'body', 'start': 0, 'count': 33},
    {'name': 'left_hand', 'start': 33, 'count': 21},
    {'name': 'right_hand', 'start': 54, 'count': 21},
]
split = manifest['split']
train_records = list(split['train']); test_records = list(split['validation'])
train_by_uid = {str(row['uid']): row for row in train_records}
test_by_uid = {str(row['uid']): row for row in test_records}
assert len(train_by_uid) == len(train_records) and len(test_by_uid) == len(test_records), 'Duplicate UID in frozen V5 split.'
fit_uids = [str(uid) for uid in manifest['training_uids_used']]
early_uids = [str(uid) for uid in manifest['early_stop_uids']]
test_uids = [str(row['uid']) for row in test_records]
assert len(set(fit_uids)) == len(fit_uids) and len(set(early_uids)) == len(early_uids)
assert set(fit_uids).isdisjoint(early_uids) and set(fit_uids).isdisjoint(test_uids) and set(early_uids).isdisjoint(test_uids)
assert set(fit_uids + early_uids).issubset(train_by_uid), 'V5 fit/stop roster is not contained in its exported train split.'
assert set(test_uids) == set(test_by_uid)
sample_by_uid = {str(sample['uid']): sample for sample in manifest['samples']}
assert set(sample_by_uid) == set(test_uids), 'V5 sample/reference list must exactly cover the frozen effective validation cohort.'
assert all(sample_by_uid[uid].get('counterfactual_text') for uid in test_uids), 'V5 counterfactual text missing; cannot reproduce paired shuffle.'
assert all(sample_by_uid[uid].get('predictions', {}).get('true_text') for uid in test_uids)
assert all(sample_by_uid[uid].get('predictions', {}).get('v4_model_true') for uid in test_uids)
fixed_set = manifest['fixed_unseen_set']
fixed_uids = list(map(str, fixed_set['uids']))
fixed_effective = list(map(str, fixed_set.get('effective_uids', [])))
fixed_missing = list(map(str, fixed_set.get('missing_uids', [])))
assert len(fixed_uids) == int(fixed_set['count']) and len(set(fixed_uids)) == len(fixed_uids)
assert set(fixed_effective).isdisjoint(fixed_missing) and set(fixed_effective) | set(fixed_missing) == set(fixed_uids)
assert set(fixed_effective).issubset(test_uids), 'A V5 fixed qualitative UID is not in the frozen effective test cohort.'
assert fixed_set.get('selected_before_decode') is True, 'The fixed qualitative list must retain its predecode selection provenance.'
assert int(fixed_set['selection_universe_count']) >= len(fixed_uids)
assert len(fixed_set['selection_universe_uid_sha256']) == 64
assert manifest['normalizer']['fit_uid_sha256'] == hashlib.sha256('\n'.join(sorted(fit_uids)).encode()).hexdigest()
FIT_UID_ORDER_SHA256 = hashlib.sha256('\n'.join(fit_uids).encode()).hexdigest()
TEST_UID_ORDER_SHA256 = hashlib.sha256('\n'.join(test_uids).encode()).hexdigest()
print('Frozen V5 roles:', len(fit_uids), 'fit /', len(early_uids), 'inner stop /', len(test_uids), 'paired development test.')
print('Fit UID ordered SHA256:', FIT_UID_ORDER_SHA256, '| test UID ordered SHA256:', TEST_UID_ORDER_SHA256)
print('Fixed qualitative UIDs:', len(manifest['fixed_unseen_set']['uids']), '| effective:', len(manifest['fixed_unseen_set'].get('effective_uids', [])), '| missing:', len(manifest['fixed_unseen_set'].get('missing_uids', [])))

def alias_set(records):
    return set().union(*(set(map(str, row.get('source_aliases', []))) for row in records)) if records else set()
fit_aliases = alias_set([train_by_uid[uid] for uid in fit_uids])
stop_aliases = alias_set([train_by_uid[uid] for uid in early_uids])
test_aliases = alias_set(test_records)
assert fit_aliases.isdisjoint(stop_aliases), 'V5 manifest has fit/early-stop alias leakage.'
assert fit_aliases.isdisjoint(test_aliases) and stop_aliases.isdisjoint(test_aliases), 'V5 manifest has train/test alias leakage.'
def caption_key(row):
    return str(row.get('normalized_caption_key') or ' '.join(str(row.get('text', '')).lower().split()))
fit_caption_keys = {caption_key(train_by_uid[uid]) for uid in fit_uids}
stop_caption_keys = {caption_key(train_by_uid[uid]) for uid in early_uids}
test_caption_keys = {caption_key(row) for row in test_records}
assert fit_caption_keys.isdisjoint(stop_caption_keys | test_caption_keys), 'V5 manifest has caption leakage into the fit set.'
assert stop_caption_keys.isdisjoint(test_caption_keys), 'V5 inner-stop captions overlap the paired test captions.'

checkpoint_paths = sorted(V5_ROOT.rglob('minilm_position_motion_v5.pt'))
if len(checkpoint_paths) != 1:
    raise RuntimeError(f'Expected one V5 checkpoint under {V5_ROOT}; found {len(checkpoint_paths)}.')
V5_CHECKPOINT_PATH = checkpoint_paths[0]
V5_CHECKPOINT_SHA256 = hashlib.sha256(V5_CHECKPOINT_PATH.read_bytes()).hexdigest().upper()
v5_checkpoint = torch.load(V5_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
assert v5_checkpoint.get('arm') == 'minilm_position_motion_v5'
assert v5_checkpoint.get('frozen_text_encoder') == TEXT_ENCODER_NAME
assert int(v5_checkpoint.get('time_steps', -1)) == TMAX
POINT_INDICES = np.asarray(v5_checkpoint['point_indices'], dtype=np.int64)
assert np.array_equal(POINT_INDICES, np.asarray(list(range(33)) + list(range(501, 543)), dtype=np.int64))

normalizer_path = (V5_ROOT / manifest['normalizer']['path']).resolve()
assert normalizer_path.is_file(), f'V5 fit-only normalizer is missing: {normalizer_path}'
with np.load(normalizer_path) as saved:
    v5_mean = np.asarray(saved[manifest['normalizer'].get('mean_key', 'mean')], dtype=np.float32).copy()
    v5_std = np.asarray(saved[manifest['normalizer'].get('std_key', 'std')], dtype=np.float32).copy()
assert v5_mean.shape == v5_std.shape == (POINTS, COORDS) and np.isfinite(v5_mean).all() and np.isfinite(v5_std).all()
assert np.all(v5_std >= 0.05)
assert np.allclose(v5_mean, np.asarray(v5_checkpoint['normalizer_mean'], dtype=np.float32), atol=1e-7, rtol=1e-7)
assert np.allclose(v5_std, np.asarray(v5_checkpoint['normalizer_std'], dtype=np.float32), atol=1e-7, rtol=1e-7)
assert v5_checkpoint.get('normalizer_fit_uid_sha256') == manifest['normalizer']['fit_uid_sha256']
print('V5 checkpoint SHA256:', V5_CHECKPOINT_SHA256, '| normalizer fit hash verified.')
header_paths = sorted(INPUT_ROOT.rglob('onNSvHmicw0_header.json'))
POSE_HEADER_PATH = None
POSE_HEADER = None
POSE_HEADER_SHA256 = None
if header_paths:
    header_hashes = {hashlib.sha256(path.read_bytes()).hexdigest().upper() for path in header_paths}
    if len(header_hashes) != 1:
        raise RuntimeError(f'Conflicting authoritative pose headers were attached: {[str(path) for path in header_paths]}')
    POSE_HEADER_PATH = header_paths[0]
    POSE_HEADER_SHA256 = next(iter(header_hashes))
    POSE_HEADER = json.loads(POSE_HEADER_PATH.read_text(encoding='utf-8'))
    print('Pose header preserved:', POSE_HEADER_PATH, '| SHA256:', POSE_HEADER_SHA256)
else:
    print('Pose header was not attached; V6 will record its metadata as unavailable.')

def named_cache_dirs(name):
    return sorted({path.resolve() for path in INPUT_ROOT.rglob(name) if path.is_dir()})
v2_cache_dirs = named_cache_dirs('isign_pose_cache_711ff3100b0f')
v5_cache_dirs = named_cache_dirs('isign_pose_cache_v5')
if not v2_cache_dirs or not v5_cache_dirs:
    raise RuntimeError(
        'Required pose cache inputs are not attached. Attach the verified full V2 cache '
        '(directory isign_pose_cache_711ff3100b0f) and the completed V5 raw fresh cache '
        '(directory isign_pose_cache_v5). V6 never re-decodes pose data from the network.'
    )
cache_dirs = v2_cache_dirs + v5_cache_dirs
cache_paths_by_uid = defaultdict(list)
for cache_dir in cache_dirs:
    for path in cache_dir.glob('*.npz'):
        cache_paths_by_uid[path.stem].append(path)
required_uids = set(fit_uids) | set(early_uids) | set(test_uids)
missing_cache_uids = sorted(required_uids - set(cache_paths_by_uid))
if missing_cache_uids:
    raise RuntimeError(
        f'Raw cache coverage is incomplete ({len(missing_cache_uids)}/{len(required_uids)} required role UIDs). '
        f'No resampling/reselection is allowed. First missing UIDs: {missing_cache_uids[:20]}'
    )

def load_raw_cache(uid):
    paths = sorted(cache_paths_by_uid[str(uid)])
    candidates = []
    for path in paths:
        with np.load(path) as saved:
            if not {'pose', 'confidence'}.issubset(saved.files):
                continue
            pose = np.asarray(saved['pose'], dtype=np.float32).copy()
            confidence = np.asarray(saved['confidence'], dtype=np.float32).copy()
            if pose.shape != (TMAX, 1728) or confidence.shape != (TMAX, 576):
                continue
            raw_count = int(np.asarray(saved['raw_frame_count']).item()) if 'raw_frame_count' in saved.files else None
            raw_fps = float(np.asarray(saved['fps']).item()) if 'fps' in saved.files else None
        candidates.append((path, pose, confidence, raw_count, raw_fps))
    if not candidates:
        raise RuntimeError(f'No raw V2/V5 cached pose with expected shapes for UID {uid}: {paths}')
    _, pose, confidence, raw_count, raw_fps = candidates[0]
    for duplicate_path, duplicate_pose, duplicate_conf, duplicate_count, duplicate_fps in candidates[1:]:
        if not np.array_equal(pose, duplicate_pose) or not np.array_equal(confidence, duplicate_conf) or raw_count != duplicate_count or raw_fps != duplicate_fps:
            raise RuntimeError(f'Conflicting raw cache copies for UID {uid}: {[str(item[0]) for item in candidates]}')
    return pose, confidence, raw_count, raw_fps

def canonicalize_v5(uid_for_error, raw_pose, raw_confidence):
    raw_pose = np.asarray(raw_pose, dtype=np.float32).reshape(TMAX, 576, 3)
    raw_confidence = np.asarray(raw_confidence, dtype=np.float32).reshape(TMAX, 576)
    finite = np.isfinite(raw_pose).all(axis=-1)
    safe = np.nan_to_num(raw_pose, nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.clip(np.nan_to_num(raw_confidence, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    selected_pose = safe[:, POINT_INDICES, :].copy()
    selected_conf = conf[:, POINT_INDICES].copy() * finite[:, POINT_INDICES]
    center = (safe[:, 11, :] + safe[:, 12, :]) * 0.5
    nose = safe[:, 0, :]
    scale = np.linalg.norm(nose[:, :2] - center[:, :2], axis=-1)
    anchors = (conf[:, 0] >= MIN_ANCHOR_CONF) & (conf[:, 11] >= MIN_ANCHOR_CONF) & (conf[:, 12] >= MIN_ANCHOR_CONF)
    anchors &= finite[:, 0] & finite[:, 11] & finite[:, 12] & np.isfinite(scale) & (scale > 1e-4)
    valid = np.flatnonzero(anchors)
    if len(valid) < 2:
        raise RuntimeError(f'V5-effective UID {uid_for_error} has only {len(valid)} reliable canonical anchors.')
    frame_ix = np.arange(TMAX)
    center_interp = np.stack([np.interp(frame_ix, valid, center[valid, axis]) for axis in range(3)], axis=-1)
    scale_interp = np.exp(np.interp(frame_ix, valid, np.log(scale[valid].clip(1e-4))))
    selected_conf *= np.where(anchors, 1.0, 0.5).astype(np.float32)[:, None]
    canonical = (selected_pose - center_interp[:, None, :]) / scale_interp[:, None, None]
    canonical = np.nan_to_num(canonical, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    info = {'direct_anchor_frames': int(len(valid)), 'interpolated_anchor_frames': int(TMAX-len(valid)),
            'unresolved_anchor_frames': 0, 'resampled_frame_count': TMAX}
    return canonical, selected_conf.astype(np.float32), info

sample_uids = fit_uids + early_uids + test_uids
if len(set(sample_uids)) != len(sample_uids):
    raise RuntimeError('A UID appears in multiple frozen V5 roles.')
canonical_pose_by_uid, canonical_conf_by_uid = {}, {}
raw_meta_by_uid, anchor_audit_by_uid = {}, {}
for n, uid in enumerate(sample_uids, start=1):
    raw_pose, raw_conf, raw_count, raw_fps = load_raw_cache(uid)
    canonical_pose, canonical_conf, anchor_info = canonicalize_v5(uid, raw_pose, raw_conf)
    canonical_pose_by_uid[uid] = canonical_pose
    canonical_conf_by_uid[uid] = canonical_conf
    anchor_audit_by_uid[uid] = anchor_info
    rec = test_by_uid[uid] if uid in test_by_uid else train_by_uid[uid]
    raw_meta_by_uid[uid] = {
        'raw_frame_count': raw_count if raw_count is not None else rec.get('raw_frame_count'),
        'fps': raw_fps if raw_fps is not None else rec.get('fps'), 'resampling_policy': rec.get('resampling_policy'),
        'raw_count_provenance': 'cache_npz' if raw_count is not None else ('v5_manifest' if rec.get('raw_frame_count') is not None else 'unavailable'),
        'fps_provenance': 'cache_npz' if raw_fps is not None else ('v5_manifest' if rec.get('fps') is not None else 'unavailable'),
    }
    if uid in test_by_uid:
        sample = sample_by_uid[uid]
        ref_path = (V5_ROOT / sample['reference']['path']).resolve()
        if not ref_path.is_file() or not ref_path.is_relative_to(V5_ROOT.resolve()):
            raise RuntimeError(f'V5 test reference missing or outside artifact root: {ref_path}')
        with np.load(ref_path) as ref:
            expected_pose = np.asarray(ref[sample['reference'].get('pose_key', 'pose')], dtype=np.float32)
            expected_conf = np.asarray(ref[sample['reference'].get('confidence_key', 'confidence')], dtype=np.float32)
        if expected_pose.shape != (TMAX, POINTS, COORDS) or expected_conf.shape != (TMAX, POINTS):
            raise RuntimeError(f'Bad canonical V5 reference shape for {uid}: {expected_pose.shape}, {expected_conf.shape}')
        if not np.allclose(canonical_pose, expected_pose, atol=2e-5, rtol=2e-5):
            raise RuntimeError(f'Cached raw pose does not reproduce V5 canonical reference for {uid}.')
        if not np.allclose(canonical_conf, expected_conf, atol=2e-6, rtol=2e-6):
            raise RuntimeError(f'Cached raw confidence does not reproduce V5 reference confidence for {uid}.')
    if n % 512 == 0 or n == len(sample_uids):
        print(f'Reconstructed V5 canonical rows {n}/{len(sample_uids)}', flush=True)

fit_pose = np.stack([canonical_pose_by_uid[uid] for uid in fit_uids])
fit_conf = np.stack([canonical_conf_by_uid[uid] for uid in fit_uids])
weights = fit_conf[..., None]
den = weights.sum(axis=(0, 1))
recomputed_mean = ((fit_pose * weights).sum(axis=(0, 1)) / np.maximum(den, 1e-6)).astype(np.float32)
variance = (((fit_pose - recomputed_mean[None, None]) ** 2) * weights).sum(axis=(0, 1)) / np.maximum(den, 1e-6)
recomputed_std = np.sqrt(np.maximum(variance, 0)).clip(0.05).astype(np.float32)
if not np.allclose(recomputed_mean, v5_mean, atol=2e-5, rtol=2e-5) or not np.allclose(recomputed_std, v5_std, atol=2e-5, rtol=2e-5):
    raise RuntimeError('Reconstructed fit-only statistics differ from the V5 normalizer; refusing a nonpaired comparison.')
del fit_pose, fit_conf, weights, den, variance
gc.collect()
all_uids = sample_uids
uid_to_index = {uid: i for i, uid in enumerate(all_uids)}
canonical_poses = np.stack([canonical_pose_by_uid[uid] for uid in all_uids]).astype(np.float32)
canonical_confs = np.stack([canonical_conf_by_uid[uid] for uid in all_uids]).astype(np.float32)
targets = np.nan_to_num((canonical_poses - v5_mean[None, None]) / v5_std[None, None], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
CONFIDENCE = canonical_confs
print('V5 raw cache coverage and canonical reference parity verified for all roles.')
print('Raw frame count/fps provenance preserved when present; no timing values synthesized.')
'''


TOKEN_CACHE = r'''# Compute and persist genuine frozen MiniLM token states in a ragged cache.
all_role_texts = [str((train_by_uid[uid] if uid in train_by_uid else test_by_uid[uid])['text']) for uid in all_uids]
counterfactual_text_by_uid = {uid: str(sample_by_uid[uid]['counterfactual_text']) for uid in test_uids}
all_texts = sorted(set(all_role_texts + list(counterfactual_text_by_uid.values())),
                   key=lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest())
assert all(text.strip() for text in all_texts)
text_to_ix = {text: i for i, text in enumerate(all_texts)}
uid_text_ix = {uid: text_to_ix[text] for uid, text in zip(all_uids, all_role_texts)}
uid_counterfactual_ix = {uid: text_to_ix[text] for uid, text in counterfactual_text_by_uid.items()}
texts_sha256 = hashlib.sha256('\n'.join(all_texts).encode('utf-8')).hexdigest()
role_uid_sha256 = hashlib.sha256('\n'.join(all_uids).encode('utf-8')).hexdigest()
TOKEN_CACHE_DIR = WORK_ROOT / 'isl_v6_minilm_token_cache'
TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
hidden_path = TOKEN_CACHE_DIR / 'hidden_fp16.npy'
offsets_path = TOKEN_CACHE_DIR / 'offsets.npy'
lengths_path = TOKEN_CACHE_DIR / 'lengths.npy'
meta_path = TOKEN_CACHE_DIR / 'metadata.json'
token_meta = {
    'model': TEXT_ENCODER_NAME, 'representation': 'AutoModel.last_hidden_state; contextual token vectors, no mean pooling',
    'max_length': LMAX_TOKENS, 'text_sha256': texts_sha256, 'all_role_uid_order_sha256': role_uid_sha256,
    'unique_text_count': len(all_texts), 'dtype': 'float16', 'hidden_size': 384,
}
cache_hit = False
if all(path.is_file() for path in (hidden_path, offsets_path, lengths_path, meta_path)):
    old_meta = json.loads(meta_path.read_text(encoding='utf-8'))
    cache_hit = old_meta == token_meta
if cache_hit:
    token_hidden = np.load(hidden_path, mmap_mode='r')
    token_offsets = np.load(offsets_path)
    token_lengths = np.load(lengths_path)
    assert token_hidden.shape[1:] == (384,) and len(token_lengths) == len(all_texts) and len(token_offsets) == len(all_texts) + 1
    print('Reusing exact V6 ragged token cache:', token_hidden.shape, '|', token_meta)
else:
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_NAME, use_fast=True)
    encoded_all = tokenizer(all_texts, padding=False, truncation=True, max_length=LMAX_TOKENS, add_special_tokens=True)
    token_lengths = np.asarray([len(ids) for ids in encoded_all['input_ids']], dtype=np.int32)
    assert (token_lengths > 0).all() and int(token_lengths.max()) <= LMAX_TOKENS
    token_offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(token_lengths, dtype=np.int64)])
    token_hidden = np.lib.format.open_memmap(hidden_path, mode='w+', dtype=np.float16, shape=(int(token_offsets[-1]), 384))
    encoder = AutoModel.from_pretrained(TEXT_ENCODER_NAME).to(device).eval()
    for parameter in encoder.parameters(): parameter.requires_grad_(False)
    with torch.inference_mode():
        for start in range(0, len(all_texts), 64):
            end = min(start + 64, len(all_texts))
            batch = tokenizer(all_texts[start:end], padding=True, truncation=True, max_length=LMAX_TOKENS, add_special_tokens=True, return_tensors='pt')
            batch = {key: value.to(device) for key, value in batch.items()}
            states = encoder(**batch).last_hidden_state
            assert states.shape[0] == end-start and states.shape[-1] == 384 and torch.isfinite(states).all()
            cpu_states = states.to(dtype=torch.float16, device='cpu').numpy()
            for local, text_ix in enumerate(range(start, end)):
                length = int(token_lengths[text_ix])
                token_hidden[int(token_offsets[text_ix]):int(token_offsets[text_ix+1])] = cpu_states[local, :length]
            if end % 512 == 0 or end == len(all_texts):
                print(f'Frozen MiniLM token vectors {end}/{len(all_texts)}', flush=True)
    token_hidden.flush()
    np.save(offsets_path, token_offsets); np.save(lengths_path, token_lengths)
    meta_path.write_text(json.dumps(token_meta, indent=2), encoding='utf-8')
    del encoder, tokenizer, encoded_all
    gc.collect(); torch.cuda.empty_cache()
    token_hidden = np.load(hidden_path, mmap_mode='r')
print('Ragged MiniLM last_hidden_state:', token_hidden.shape, '| token lengths p50/p95/max:',
      int(np.median(token_lengths)), int(np.percentile(token_lengths, 95)), int(token_lengths.max()),
      '| text hash:', texts_sha256)
'''


DATA_AND_LOSS = r'''# Build padded minibatches from ragged text-state storage and reproduce the fixed V5 objective.
from torch.utils.data import DataLoader, Dataset

class UIDDataset(Dataset):
    def __init__(self, uids): self.uids = list(uids)
    def __len__(self): return len(self.uids)
    def __getitem__(self, index): return self.uids[index]

def collate_uids(batch_uids, counterfactual=False):
    ix = [uid_counterfactual_ix[uid] if counterfactual else uid_text_ix[uid] for uid in batch_uids]
    lengths = [int(token_lengths[item]) for item in ix]
    width = max(lengths)
    hidden = np.zeros((len(ix), width, 384), dtype=np.float16)
    mask = np.zeros((len(ix), width), dtype=np.bool_)
    for row, (text_ix, length) in enumerate(zip(ix, lengths)):
        hidden[row, :length] = token_hidden[int(token_offsets[text_ix]):int(token_offsets[text_ix+1])]
        mask[row, :length] = True
    ids = np.asarray([uid_to_index[uid] for uid in batch_uids], dtype=np.int64)
    target = targets[ids]
    confidence = CONFIDENCE[ids]
    return (torch.from_numpy(hidden), torch.from_numpy(mask),
            torch.from_numpy(target), torch.from_numpy(confidence), list(batch_uids))

def make_loader(uids, batch_size, shuffle=False):
    return DataLoader(UIDDataset(uids), batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=True, drop_last=False, collate_fn=collate_uids)

def grouped_terms(prediction, target, confidence):
    prediction = prediction.reshape(-1, TMAX, POINTS, COORDS)
    target = target.reshape(-1, TMAX, POINTS, COORDS)
    confidence = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
    terms = {}
    for name, (start, end, weight) in GROUPS.items():
        pos_err = (prediction[:, :, start:end] - target[:, :, start:end]).square()
        pos_w = confidence[:, :, start:end, None].expand_as(pos_err)
        pos_num, pos_den = (pos_err * pos_w).sum(), pos_w.sum().clamp_min(1.0)
        vel_err = ((prediction[:, 1:, start:end] - prediction[:, :-1, start:end])
                   - (target[:, 1:, start:end] - target[:, :-1, start:end])).square()
        pair_conf = torch.minimum(confidence[:, 1:, start:end], confidence[:, :-1, start:end])
        vel_w = pair_conf[..., None].expand_as(vel_err)
        vel_num, vel_den = (vel_err * vel_w).sum(), vel_w.sum().clamp_min(1.0)
        terms[name] = (pos_num, pos_den, vel_num, vel_den, weight)
    position = sum(weight * (pn / pd) for pn, pd, _, _, weight in terms.values())
    velocity = sum(weight * (vn / vd) for _, _, vn, vd, weight in terms.values())
    return position, velocity, terms

def score_free_run(uids, counterfactual=False, batch_size=FREE_RUN_BATCH, keep_predictions=False):
    parallel.eval()
    totals = {name: np.zeros(4, dtype=np.float64) for name in GROUPS}
    motion = {name: np.zeros(3, dtype=np.float64) for name in GROUPS}
    predictions = {}
    with torch.inference_mode():
        for start in range(0, len(uids), batch_size):
            batch_uids = list(uids[start:start+batch_size])
            hidden, mask, target, confidence, _ = collate_uids(batch_uids, counterfactual=counterfactual)
            hidden = hidden.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True); confidence = confidence.to(device, non_blocking=True)
            # No target pose is passed into the autoregressive branch.
            prediction = parallel(hidden, mask, None, 'generate')
            assert prediction.shape == target.shape and torch.isfinite(prediction).all()
            terms = grouped_terms(prediction, target, confidence)[2]
            p = prediction.reshape(-1, TMAX, POINTS, COORDS)
            y = target.reshape(-1, TMAX, POINTS, COORDS)
            for name, (pn, pd, vn, vd, _) in terms.items():
                totals[name] += [pn.item(), pd.item(), vn.item(), vd.item()]
                lo, hi, _ = GROUPS[name]
                pair_conf = torch.minimum(confidence[:, 1:, lo:hi], confidence[:, :-1, lo:hi])
                p_speed = torch.linalg.vector_norm(p[:, 1:, lo:hi] - p[:, :-1, lo:hi], dim=-1)
                y_speed = torch.linalg.vector_norm(y[:, 1:, lo:hi] - y[:, :-1, lo:hi], dim=-1)
                motion[name] += [float((p_speed*pair_conf).sum()), float((y_speed*pair_conf).sum()), float(pair_conf.sum())]
            if keep_predictions:
                raw = (prediction.detach().cpu().numpy().astype(np.float32) * v5_std[None,None] + v5_mean[None,None])
                for uid, value in zip(batch_uids, raw): predictions[uid] = value
    pos_group = {name: totals[name][0]/max(totals[name][1],1.0) for name in GROUPS}
    vel_group = {name: totals[name][2]/max(totals[name][3],1.0) for name in GROUPS}
    pos = sum(GROUPS[name][2]*pos_group[name] for name in GROUPS)
    vel = sum(GROUPS[name][2]*vel_group[name] for name in GROUPS)
    pred_motion = {name: motion[name][0]/max(motion[name][2],1.0) for name in GROUPS}
    ref_motion = {name: motion[name][1]/max(motion[name][2],1.0) for name in GROUPS}
    ratios = {name: pred_motion[name]/max(ref_motion[name],1e-8) for name in GROUPS}
    result = {'position':float(pos),'velocity':float(vel),'selection_score':float(pos+VELOCITY_WEIGHT*vel),
              'position_by_group':pos_group,'velocity_by_group':vel_group,'prediction_motion_by_group':pred_motion,
              'reference_motion_by_group':ref_motion,'motion_ratio_by_group':ratios}
    return (result, predictions) if keep_predictions else result

print('Objective locked to V5: group weights body=.20, left hand=.40, right hand=.40; velocity coefficient=.05.')
print('Frozen roles:', len(fit_uids), 'fit /', len(early_uids), 'free-run early stop /', len(test_uids), 'paired test.')
'''


TRAIN = r'''# Train with teacher forcing, but choose checkpoints only by free-running inner-stop loss.
class ParallelDecoder(nn.Module):
    def __init__(self, decoder):
        super().__init__(); self.decoder = decoder
    def forward(self, text_hidden, text_mask, target_pose=None, mode='generate'):
        if mode == 'teacher':
            if target_pose is None: raise ValueError('Teacher mode requires target poses.')
            return self.decoder.forward_teacher(text_hidden, text_mask, target_pose)
        if mode == 'generate':
            if target_pose is not None: raise ValueError('Free-run mode must not receive target poses.')
            return self.decoder.generate(text_hidden, text_mask, frames=TMAX)
        raise ValueError(f'Unknown decoder mode: {mode}')

decoder_core = ProgressivePoseDecoder(text_dim=384, model_dim=256, num_heads=8, num_layers=2,
                                      feedforward_dim=1024, dropout=0.1, max_frames=TMAX,
                                      points=POINTS, coordinates=COORDS).to(device)
parallel = nn.DataParallel(ParallelDecoder(decoder_core), device_ids=[0, 1])

# Verify both GPUs and target-free inference with a measured in-notebook probe.
probe_uids = fit_uids[:BATCH_SIZE]
assert len(probe_uids) == BATCH_SIZE
probe_h, probe_m, probe_y, probe_c, _ = collate_uids(probe_uids)
probe_h=probe_h.to(device); probe_m=probe_m.to(device); probe_y=probe_y.to(device); probe_c=probe_c.to(device)
seen_devices = set()
hook = decoder_core.text_projection.register_forward_pre_hook(lambda module, inputs: seen_devices.add(inputs[0].device.index))
optimizer = torch.optim.AdamW(parallel.parameters(), lr=3e-4, weight_decay=1e-4)
parallel.train(); optimizer.zero_grad(set_to_none=True)
t0=time.perf_counter()
probe_pred=parallel(probe_h,probe_m,probe_y,'teacher')
assert probe_pred.shape == probe_y.shape and torch.isfinite(probe_pred).all()
probe_pos,probe_vel,_=grouped_terms(probe_pred,probe_y,probe_c)
probe_loss=probe_pos+VELOCITY_WEIGHT*probe_vel
assert torch.isfinite(probe_loss)
probe_loss.backward()
probe_grad=torch.nn.utils.clip_grad_norm_(parallel.parameters(),1.0)
assert torch.isfinite(probe_grad) and float(probe_grad)>0
optimizer.zero_grad(set_to_none=True)
teacher_probe_seconds=time.perf_counter()-t0
parallel.eval()
t0=time.perf_counter()
with torch.inference_mode():
    free_probe=parallel(probe_h,probe_m,None,'generate')
free_probe_seconds=time.perf_counter()-t0
hook.remove()
assert free_probe.shape == probe_y.shape and torch.isfinite(free_probe).all()
assert {0,1}.issubset(seen_devices), f'Two-GPU forward/backward probe failed; devices seen={sorted(seen_devices)}'
train_batches=math.ceil(len(fit_uids)/BATCH_SIZE)
stop_batches=math.ceil(len(early_uids)/FREE_RUN_BATCH)
print('T4x2 probe passed:', tuple(probe_pred.shape), '| grad norm',round(float(probe_grad),5),
      '| teacher forward/backward seconds',round(teacher_probe_seconds,2),'| 48-frame free-run seconds',round(free_probe_seconds,2))
print('Measured rough epoch estimate: teacher fit',
      round(train_batches*teacher_probe_seconds,1),'sec + free-run stop',
      round(stop_batches*free_probe_seconds,1),'sec; batch/length variability makes this approximate.')

train_loader = make_loader(fit_uids,BATCH_SIZE,shuffle=True)
signature_payload={'decoder_module_sha256':'MODULE_SHA256_VALUE','fit_uid_order_sha256':FIT_UID_ORDER_SHA256,
                   'stop_uid_order_sha256':hashlib.sha256('\n'.join(early_uids).encode()).hexdigest(),
                   'test_uid_order_sha256':TEST_UID_ORDER_SHA256,'normalizer_fit_sha256':manifest['normalizer']['fit_uid_sha256'],
                   'model_dim':256,'layers':2,'heads':8,'max_epochs':MAX_EPOCHS,'velocity_weight':VELOCITY_WEIGHT,
                   'token_text_sha256':texts_sha256}
RUN_SIGNATURE=hashlib.sha256(json.dumps(signature_payload,sort_keys=True).encode()).hexdigest()
RESUME_PATH=WORK_ROOT/'isl_v6_progressive_resume.pt'
history=[]; start_epoch=0; stale=0; best_score=float('inf'); best_epoch=0; best_state=None
if RESUME_PATH.is_file():
    saved=torch.load(RESUME_PATH,map_location=device,weights_only=False)
    if saved.get('signature')==RUN_SIGNATURE:
        decoder_core.load_state_dict(saved['current_state']); optimizer.load_state_dict(saved['optimizer'])
        history=saved['history']; start_epoch=int(saved['epoch']); stale=int(saved['stale'])
        best_score=float(saved['best_score']); best_epoch=int(saved['best_epoch']); best_state=saved['best_state']
        print('Resuming exact V6 signature after epoch',start_epoch,'| best free-run selection',best_score,flush=True)
    else:
        print('Ignoring a prior resume file with a different experiment signature.')

for epoch in range(start_epoch+1,MAX_EPOCHS+1):
    parallel.train(); started=time.perf_counter(); train_rows=[]
    for text_hidden,text_mask,target,confidence,_uids in train_loader:
        text_hidden=text_hidden.to(device,non_blocking=True); text_mask=text_mask.to(device,non_blocking=True)
        target=target.to(device,non_blocking=True); confidence=confidence.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction=parallel(text_hidden,text_mask,target,'teacher')
        position,velocity,_=grouped_terms(prediction,target,confidence)
        loss=position+VELOCITY_WEIGHT*velocity
        if not torch.isfinite(loss): raise RuntimeError('Non-finite teacher-forced training objective.')
        loss.backward(); torch.nn.utils.clip_grad_norm_(parallel.parameters(),1.0); optimizer.step()
        train_rows.append((float(position.detach()),float(velocity.detach())))
    stop=score_free_run(early_uids)
    row={'epoch':epoch,'fit_teacher_position':float(np.mean([v[0] for v in train_rows])),
         'fit_teacher_velocity':float(np.mean([v[1] for v in train_rows])),**{'early_free_'+k:v for k,v in stop.items()},
         'seconds':round(time.perf_counter()-started,1)}
    history.append(row)
    if stop['selection_score']<best_score:
        best_score=stop['selection_score']; best_epoch=epoch; stale=0
        best_state={key:value.detach().cpu().clone() for key,value in decoder_core.state_dict().items()}
    else: stale+=1
    torch.save({'signature':RUN_SIGNATURE,'epoch':epoch,'stale':stale,
                'current_state':{key:value.detach().cpu().clone() for key,value in decoder_core.state_dict().items()},
                'optimizer':optimizer.state_dict(),'history':history,'best_score':best_score,
                'best_epoch':best_epoch,'best_state':best_state},RESUME_PATH)
    print(f'V6 epoch {epoch:02d}/{MAX_EPOCHS}: teacher_pos={row["fit_teacher_position"]:.5f} '
          f'teacher_vel={row["fit_teacher_velocity"]:.5f} free_pos={stop["position"]:.5f} '
          f'free_vel={stop["velocity"]:.5f} score={stop["selection_score"]:.5f} '
          f'motion={stop["motion_ratio_by_group"]} seconds={row["seconds"]}',flush=True)
    if stale>=PATIENCE: break
if best_state is None: raise RuntimeError('V6 produced no free-run-selected checkpoint.')
decoder_core.load_state_dict(best_state); parallel.eval()
print('Selected V6 epoch by free-run internal objective:',best_epoch,'| score:',best_score)
'''


EXPORT = r'''# Score the frozen paired development cohort and write an auditable artifact package.
test_true_metrics,true_predictions=score_free_run(test_uids,counterfactual=False,keep_predictions=True)
test_shuffle_metrics,shuffle_predictions=score_free_run(test_uids,counterfactual=True,keep_predictions=True)

def metric_for_raw_predictions(prediction, reference, confidence):
    prediction=np.asarray(prediction,dtype=np.float32); reference=np.asarray(reference,dtype=np.float32)
    confidence=np.asarray(confidence,dtype=np.float32)
    zpred=(prediction-v5_mean[None,None])/v5_std[None,None]
    zref=(reference-v5_mean[None,None])/v5_std[None,None]
    out={}; balanced_pos=0.0; balanced_vel=0.0
    for name,(lo,hi,weight) in GROUPS.items():
        conf=confidence[:,:,lo:hi,None]
        err=(zpred[:,:,lo:hi]-zref[:,:,lo:hi])**2
        raw=(prediction[:,:,lo:hi]-reference[:,:,lo:hi])**2
        den=max(float(conf.sum()*3),1e-8)
        pos=float((err*conf).sum()/den); balanced_pos+=weight*pos
        pair=np.minimum(confidence[:,1:,lo:hi],confidence[:,:-1,lo:hi])
        vel_err=(np.diff(zpred[:,:,lo:hi],axis=1)-np.diff(zref[:,:,lo:hi],axis=1))**2
        vel_den=max(float(pair.sum()*3),1e-8)
        vel=float((vel_err*pair[...,None]).sum()/vel_den); balanced_vel+=weight*vel
        ps=np.linalg.norm(np.diff(prediction[:,:,lo:hi],axis=1),axis=-1)
        rs=np.linalg.norm(np.diff(reference[:,:,lo:hi],axis=1),axis=-1)
        md=max(float(pair.sum()),1e-8)
        pm=float((ps*pair).sum()/md); rm=float((rs*pair).sum()/md)
        out[name]={'normalized_mse':pos,'normalized_velocity_mse':vel,'canonical_rmse':float(np.sqrt((raw*conf).sum()/den)),
                   'prediction_motion':pm,'reference_motion':rm,'motion_ratio':pm/max(rm,1e-8)}
    out['group_balanced_normalized_mse']=balanced_pos
    out['group_balanced_normalized_velocity_mse']=balanced_vel
    out['selection_score']=balanced_pos+VELOCITY_WEIGHT*balanced_vel
    return out

reference_by_uid={uid:canonical_pose_by_uid[uid] for uid in test_uids}
confidence_by_uid={uid:canonical_conf_by_uid[uid] for uid in test_uids}
def load_v5_array(uid,key):
    sample=sample_by_uid[uid]
    path=(V5_ROOT/sample['predictions'][key]).resolve()
    if not path.is_file() or not path.is_relative_to(V5_ROOT.resolve()):
        raise RuntimeError(f'Paired baseline {key} missing for {uid}: {path}')
    array=np.asarray(np.load(path),dtype=np.float32)
    if array.shape!=(TMAX,POINTS,COORDS) or not np.isfinite(array).all():
        raise RuntimeError(f'Bad paired baseline shape/content for {uid}/{key}: {array.shape}')
    return array

v5_true=np.stack([load_v5_array(uid,'true_text') for uid in test_uids])
v5_shuffle=np.stack([load_v5_array(uid,'shuffled_text') for uid in test_uids])
v4_true=np.stack([load_v5_array(uid,'v4_model_true') for uid in test_uids])
references=np.stack([reference_by_uid[uid] for uid in test_uids])
test_conf=np.stack([confidence_by_uid[uid] for uid in test_uids])
v6_true=np.stack([true_predictions[uid] for uid in test_uids])
v6_shuffle=np.stack([shuffle_predictions[uid] for uid in test_uids])
paired_metrics={
    'v6_true_text':metric_for_raw_predictions(v6_true,references,test_conf),
    'v6_shuffled_text':metric_for_raw_predictions(v6_shuffle,references,test_conf),
    'v5_true_text':metric_for_raw_predictions(v5_true,references,test_conf),
    'v5_shuffled_text':metric_for_raw_predictions(v5_shuffle,references,test_conf),
    'v4_true_text':metric_for_raw_predictions(v4_true,references,test_conf),
}
print('Paired cohort position / velocity / selection objective:',
      {key:(value['group_balanced_normalized_mse'],value['group_balanced_normalized_velocity_mse'],value['selection_score'])
       for key,value in paired_metrics.items()})
print('V6 free-run position/velocity:',test_true_metrics['position'],test_true_metrics['velocity'],
      '| motion ratios:',test_true_metrics['motion_ratio_by_group'])
print('V6 shuffled position/velocity:',test_shuffle_metrics['position'],test_shuffle_metrics['velocity'])

EXPORT_ROOT=WORK_ROOT/'isl_v6_progressive_export'
if EXPORT_ROOT.exists(): shutil.rmtree(EXPORT_ROOT)
for name in ('references','predictions/v6','predictions/v5','predictions/v4','checkpoints','diagnostics'):
    (EXPORT_ROOT/name).mkdir(parents=True,exist_ok=True)
shutil.copy2(V5_MANIFEST_PATH,EXPORT_ROOT/'manifest_v5_source.json')
shutil.copy2(normalizer_path,EXPORT_ROOT/'normalizer.npz')
shutil.copy2(V5_CHECKPOINT_PATH,EXPORT_ROOT/'v5_checkpoint_reference.pt')
if POSE_HEADER_PATH is not None:
    shutil.copy2(POSE_HEADER_PATH,EXPORT_ROOT/'onNSvHmicw0_header.json')
np.save(EXPORT_ROOT/'point_indices.npy',POINT_INDICES)
np.save(EXPORT_ROOT/'token_cache_lengths.npy',token_lengths)
np.save(EXPORT_ROOT/'token_cache_offsets.npy',token_offsets)
shutil.copy2(meta_path,EXPORT_ROOT/'token_cache_metadata.json')
np.save(EXPORT_ROOT/'diagnostics'/'v6_test_true.npy',v6_true)
np.save(EXPORT_ROOT/'diagnostics'/'v6_test_shuffled.npy',v6_shuffle)
np.save(EXPORT_ROOT/'diagnostics'/'v5_test_true.npy',v5_true)
np.save(EXPORT_ROOT/'diagnostics'/'v4_test_true.npy',v4_true)

sample_records=[]
for i,uid in enumerate(test_uids):
    sample=sample_by_uid[uid]
    slug=''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in uid)
    np.savez_compressed(EXPORT_ROOT/'references'/(slug+'.npz'),pose=references[i],confidence=test_conf[i])
    v6true=EXPORT_ROOT/'predictions'/'v6'/(slug+'_true.npy')
    v6shuffle=EXPORT_ROOT/'predictions'/'v6'/(slug+'_shuffled.npy')
    v5path=EXPORT_ROOT/'predictions'/'v5'/(slug+'_true.npy')
    v5shufpath=EXPORT_ROOT/'predictions'/'v5'/(slug+'_shuffled.npy')
    v4path=EXPORT_ROOT/'predictions'/'v4'/(slug+'_true.npy')
    np.save(v6true,v6_true[i]); np.save(v6shuffle,v6_shuffle[i])
    np.save(v5path,v5_true[i]); np.save(v5shufpath,v5_shuffle[i]); np.save(v4path,v4_true[i])
    row=test_by_uid[uid]
    sample_records.append({
        'uid':uid,'source_video_id':row.get('source_video_id'),'source_aliases':row.get('source_aliases',[]),
        'source_uid_rule':row.get('source_uid_rule'),'source_video_id_provenance':row.get('source_video_id_provenance'),
        'normalized_caption_key':row.get('normalized_caption_key'),'text':str(row['text']),
        'counterfactual_text':counterfactual_text_by_uid[uid],'counterfactual_source_uid':sample.get('counterfactual_source_uid'),
        'raw_frame_count':raw_meta_by_uid[uid]['raw_frame_count'],'fps':raw_meta_by_uid[uid]['fps'],
        'raw_count_provenance':raw_meta_by_uid[uid]['raw_count_provenance'],'fps_provenance':raw_meta_by_uid[uid]['fps_provenance'],
        'resampled_frame_count':TMAX,'resampling_policy':raw_meta_by_uid[uid]['resampling_policy'],
        'canonicalization_anchor_audit':anchor_audit_by_uid[uid],
        'reference':{'path':'references/'+slug+'.npz','pose_key':'pose','confidence_key':'confidence'},
        'predictions':{'true_text':'predictions/v6/'+slug+'_true.npy','shuffled_text':'predictions/v6/'+slug+'_shuffled.npy',
                       'v5_model_true':'predictions/v5/'+slug+'_true.npy','v5_model_shuffled':'predictions/v5/'+slug+'_shuffled.npy',
                       'v4_model_true':'predictions/v4/'+slug+'_true.npy'},
    })

normalizer_hash=manifest['normalizer']['fit_uid_sha256']
v6_manifest={
    'schema_version':1,'dataset':manifest.get('dataset'),'model_arm':'progressive_token_minilm_v6',
    'evaluation_status':'paired_development_benchmark_not_final_test; inherited V5 UID-inferred source caveats',
    'comparison_design':'V6 ProgressiveTransformer is trained on the exact V5 fit/early-stop UID roles and compared on the exact same V5 fresh development cohort against frozen V5 and V4 outputs. This tests a combined sequence-decoder factor under the V5 data/split policy; this cohort is development evidence, not an untouched final test.',
    'selected_for_follow_up':True,'selection_metric':'free-running group-balanced normalized position MSE + 0.05 x free-running group-balanced normalized velocity MSE on the frozen V5 inner-stop UID list only',
    'objective':manifest['objective'],'training_uids_used':fit_uids,'early_stop_uids':early_uids,
    'fit_uid_order_sha256':FIT_UID_ORDER_SHA256,'early_stop_uid_order_sha256':hashlib.sha256('\n'.join(early_uids).encode()).hexdigest(),
    'test_uid_order_sha256':TEST_UID_ORDER_SHA256,'v5_source_manifest_sha256':hashlib.sha256(V5_MANIFEST_PATH.read_bytes()).hexdigest().upper(),
    'v5_checkpoint_sha256':V5_CHECKPOINT_SHA256,'progressive_module_sha256':'MODULE_SHA256_VALUE',
    'text_encoder':TEXT_ENCODER_NAME,'text_representation':'frozen AutoModel.last_hidden_state token features; contextual 384D token vectors; no sentence mean pooling',
    'tokenization':{'max_length':LMAX_TOKENS,'truncation':'longest_first','ragged_cache_dtype':'float16','ragged_cache_shape':list(token_hidden.shape),'unique_text_count':len(all_texts),'text_sha256':texts_sha256},
    'model_config':{'model_dim':256,'num_heads':8,'num_layers':2,'feedforward_dim':1024,'dropout':0.1,'decoder':'causal autoregressive TransformerDecoder cross-attending to frozen text token states','teacher_forcing':'BOS followed by target poses shifted right by one frame','selection_and_inference':'48-frame free run; target poses are not passed to generate'},
    'fit_checkpoint':{'epoch':best_epoch,'free_run_selection_score':best_score,'history':history,'gpus':GPU_NAMES,
                      'gpu_probe':{'teacher_forward_backward_seconds':teacher_probe_seconds,'free_run_batch_seconds':free_probe_seconds,'batch_size':BATCH_SIZE,'free_run_batch_size':len(probe_uids)}},
    'paired_metrics':paired_metrics,'v6_true_text_free_run_metrics':test_true_metrics,'v6_shuffled_text_free_run_metrics':test_shuffle_metrics,
    'paired_condition_comparisons':[{'name':'V6 vs V5 on frozen development cohort','left_condition':'true_text','right_condition':'v5_model_true','scope':'all_samples'},
                                    {'name':'V6 vs V4 on frozen development cohort','left_condition':'true_text','right_condition':'v4_model_true','scope':'all_samples'}],
    'split':{key:value for key,value in split.items()},
    'fixed_unseen_set':manifest['fixed_unseen_set'],
    'topology':manifest['topology'],'normalizer':{'path':'normalizer.npz','mean_key':'mean','std_key':'std','fit_split':'train','parent_fit_role':'V5_gradient_fit',
        'fit_uid_sha256':normalizer_hash,'verified_against':'V5 checkpoint and exact fit-only recomputation'},
    'coordinate_space':manifest['coordinate_space'],'canonicalization':manifest['canonicalization'],
    'pose_header':{'path':'onNSvHmicw0_header.json' if POSE_HEADER_PATH is not None else None,'sha256':POSE_HEADER_SHA256,'metadata':POSE_HEADER},
    'resampling':manifest.get('resampling'),'raw_metadata_note':'Raw frame counts and FPS are copied from cache or V5 records when present. Unavailable V2 original lengths and FPS remain null; no values were inferred.',
    'samples':sample_records,'metrics_path':'metrics_summary.json',
    'follow_up_requirements':['The 500 V5 cohort is now development evidence because it informed decoder selection; reserve a new untouched source/caption cohort after architecture choice.','Current output covers 75 body and hand points only; final scope still needs face and nonmanual expression representation.','Movement and coordinate metrics do not prove sign correctness. Fluent ISL signers must review a fixed randomized sample for meaning, grammar, and naturalness.'],
    'license':manifest.get('license','iSign CC-BY-NC-SA-4.0; research/non-commercial use only')
}
(EXPORT_ROOT/'manifest_progressive_v6.json').write_text(json.dumps(v6_manifest,indent=2),encoding='utf-8')
(EXPORT_ROOT/'metrics_summary.json').write_text(json.dumps({
    'selected_epoch':best_epoch,'internal_free_run_selection_score':best_score,'history':history,
    'test_metrics':paired_metrics,'v6_true_text_free_run':test_true_metrics,'v6_shuffled_text_free_run':test_shuffle_metrics
},indent=2),encoding='utf-8')
checkpoint={'format':'Hackcessible iSign V6 progressive causal decoder','state_dict':best_state,
            'config':signature_payload,'normalizer_mean':v5_mean,'normalizer_std':v5_std,
            'normalizer_fit_uid_sha256':normalizer_hash,'point_indices':POINT_INDICES,'time_steps':TMAX,
            'text_encoder':TEXT_ENCODER_NAME,'text_representation':'last_hidden_state token vectors; frozen encoder',
            'progressive_module_sha256':'MODULE_SHA256_VALUE','license':manifest.get('license')}
torch.save(checkpoint,EXPORT_ROOT/'checkpoints'/'progressive_token_minilm_v6.pt')
readme=(
    'V6 is a paired development experiment using the exact V5 fit, early-stop, and 500-row fresh development roles. '
    'The decoder consumes genuine frozen MiniLM contextual token states. Training is teacher-forced with a right shift; '
    'checkpoint selection and test generation are autoregressive free runs. The V5 fit-only normalizer was recomputed and verified. '
    'The saved 48-frame output contains 75 body and hand points only. Face and nonmanual grammar cues are absent. '
    'The test set is no longer untouched after architecture comparison; a later signer-reviewed final cohort is required. '
    'Metrics measure coordinates/motion and do not establish ISL meaning or grammar. '
    'The iSign dataset is CC-BY-NC-SA-4.0 for research/non-commercial use.\n'
)
(EXPORT_ROOT/'README.txt').write_text(readme,encoding='utf-8')
archive_path=shutil.make_archive(str(WORK_ROOT/'hackcessible_isl_v6_progressive_artifacts'),'zip',root_dir=EXPORT_ROOT)
print('V6 export:',archive_path,'bytes=',Path(archive_path).stat().st_size)
print('V6 true/shuffle objective:',test_true_metrics['selection_score'],test_shuffle_metrics['selection_score'])
print('Paired V5/V4 true-text position MSE:',paired_metrics['v5_true_text']['group_balanced_normalized_mse'],
      paired_metrics['v4_true_text']['group_balanced_normalized_mse'])
'''


def main() -> None:
    setup = SETUP
    module_cell = MODULE_SOURCE + "\n\nprint('Progressive decoder module SHA256:', '" + MODULE_SHA256 + "')\nprint('Module CPU contract:', run_cpu_smoke_checks())\n"
    train = TRAIN.replace('MODULE_SHA256_VALUE', MODULE_SHA256)
    export = EXPORT.replace('MODULE_SHA256_VALUE', MODULE_SHA256)
    cells = [
        md('''# Hackcessible ISL text-to-pose — V6 Progressive decoder\n\nThis is a separate, frozen architecture experiment. It consumes the completed V5 manifest/checkpoint and the verified V2 plus V5 raw pose caches. It does not download or decode pose archives. It reuses the exact V5 fit, inner-stop, and fresh development-test UID lists and fit-only normalizer; every test reference is checked against the raw cache before training.\n\nV6 feeds contextual MiniLM token states to a causal autoregressive pose decoder. Training uses a one-frame right shift; early stopping and final predictions use free-running generation. The 500-row V5 cohort is development evidence, not an untouched final test. Output is still 75 body/hand points and does not include facial/nonmanual grammar signals or signer intelligibility review. iSign is CC-BY-NC-SA-4.0 for research/non-commercial use.'''),
        code(setup),
        code(LOAD_V5_AND_POSES),
        code(TOKEN_CACHE),
        code(module_cell),
        code(DATA_AND_LOSS),
        code(train),
        code(export),
    ]
    # Bind the exact frozen module hash into notebook experiment signatures and outputs.
    cells[5]["source"] = [line.replace("MODULE_SHA256_VALUE", MODULE_SHA256) for line in cells[5]["source"]]
    notebook = {
        "cells": cells,
        "metadata": {
            "kaggle": {"accelerator": "GPU T4 x2", "isInternetEnabled": True, "language": "python"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"V6 notebook cell {index}")
    check_canonicalization_fixture()
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Notebook SHA256: {hashlib.sha256(OUTPUT.read_bytes()).hexdigest().upper()}")
    print(f"Embedded progressive module SHA256: {MODULE_SHA256}")
    print(f"Code cells parsed: {sum(cell['cell_type']=='code' for cell in notebook['cells'])}")
    print("Canonicalization fixture: shape, finite coordinates, nose/shoulder scale, gap interpolation, confidence downweight, and no-anchor rejection passed.")


if __name__ == "__main__":
    main()
