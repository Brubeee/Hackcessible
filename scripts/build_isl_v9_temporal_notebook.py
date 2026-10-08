"""Build the aggregate-only Hackcessible ISL V9 ranked experiment notebook.

The notebook verifies mounted V5 artifacts and cached pose parity, pins and
fingerprints MiniLM, runs CPU and synchronized two-GPU preflight, trains with
an objective-aligned caption ranking loss, then evaluates the reused development
cohort and geometry gates. It exports only aggregate metrics and a gated
checkpoint; it never writes per-row pose predictions or identifiers.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "isl_v9_temporal_ranked_transformer.py"
OUTPUT = ROOT / "hackcessible-isl-text-to-pose-v9-ranked.ipynb"
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
    exec(compile(ast.fix_missing_locations(fixture_module), "V9 canonicalization fixture", "exec"), namespace)
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


SETUP = r'''# Environment setup: consumes mounted artifacts and cached poses; never downloads pose data.
import gc, hashlib, json, math, os, random, shutil, sys, time, zipfile
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
BATCH_SIZE, EVAL_BATCH = 16, 16
MAX_EPOCHS, PATIENCE = 25, 5
VELOCITY_WEIGHT, PHASE_WEIGHT, DYNAMIC_WEIGHT = 1.0, 0.25, 1.0
RANK_WEIGHT, RANK_MARGIN = 0.20, 0.01
NEAR_DUPLICATE_COSINE_THRESHOLD = 0.92
MIN_ANCHOR_CONF = 0.15
BONE_MIN_EDGE_SUPPORT = 100
BONE_MIN_CLIP_GROUP_SUPPORT = 96
BONE_REFERENCE_P95_MIN = 1e-3
GROUPS = {'body': (0, 33, 0.20), 'left_hand': (33, 54, 0.40), 'right_hand': (54, 75, 0.40)}
TEXT_ENCODER_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TEXT_ENCODER_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
INPUT_ROOT = Path('/kaggle/input')
WORK_ROOT = Path('/kaggle/working')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, 'Select Kaggle GPU T4 x2 before running.'
GPU_NAMES = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
device = torch.device('cuda:0')
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
    torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.matmul.allow_tf32 = False
ATTENTION_BACKEND_CONFIG = {
    'scaled_dot_product_attention': 'math_only', 'flash': torch.backends.cuda.flash_sdp_enabled(),
    'memory_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(),
    'math': torch.backends.cuda.math_sdp_enabled(),
    'cudnn': torch.backends.cuda.cudnn_sdp_enabled() if hasattr(torch.backends.cuda, 'cudnn_sdp_enabled') else None,
    'tf32': torch.backends.cuda.matmul.allow_tf32, 'model_dtype': 'float32',
}
assert ATTENTION_BACKEND_CONFIG['math'] and not ATTENTION_BACKEND_CONFIG['flash'] and not ATTENTION_BACKEND_CONFIG['memory_efficient']
BUILDER_SHA256 = 'BUILDER_SHA256_VALUE'
print('V9 builder SHA256:', BUILDER_SHA256)
print('V9 environment:', torch.__version__, '| GPUs:', GPU_NAMES, '| device:', device)
print('V9 attention backend and precision:', ATTENTION_BACKEND_CONFIG)
'''

LOAD_V5_AND_POSES = r'''# Load the exact V5 cohort/checkpoint and rebuild only from attached raw caches.
V5_MANIFEST_NAME = 'manifest_minilm_position_motion_v5.json'
manifest_paths = sorted(INPUT_ROOT.rglob(V5_MANIFEST_NAME))
if not manifest_paths:
    artifact_zips = sorted(INPUT_ROOT.rglob('hackcessible_isl_v5_artifacts.zip'))
    if len(artifact_zips) != 1:
        raise RuntimeError(
            f'Expected the V5 artifact tree or exactly one hackcessible_isl_v5_artifacts.zip; found {len(artifact_zips)} ZIPs.'
        )
    artifact_sha = hashlib.sha256(artifact_zips[0].read_bytes()).hexdigest().upper()
    extracted_root = WORK_ROOT / f'v5_artifacts_{artifact_sha[:12]}'
    extracted_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(artifact_zips[0], 'r') as archive:
        members = archive.infolist()
        for member in members:
            member_path = Path(member.filename)
            if member_path.is_absolute() or '..' in member_path.parts:
                raise RuntimeError(f'Unsafe path in attached V5 artifact ZIP: {member.filename}')
        archive.extractall(extracted_root)
    manifest_paths = sorted(extracted_root.rglob(V5_MANIFEST_NAME))
    print('Extracted attached V5 artifact ZIP:', artifact_zips[0], '| SHA256:', artifact_sha)
if len(manifest_paths) != 1:
    raise RuntimeError(
        f'Expected exactly one attached V5 artifact manifest named {V5_MANIFEST_NAME}; '
        f'found {len(manifest_paths)}. Attach the completed V5 artifact dataset.'
    )
V5_MANIFEST_PATH = manifest_paths[0]
V5_ROOT = V5_MANIFEST_PATH.parent
manifest = json.loads(V5_MANIFEST_PATH.read_text(encoding='utf-8'))
V5_METRICS_PATH = V5_ROOT / 'metrics_summary.json'
V5_SELECTION_PATH = V5_ROOT / 'selection_manifest_predecode.json'
if not V5_METRICS_PATH.is_file() or not V5_SELECTION_PATH.is_file():
    raise RuntimeError('V5 metrics_summary.json and selection_manifest_predecode.json are required for raw-cache coverage and exclusion audits.')
v5_metrics = json.loads(V5_METRICS_PATH.read_text(encoding='utf-8'))
V5_MANIFEST_SHA256 = hashlib.sha256(V5_MANIFEST_PATH.read_bytes()).hexdigest().upper()
V5_METRICS_SHA256 = hashlib.sha256(V5_METRICS_PATH.read_bytes()).hexdigest().upper()
v5_selection = json.loads(V5_SELECTION_PATH.read_text(encoding='utf-8'))
fresh_train_selected_records = list(v5_selection['fresh_train'])
fresh_test_selected_records = list(v5_selection['fresh_test'])
fresh_train_selected_uids = [str(row['uid']) for row in fresh_train_selected_records]
fresh_test_selected_uids = [str(row['uid']) for row in fresh_test_selected_records]
fresh_selected_uids = fresh_train_selected_uids + fresh_test_selected_uids
fresh_train_selected_uid_set = set(fresh_train_selected_uids)
fresh_test_selected_uid_set = set(fresh_test_selected_uids)
assert len(fresh_train_selected_uids) == int(v5_metrics['fresh_train_rows_selected']) == 8000
assert len(fresh_test_selected_uids) == int(v5_metrics['fresh_test_rows_selected']) == 500
assert len(set(fresh_selected_uids)) == len(fresh_selected_uids) == 8500, 'V5 predecode selection must contain exactly 8,500 unique fresh UIDs.'
assert float(v5_selection.get('minimum_fresh_coverage', 0.0)) >= 0.98
fresh_selected_uid_set = set(fresh_selected_uids)
v5_exclusions = list(v5_metrics.get('load_and_canonicalization_exclusions', []))
fresh_decode_failure_uids = {
    str(row['uid']) for row in v5_exclusions
    if str(row.get('reason', '')).startswith('fresh_pose_decode_error:')
}
fresh_selection_by_uid = {str(row['uid']): row for row in fresh_train_selected_records + fresh_test_selected_records}
fresh_exclusion_by_uid = {str(row['uid']): row for row in v5_exclusions if str(row.get('uid')) in fresh_selected_uid_set}
assert fresh_decode_failure_uids.issubset(fresh_selected_uid_set), 'A logged fresh decode failure is outside the fixed predecode roster.'
v5_exclusion_anchor_accounting = []
for row in v5_exclusions:
    frames = int(row.get('resampled_frame_count') or TMAX)
    direct = int(row.get('direct_anchor_frames') or 0)
    fallback = int(row.get('fallback_anchor_frames') or 0)
    interpolated = int(row.get('interpolated_anchor_frames') or 0)
    unresolved = int(row.get('unresolved_anchor_frames') or 0)
    reason = str(row.get('reason', ''))
    rejected = row.get('anchor_mode') == 'unusable' or 'fewer_than_two_reliable' in reason or reason.startswith('fresh_pose_decode_error:') or reason.startswith('selected_v2_pose_missing') or reason.startswith('selected_v2_cache_error:')
    record = dict(row)
    record.update({
        'observed_anchor_frames': direct + fallback,
        'canonicalized_output_frames': 0 if rejected else max(0, frames - unresolved),
        'clip_rejected': bool(rejected),
        'logged_unresolved_frames_semantics': 'clip-level rejected/unavailable coverage; may overlap observed direct/fallback anchor counts' if rejected else 'unresolved frames in accepted clip',
        'anchor_counts_form_disjoint_frame_partition': not rejected,
    })
    if rejected:
        assert unresolved == frames, f'Rejected V5 clip {row.get("uid")} must retain its full clip-level unresolved count.'
        assert direct + fallback == int(row.get('total_anchor_frames') or 0), f'Observed-anchor counts disagree for rejected V5 clip {row.get("uid")}.'
    else:
        assert direct + fallback + interpolated + unresolved == frames, f'Accepted V5 anchor counts do not partition frames for {row.get("uid")}.'
    v5_exclusion_anchor_accounting.append(record)
assert manifest.get('model_arm') == 'minilm_position_motion_v5', f'Unexpected V5 model arm: {manifest.get("model_arm")}'
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
assert len(train_by_uid) == len(train_records) and len(test_by_uid) == len(test_records)
fit_uids = [str(uid) for uid in manifest['training_uids_used']]
early_uids = [str(uid) for uid in manifest['early_stop_uids']]
test_uids = [str(row['uid']) for row in test_records]
assert len(set(fit_uids)) == len(fit_uids) and len(set(early_uids)) == len(early_uids)
assert set(fit_uids).isdisjoint(early_uids) and set(fit_uids).isdisjoint(test_uids) and set(early_uids).isdisjoint(test_uids)
assert set(fit_uids + early_uids).issubset(train_by_uid)
assert set(test_uids) == set(test_by_uid)
sample_by_uid = {str(sample['uid']): sample for sample in manifest['samples']}
assert set(sample_by_uid) == set(test_uids)
assert all(sample_by_uid[uid].get('counterfactual_text') for uid in test_uids)
assert all(sample_by_uid[uid].get('predictions', {}).get('true_text') for uid in test_uids)
assert all(sample_by_uid[uid].get('predictions', {}).get('train_frame_mean') for uid in test_uids)
assert all(sample_by_uid[uid].get('predictions', {}).get('v4_model_true') for uid in test_uids)
for uid in test_uids:
    for prediction_key in ('true_text','train_frame_mean'):
        baseline_path = (V5_ROOT / sample_by_uid[uid]['predictions'][prediction_key]).resolve()
        assert baseline_path.is_relative_to(V5_ROOT.resolve()) and baseline_path.is_file(), \
            f'Missing or unsafe paired V5 baseline cache for role={prediction_key}.'
fixed_set = manifest['fixed_unseen_set']
fixed_uids = list(map(str, fixed_set['uids']))
fixed_effective = list(map(str, fixed_set.get('effective_uids', [])))
fixed_missing = list(map(str, fixed_set.get('missing_uids', [])))
assert len(fixed_uids) == int(fixed_set['count']) and len(set(fixed_uids)) == len(fixed_uids)
assert set(fixed_effective).isdisjoint(fixed_missing) and set(fixed_effective) | set(fixed_missing) == set(fixed_uids)
assert set(fixed_effective).issubset(test_uids)
assert manifest['normalizer']['fit_uid_sha256'] == hashlib.sha256('\n'.join(sorted(fit_uids)).encode()).hexdigest()
FIT_UID_ORDER_SHA256 = hashlib.sha256('\n'.join(fit_uids).encode()).hexdigest()
TEST_UID_ORDER_SHA256 = hashlib.sha256('\n'.join(test_uids).encode()).hexdigest()
print('Frozen V5 roles:', len(fit_uids), 'fit /', len(early_uids), 'inner stop /', len(test_uids), 'paired test.')

def caption_key(row):
    return str(row.get('normalized_caption_key') or ' '.join(str(row.get('text', '')).lower().split()))
fit_caption_keys = {caption_key(train_by_uid[uid]) for uid in fit_uids}
stop_caption_keys = {caption_key(train_by_uid[uid]) for uid in early_uids}
test_caption_keys = {caption_key(row) for row in test_records}
assert fit_caption_keys.isdisjoint(stop_caption_keys | test_caption_keys)
assert stop_caption_keys.isdisjoint(test_caption_keys)
rank_role_uids = fit_uids + early_uids
assert all(train_by_uid[uid].get('source_video_cluster') and train_by_uid[uid].get('source_video_id') for uid in rank_role_uids)
assert all(isinstance(train_by_uid[uid].get('source_aliases'), list) for uid in rank_role_uids)
SOURCE_CLUSTER_COUNT = len({str(train_by_uid[uid]['source_video_cluster']) for uid in rank_role_uids})
def role_source_aliases(uids, rows):
    aliases = set()
    for uid in uids:
        row = rows[uid]
        aliases.update(str(value) for value in row.get('source_aliases', []) if value)
        aliases.update(str(value) for value in (row.get('source_video_cluster'), row.get('source_video_id')) if value)
    return aliases
fit_source_aliases = role_source_aliases(fit_uids, train_by_uid)
stop_source_aliases = role_source_aliases(early_uids, train_by_uid)
test_source_aliases = role_source_aliases(test_uids, test_by_uid)
assert fit_source_aliases.isdisjoint(stop_source_aliases), 'Fit and inner-stop source clusters/aliases overlap.'
assert fit_source_aliases.isdisjoint(test_source_aliases), 'Fit and development source clusters/aliases overlap.'
assert stop_source_aliases.isdisjoint(test_source_aliases), 'Inner-stop and development source clusters/aliases overlap.'
assert len({str(test_by_uid[uid].get('source_video_cluster')) for uid in test_uids}) == len(test_uids), \
    'Development bootstrap assumes one row per source-video cluster.'
SPLIT_ROLE_SHA256 = hashlib.sha256(json.dumps({
    'fit': hashlib.sha256('\n'.join(sorted(fit_uids)).encode()).hexdigest(),
    'inner_stop': hashlib.sha256('\n'.join(sorted(early_uids)).encode()).hexdigest(),
    'development_only': hashlib.sha256('\n'.join(sorted(test_uids)).encode()).hexdigest(),
}, sort_keys=True, separators=(',', ':')).encode()).hexdigest().upper()
print('Caption/source ranking metadata:', len(rank_role_uids), 'fit+stop rows;', SOURCE_CLUSTER_COUNT, 'source clusters; aliases available for all rows.')
print('Explicit disjoint roles verified: fit / inner-stop / 498-row V7/V8 development-only cohort; split-role SHA256:', SPLIT_ROLE_SHA256)
v5_unassigned_train_uids = sorted(set(train_by_uid) - set(fit_uids) - set(early_uids))
v5_unassigned_train_rows = []
for uid in v5_unassigned_train_uids:
    row = train_by_uid[uid]
    key = caption_key(row)
    reason = 'excluded_from_fit_by_normalized_caption_overlap'
    v5_unassigned_train_rows.append({'uid': uid, 'text': row.get('text'), 'reason': reason})

checkpoint_paths = sorted(V5_ROOT.rglob('minilm_position_motion_v5.pt'))
if len(checkpoint_paths) != 1:
    raise RuntimeError(f'Expected one V5 checkpoint under {V5_ROOT}; found {len(checkpoint_paths)}.')
V5_CHECKPOINT_PATH = checkpoint_paths[0]
V5_CHECKPOINT_SHA256 = hashlib.sha256(V5_CHECKPOINT_PATH.read_bytes()).hexdigest().upper()
v5_checkpoint = torch.load(V5_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
POINT_INDICES = np.asarray(v5_checkpoint['point_indices'], dtype=np.int64)

normalizer_path = (V5_ROOT / manifest['normalizer']['path']).resolve()
assert normalizer_path.is_file(), f'V5 fit-only normalizer is missing: {normalizer_path}'
with np.load(normalizer_path) as saved:
    v5_mean = np.asarray(saved[manifest['normalizer'].get('mean_key', 'mean')], dtype=np.float32).copy()
    v5_std = np.asarray(saved[manifest['normalizer'].get('std_key', 'std')], dtype=np.float32).copy()
assert v5_mean.shape == v5_std.shape == (POINTS, COORDS) and np.isfinite(v5_mean).all() and np.isfinite(v5_std).all()
assert np.all(v5_std >= 0.05)
assert np.allclose(v5_mean, np.asarray(v5_checkpoint['normalizer_mean'], dtype=np.float32), atol=1e-7, rtol=1e-7)
assert np.allclose(v5_std, np.asarray(v5_checkpoint['normalizer_std'], dtype=np.float32), atol=1e-7, rtol=1e-7)

header_paths = sorted(INPUT_ROOT.rglob('onNSvHmicw0_header.json'))
POSE_HEADER_PATH = header_paths[0] if header_paths else None
POSE_HEADER_SHA256 = hashlib.sha256(POSE_HEADER_PATH.read_bytes()).hexdigest().upper() if POSE_HEADER_PATH else None
POSE_HEADER = json.loads(POSE_HEADER_PATH.read_text(encoding='utf-8')) if POSE_HEADER_PATH else None

def named_cache_dirs(name):
    return sorted({path.resolve() for path in INPUT_ROOT.rglob(name) if path.is_dir()})
v2_cache_dirs = named_cache_dirs('isign_pose_cache_711ff3100b0f')
v5_cache_dirs = named_cache_dirs('isign_pose_cache_v5')
if not v2_cache_dirs or not v5_cache_dirs:
    raise RuntimeError('Required pose cache inputs are not attached.')
cache_dirs = v2_cache_dirs + v5_cache_dirs

cache_paths_by_uid = defaultdict(list)
for cache_dir in cache_dirs:
    for path in cache_dir.glob('*.npz'):
        cache_paths_by_uid[path.stem].append(path)
required_uids = set(fit_uids) | set(early_uids) | set(test_uids)
missing_cache_uids = sorted(required_uids - set(cache_paths_by_uid))
if missing_cache_uids:
    raise RuntimeError(f'Raw cache coverage incomplete: {len(missing_cache_uids)} missing.')

def load_raw_cache(uid):
    paths = sorted(cache_paths_by_uid[str(uid)])
    for path in paths:
        with np.load(path) as saved:
            if {'pose', 'confidence'}.issubset(saved.files):
                pose = np.asarray(saved['pose'], dtype=np.float32).copy()
                confidence = np.asarray(saved['confidence'], dtype=np.float32).copy()
                if pose.shape == (TMAX, 1728) and confidence.shape == (TMAX, 576):
                    raw_count = int(np.asarray(saved['raw_frame_count']).item()) if 'raw_frame_count' in saved.files else None
                    raw_fps = float(np.asarray(saved['fps']).item()) if 'fps' in saved.files else None
                    return pose, confidence, raw_count, raw_fps
    raise RuntimeError(f'No valid cached pose for {uid}')

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
canonical_pose_by_uid, canonical_conf_by_uid = {}, {}
raw_meta_by_uid, anchor_audit_by_uid = {}, {}
for n, uid in enumerate(sample_uids, start=1):
    raw_pose, raw_conf, raw_count, raw_fps = load_raw_cache(uid)
    canonical_pose, canonical_conf, anchor_info = canonicalize_v5(uid, raw_pose, raw_conf)
    canonical_pose_by_uid[uid] = canonical_pose
    canonical_conf_by_uid[uid] = canonical_conf
    anchor_audit_by_uid[uid] = anchor_info
    rec = test_by_uid[uid] if uid in test_by_uid else train_by_uid[uid]
    raw_meta_by_uid[uid] = {'raw_frame_count': raw_count, 'fps': raw_fps, 'resampling_policy': rec.get('resampling_policy')}
    if uid in test_by_uid:
        sample = sample_by_uid[uid]
        ref_path = (V5_ROOT / sample['reference']['path']).resolve()
        with np.load(ref_path) as ref:
            expected_pose = np.asarray(ref['pose'], dtype=np.float32)
            expected_conf = np.asarray(ref['confidence'], dtype=np.float32)
        assert np.allclose(canonical_pose, expected_pose, atol=2e-5, rtol=2e-5)
        assert np.allclose(canonical_conf, expected_conf, atol=2e-6, rtol=2e-6)
    if n % 1000 == 0 or n == len(sample_uids):
        print(f'Reconstructed V5 canonical rows {n}/{len(sample_uids)}', flush=True)

# Verify normalization match
fit_pose = np.stack([canonical_pose_by_uid[uid] for uid in fit_uids])
fit_conf = np.stack([canonical_conf_by_uid[uid] for uid in fit_uids])
weights = fit_conf[..., None]
den = weights.sum(axis=(0, 1))
recomputed_mean = ((fit_pose * weights).sum(axis=(0, 1)) / np.maximum(den, 1e-6)).astype(np.float32)
variance = (((fit_pose - recomputed_mean[None, None]) ** 2) * weights).sum(axis=(0, 1)) / np.maximum(den, 1e-6)
recomputed_std = np.sqrt(np.maximum(variance, 0)).clip(0.05).astype(np.float32)
assert np.allclose(recomputed_mean, v5_mean, atol=2e-5, rtol=2e-5)
assert np.allclose(recomputed_std, v5_std, atol=2e-5, rtol=2e-5)
del fit_pose, fit_conf, weights, den, variance; gc.collect()
print('Normalized targets verified and aligned with V5!')
'''

TOKEN_CACHE = r'''# Filter cross-role near duplicates, then build the frozen MiniLM token cache.
tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_NAME, revision=TEXT_ENCODER_REVISION)
text_model = AutoModel.from_pretrained(TEXT_ENCODER_NAME, revision=TEXT_ENCODER_REVISION)
ACTUAL_TEXT_ENCODER_REVISION = str(getattr(text_model.config, '_commit_hash', '') or '')
assert ACTUAL_TEXT_ENCODER_REVISION == TEXT_ENCODER_REVISION, 'Loaded MiniLM revision does not match the pinned revision.'
encoder_hash = hashlib.sha256()
for name, tensor in sorted(text_model.state_dict().items()):
    encoder_hash.update(name.encode('utf-8'))
    encoder_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes(order='C'))
TEXT_ENCODER_STATE_SHA256 = encoder_hash.hexdigest().upper()
TOKENIZER_VOCAB_SHA256 = hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest().upper()
text_model = text_model.to(device=device)
text_model.eval()

# The pairing audit found six fit/stop rows with unsupported source identity
# and one MiniLM-near-duplicate caption edge crossing those roles. Exclude
# unsupported rows and affected stop rows before rebuilding targets or token
# caches; this prevents questionable source grouping or a leaked stop score.
unsupported_fit_source_rows_removed = sum(
    not bool(train_by_uid[uid].get('source_uid_supported', True)) for uid in fit_uids
)
unsupported_stop_source_rows_removed = sum(
    not bool(train_by_uid[uid].get('source_uid_supported', True)) for uid in early_uids
)
assert (unsupported_fit_source_rows_removed, unsupported_stop_source_rows_removed) == (5, 1), (
    'The audited unsupported source-identity row counts changed; rerun the pairing audit before training.'
)
fit_uids = [uid for uid in fit_uids if bool(train_by_uid[uid].get('source_uid_supported', True))]
early_uids = [uid for uid in early_uids if bool(train_by_uid[uid].get('source_uid_supported', True))]
assert fit_uids and early_uids
FIT_UID_ORDER_SHA256 = hashlib.sha256('\n'.join(fit_uids).encode()).hexdigest()
rank_uids = fit_uids + early_uids
rank_caption_key_by_uid = {uid: caption_key(train_by_uid[uid]) for uid in rank_uids}
assert all(rank_caption_key_by_uid.values()), 'A fit or inner-stop row has an empty normalized caption key.'
rank_caption_keys = sorted(
    set(rank_caption_key_by_uid.values()),
    key=lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest(),
)
rank_caption_key_to_ix = {key: ix for ix, key in enumerate(rank_caption_keys)}
rank_caption_units = []
with torch.no_grad():
    for start in range(0, len(rank_caption_keys), 64):
        batch_keys = rank_caption_keys[start:start+64]
        inputs = tokenizer(batch_keys, max_length=LMAX_TOKENS, padding=True, truncation=True, return_tensors='pt').to(device)
        output = text_model(**inputs).last_hidden_state
        mask = inputs['attention_mask'].to(output.dtype).unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        unit = torch.nn.functional.normalize(pooled, p=2, dim=1)
        rank_caption_units.append(unit.cpu().numpy().astype(np.float32, copy=False))
rank_caption_unit = np.concatenate(rank_caption_units, axis=0)
fit_key_ix = np.asarray([rank_caption_key_to_ix[rank_caption_key_by_uid[uid]] for uid in fit_uids], dtype=np.int64)
stop_key_ix = np.asarray([rank_caption_key_to_ix[rank_caption_key_by_uid[uid]] for uid in early_uids], dtype=np.int64)
stop_near_duplicate = np.zeros(len(early_uids), dtype=np.bool_)
cross_role_near_edge_count = 0
for start in range(0, len(fit_uids), 512):
    similarities = rank_caption_unit[fit_key_ix[start:start+512]] @ rank_caption_unit[stop_key_ix].T
    near = similarities >= float(NEAR_DUPLICATE_COSINE_THRESHOLD)
    cross_role_near_edge_count += int(near.sum())
    stop_near_duplicate |= near.any(axis=0)
stop_rows_removed_for_near_duplicate = int(stop_near_duplicate.sum())
assert cross_role_near_edge_count <= 1, (
    'The audited fit/inner-stop near-caption overlap changed; rerun the pairing audit and review before training.'
)
assert stop_rows_removed_for_near_duplicate <= cross_role_near_edge_count
early_uids = [uid for i, uid in enumerate(early_uids) if not stop_near_duplicate[i]]
assert early_uids, 'Near-caption filtering removed the entire inner-stop role.'
remaining_stop_key_ix = np.asarray(
    [rank_caption_key_to_ix[caption_key(train_by_uid[uid])] for uid in early_uids], dtype=np.int64
)
remaining_cross_role_edges = 0
for start in range(0, len(fit_uids), 512):
    similarities = rank_caption_unit[fit_key_ix[start:start+512]] @ rank_caption_unit[remaining_stop_key_ix].T
    remaining_cross_role_edges += int((similarities >= float(NEAR_DUPLICATE_COSINE_THRESHOLD)).sum())
assert remaining_cross_role_edges == 0, 'Filtered inner-stop role still overlaps fit by near-caption similarity.'

# Recompute role checks and signature inputs after removing the single affected
# inner-stop row. Exact-caption and source/alias checks remain mandatory.
fit_caption_keys = {caption_key(train_by_uid[uid]) for uid in fit_uids}
stop_caption_keys = {caption_key(train_by_uid[uid]) for uid in early_uids}
assert fit_caption_keys.isdisjoint(stop_caption_keys | test_caption_keys)
assert stop_caption_keys.isdisjoint(test_caption_keys)
rank_role_uids = fit_uids + early_uids
fit_source_aliases = role_source_aliases(fit_uids, train_by_uid)
stop_source_aliases = role_source_aliases(early_uids, train_by_uid)
assert fit_source_aliases.isdisjoint(stop_source_aliases)
assert fit_source_aliases.isdisjoint(test_source_aliases)
assert stop_source_aliases.isdisjoint(test_source_aliases)
SOURCE_CLUSTER_COUNT = len({str(train_by_uid[uid]['source_video_cluster']) for uid in rank_role_uids})
SPLIT_ROLE_SHA256 = hashlib.sha256(json.dumps({
    'fit': hashlib.sha256('\n'.join(sorted(fit_uids)).encode()).hexdigest(),
    'inner_stop': hashlib.sha256('\n'.join(sorted(early_uids)).encode()).hexdigest(),
    'development_only': hashlib.sha256('\n'.join(sorted(test_uids)).encode()).hexdigest(),
}, sort_keys=True, separators=(',', ':')).encode()).hexdigest().upper()
print('Source-identity containment:', unsupported_fit_source_rows_removed, 'fit and',
      unsupported_stop_source_rows_removed, 'inner-stop rows removed for unsupported identity.')
print('Near-caption stop containment:', cross_role_near_edge_count, 'fit/stop edge(s) after source filtering;',
      stop_rows_removed_for_near_duplicate, 'inner-stop row(s) removed; remaining cross-role near edges: 0; threshold:',
      NEAR_DUPLICATE_COSINE_THRESHOLD)
print('Post-filter roles:', len(fit_uids), 'fit /', len(early_uids), 'inner stop /', len(test_uids),
      'development-only; role SHA256:', SPLIT_ROLE_SHA256)
del rank_caption_units, rank_caption_unit, rank_caption_key_by_uid, rank_caption_keys
del rank_caption_key_to_ix, fit_key_ix, stop_key_ix, remaining_stop_key_ix, stop_near_duplicate, rank_uids
gc.collect(); torch.cuda.empty_cache()

all_uids = fit_uids + early_uids + test_uids
uid_to_index = {uid: i for i, uid in enumerate(all_uids)}
canonical_poses = np.stack([canonical_pose_by_uid[uid] for uid in all_uids]).astype(np.float32)
canonical_confs = np.stack([canonical_conf_by_uid[uid] for uid in all_uids]).astype(np.float32)
targets = np.nan_to_num((canonical_poses - v5_mean[None, None]) / v5_std[None, None], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
CONFIDENCE = canonical_confs
assert targets.shape == (len(all_uids), TMAX, POINTS, COORDS), f"Unexpected targets shape: {targets.shape}"
assert CONFIDENCE.shape == (len(all_uids), TMAX, POINTS), f"Unexpected CONFIDENCE shape: {CONFIDENCE.shape}"
assert np.isfinite(targets).all() and np.isfinite(CONFIDENCE).all()

all_role_texts = [str(train_by_uid[uid]['text']) for uid in fit_uids + early_uids] + [str(test_by_uid[uid]['text']) for uid in test_uids]
counterfactual_text_by_uid = {uid: str(sample_by_uid[uid]['counterfactual_text']) for uid in test_uids}
all_texts = sorted(set(all_role_texts + list(counterfactual_text_by_uid.values())),
                   key=lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest())
text_to_ix = {text: ix for ix, text in enumerate(all_texts)}
uid_text_ix = {uid: text_to_ix[str(test_by_uid[uid]['text'] if uid in test_by_uid else train_by_uid[uid]['text'])] for uid in all_uids}
uid_counterfactual_ix = {uid: text_to_ix[text] for uid, text in counterfactual_text_by_uid.items()}
texts_sha256 = hashlib.sha256('\n'.join(all_texts).encode('utf-8')).hexdigest()

token_hiddens, token_lengths = [], []
with torch.no_grad():
    for start in range(0, len(all_texts), 64):
        batch_texts = all_texts[start:start+64]
        inputs = tokenizer(batch_texts, max_length=LMAX_TOKENS, padding=True, truncation=True, return_tensors='pt').to(device)
        out = text_model(**inputs)
        mask = inputs['attention_mask'].bool()
        for i in range(len(batch_texts)):
            length = int(mask[i].sum().item())
            token_hiddens.append(out.last_hidden_state[i, :length].cpu().half().numpy())
            token_lengths.append(length)

del text_model; torch.cuda.empty_cache(); gc.collect()
token_lengths = np.asarray(token_lengths, dtype=np.int32)
token_offsets = np.concatenate(([0], np.cumsum(token_lengths))).astype(np.int64)
token_hidden = np.concatenate(token_hiddens, axis=0)
text_pooled = np.stack([
    token_hidden[int(token_offsets[i]):int(token_offsets[i+1])].astype(np.float32).mean(axis=0)
    for i in range(len(all_texts))
])
text_unit = text_pooled / np.maximum(np.linalg.norm(text_pooled, axis=1, keepdims=True), 1e-12)
TEXT_EMBEDDINGS_SHA256 = hashlib.sha256(np.ascontiguousarray(text_unit, dtype=np.float32).tobytes(order='C')).hexdigest().upper()
token_cache_hasher = hashlib.sha256()
token_cache_hasher.update(memoryview(np.ascontiguousarray(token_hidden, dtype=np.float16)).cast('B'))
token_cache_hasher.update(memoryview(np.ascontiguousarray(token_offsets, dtype=np.int64)).cast('B'))
token_cache_hasher.update(memoryview(np.ascontiguousarray(token_lengths, dtype=np.int32)).cast('B'))
TOKEN_HIDDEN_CACHE_SHA256 = token_cache_hasher.hexdigest().upper()
normalizer_hash = hashlib.sha256()
normalizer_hash.update(np.ascontiguousarray(v5_mean, dtype=np.float32).tobytes(order='C'))
normalizer_hash.update(np.ascontiguousarray(v5_std, dtype=np.float32).tobytes(order='C'))
NORMALIZER_SHA256 = normalizer_hash.hexdigest().upper()
target_hash = hashlib.sha256()
target_hash.update(memoryview(np.ascontiguousarray(targets, dtype=np.float32)).cast('B'))
target_hash.update(memoryview(np.ascontiguousarray(CONFIDENCE, dtype=np.float32)).cast('B'))
target_hash.update(FIT_UID_ORDER_SHA256.encode('ascii'))
target_hash.update(hashlib.sha256('\n'.join(early_uids).encode()).hexdigest().encode('ascii'))
target_hash.update(TEST_UID_ORDER_SHA256.encode('ascii'))
TARGET_CACHE_SHA256 = target_hash.hexdigest().upper()
del token_hiddens; gc.collect()
print('Pinned text encoder:', TEXT_ENCODER_NAME, ACTUAL_TEXT_ENCODER_REVISION)
print('Encoder/tokenizer/pooled and decoder token-cache hashes:', TEXT_ENCODER_STATE_SHA256, TOKENIZER_VOCAB_SHA256, TEXT_EMBEDDINGS_SHA256, TOKEN_HIDDEN_CACHE_SHA256)
print('Normalizer/target/caption hashes:', NORMALIZER_SHA256, TARGET_CACHE_SHA256, texts_sha256.upper())
print('Token cache ready:', len(all_texts), 'texts, total tokens:', len(token_hidden), '| RAM shape:', token_hidden.shape)
'''

DATA_AND_LOSS = r'''# DataLoader, exact objective-aligned evaluation, and aggregate stability diagnostics.
BONE_EDGES = [
    (11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),
]
for offset in (33, 54):
    BONE_EDGES.extend((offset+a, offset+b) for a,b in
        [(0,1),(0,5),(0,9),(0,13),(0,17),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),
         (9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20)])
BONE_EDGES = np.asarray(BONE_EDGES, dtype=np.int64)
RANK_SOURCE_SET = {}
for uid in fit_uids + early_uids:
    row = train_by_uid[uid]
    aliases = row.get('source_aliases') or []
    RANK_SOURCE_SET[uid] = {str(row.get('source_video_cluster') or ''),
                            str(row.get('source_video_id') or ''),
                            *[str(alias) for alias in aliases if alias]}
assert all(RANK_SOURCE_SET[uid] - {''} for uid in fit_uids + early_uids)

def rank_pair_mask(batch_uids, shift):
    n = len(batch_uids)
    if n < 2:
        return np.zeros(n, dtype=np.bool_), {'candidate_pairs':0,'same_source_rejects':0,'same_caption_rejects':0,'near_duplicate_rejects':0,'valid_pairs':0}
    negative_uids = [batch_uids[(i-shift) % n] for i in range(n)]
    same_source = np.asarray([bool((RANK_SOURCE_SET[a] - {''}) & (RANK_SOURCE_SET[b] - {''}))
                              for a,b in zip(batch_uids, negative_uids)], dtype=np.bool_)
    same_caption = np.asarray([caption_key(train_by_uid[a]) == caption_key(train_by_uid[b])
                               for a,b in zip(batch_uids, negative_uids)], dtype=np.bool_)
    left_ix = np.asarray([uid_text_ix[uid] for uid in batch_uids], dtype=np.int64)
    right_ix = np.asarray([uid_text_ix[uid] for uid in negative_uids], dtype=np.int64)
    cosine = np.sum(text_unit[left_ix] * text_unit[right_ix], axis=1)
    near_duplicate = cosine >= NEAR_DUPLICATE_COSINE_THRESHOLD
    valid = ~(same_source | same_caption | near_duplicate)
    return valid, {'candidate_pairs':n,'same_source_rejects':int(same_source.sum()),
                   'same_caption_rejects':int(same_caption.sum()),
                   'near_duplicate_rejects':int(near_duplicate.sum()),'valid_pairs':int(valid.sum())}

def choose_rank_pairing(batch_uids, start_shift=1):
    n = len(batch_uids)
    if n < 2:
        mask, stats = rank_pair_mask(batch_uids, 0)
        return 0, mask, stats
    last = None
    for step in range(n-1):
        shift = ((start_shift - 1 + step) % (n-1)) + 1
        mask, stats = rank_pair_mask(batch_uids, shift)
        last = (shift, mask, stats)
        if stats['valid_pairs'] > 0:
            return last
    return last

def build_fixed_stop_pairing(stop_uids):
    """Freeze the highest-coverage deterministic filtered negative cycle."""
    if len(stop_uids) < 2:
        raise RuntimeError('Inner-stop role is too small for temporal ranking.')
    best = None
    for shift in range(1, len(stop_uids)):
        valid, stats = rank_pair_mask(stop_uids, shift)
        if best is None or stats['valid_pairs'] > best[2]['valid_pairs']:
            best = (shift, valid, stats)
    shift, valid, stats = best
    if stats['valid_pairs'] <= 0:
        raise RuntimeError('No valid filtered true/negative temporal ranking pairs in inner-stop role.')
    negative_by_uid = {uid: stop_uids[(i-shift) % len(stop_uids)] for i, uid in enumerate(stop_uids)}
    valid_by_uid = {uid: bool(valid[i]) for i, uid in enumerate(stop_uids)}
    filter_by_uid = {}
    for uid in stop_uids:
        negative_uid = negative_by_uid[uid]
        shared_source = bool((RANK_SOURCE_SET[uid]-{''}) & (RANK_SOURCE_SET[negative_uid]-{''}))
        same_caption = caption_key(train_by_uid[uid]) == caption_key(train_by_uid[negative_uid])
        left_ix, right_ix = uid_text_ix[uid], uid_text_ix[negative_uid]
        cosine = float(np.dot(text_unit[left_ix], text_unit[right_ix]))
        filter_by_uid[uid] = {
            'same_source':shared_source, 'same_caption':same_caption,
            'near_duplicate':cosine >= NEAR_DUPLICATE_COSINE_THRESHOLD,
        }
        assert valid_by_uid[uid] == (not any(filter_by_uid[uid].values()))
    mapping_payload = '\n'.join(f'{uid}>{negative_by_uid[uid]}:{int(valid_by_uid[uid])}' for uid in stop_uids)
    mapping_sha = hashlib.sha256(mapping_payload.encode()).hexdigest().upper()
    return negative_by_uid, valid_by_uid, filter_by_uid, shift, stats, mapping_sha

STOP_NEGATIVE_UID_BY_UID, STOP_RANK_VALID_BY_UID, STOP_RANK_FILTER_BY_UID, \
    STOP_RANK_SHIFT, STOP_RANK_FILTER_COUNTS, STOP_RANK_MAPPING_SHA256 = build_fixed_stop_pairing(early_uids)
print('Fixed inner-stop temporal negatives:', STOP_RANK_FILTER_COUNTS,
      '| mapping SHA256:', STOP_RANK_MAPPING_SHA256)

class UIDDataset(Dataset):
    def __init__(self, uids): self.uids = list(uids)
    def __len__(self): return len(self.uids)
    def __getitem__(self, index): return self.uids[index]

def collate_uids(batch_uids, text_mode='true'):
    ids = np.asarray([uid_to_index[uid] for uid in batch_uids], dtype=np.int64)
    target = targets[ids]
    confidence = CONFIDENCE[ids]
    if text_mode == 'no_text':
        hidden = np.zeros((len(batch_uids), 1, 384), dtype=np.float32)
        mask = np.ones((len(batch_uids), 1), dtype=np.bool_)
    else:
        if text_mode not in ('true', 'shuffled'):
            raise ValueError(f'Unknown text mode: {text_mode}')
        ix = [uid_counterfactual_ix[uid] if text_mode == 'shuffled' else uid_text_ix[uid] for uid in batch_uids]
        lengths = [int(token_lengths[item]) for item in ix]
        width = max(lengths)
        hidden = np.zeros((len(ix), width, 384), dtype=np.float16)
        mask = np.zeros((len(ix), width), dtype=np.bool_)
        for row, (text_ix, length) in enumerate(zip(ix, lengths)):
            hidden[row, :length] = token_hidden[int(token_offsets[text_ix]):int(token_offsets[text_ix+1])]
            mask[row, :length] = True
    return (torch.from_numpy(hidden).float(), torch.from_numpy(mask),
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

def temporal_grouped_scores(prediction, target, confidence):
    return _grouped_temporal_scores(
        prediction.reshape(-1,TMAX,POINTS,COORDS),
        target.reshape(-1,TMAX,POINTS,COORDS),
        confidence.reshape(-1,TMAX,POINTS), GROUPS, phase_weight=PHASE_WEIGHT
    )

def evaluate_cohort(uids, text_mode='true', batch_size=EVAL_BATCH, include_ranking=False):
    parallel.eval()
    totals = {name: np.zeros(4, dtype=np.float64) for name in GROUPS}
    motion = {name: np.zeros(3, dtype=np.float64) for name in GROUPS}
    sample_positions, sample_velocities, sample_phases, sample_scores = [], [], [], []
    frame_max_abs_xy, frame_max_abs_xyz, endpoint_to_median_speed = [], [], []
    bone_pred_by_edge = [[] for _ in range(len(BONE_EDGES))]
    bone_ref_by_edge = [[] for _ in range(len(BONE_EDGES))]
    bone_group_support = {name:0 for name in ('body','left_hand','right_hand')}
    bone_group_pred = {name:[] for name in bone_group_support}
    bone_group_ref = {name:[] for name in bone_group_support}
    clip_bone_group_ratios = []
    bone_unscorable_clip_groups = 0
    valid_abs_xy_gt5 = 0
    rank_loss_numerator, rank_pair_count = 0.0, 0
    rank_filter_totals = {'candidate_pairs':0,'same_source_rejects':0,'same_caption_rejects':0,
                          'near_duplicate_rejects':0,'valid_pairs':0}
    with torch.inference_mode():
        for start in range(0, len(uids), batch_size):
            batch_uids = list(uids[start:start+batch_size])
            hidden, mask, target, confidence, _ = collate_uids(batch_uids, text_mode=text_mode)
            hidden = hidden.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True); confidence = confidence.to(device, non_blocking=True)
            prediction = parallel(hidden, mask)
            assert prediction.shape == target.shape and torch.isfinite(prediction).all()

            negative_prediction, ranking_mask = None, None
            if include_ranking:
                negative_uids = [STOP_NEGATIVE_UID_BY_UID[uid] for uid in batch_uids]
                valid_pairs = np.asarray([STOP_RANK_VALID_BY_UID[uid] for uid in batch_uids], dtype=np.bool_)
                filter_counts = {'candidate_pairs':len(batch_uids),'same_source_rejects':0,
                                 'same_caption_rejects':0,'near_duplicate_rejects':0,
                                 'valid_pairs':int(valid_pairs.sum())}
                for uid in batch_uids:
                    for key, flag in STOP_RANK_FILTER_BY_UID[uid].items():
                        target_key = {'same_source':'same_source_rejects',
                                      'same_caption':'same_caption_rejects',
                                      'near_duplicate':'near_duplicate_rejects'}[key]
                        filter_counts[target_key] += int(flag)
                for key,value in filter_counts.items(): rank_filter_totals[key] += value
                ranking_mask = torch.as_tensor(valid_pairs, dtype=torch.bool, device=device)
                n_valid = int(valid_pairs.sum())
                if n_valid:
                    negative_hidden, negative_mask, _, _, _ = collate_uids(negative_uids, text_mode='true')
                    negative_hidden = negative_hidden.to(device, non_blocking=True)
                    negative_mask = negative_mask.to(device, non_blocking=True)
                    negative_prediction = parallel(negative_hidden, negative_mask)
                    assert torch.isfinite(negative_prediction).all()
                    _, rank_metrics = compute_motion_losses(
                        prediction,target,confidence,GROUPS,negative_prediction=negative_prediction,
                        ranking_mask=ranking_mask,velocity_weight=VELOCITY_WEIGHT,
                        rank_weight=RANK_WEIGHT,rank_margin=RANK_MARGIN,phase_weight=PHASE_WEIGHT
                    )
                    rank_loss_numerator += rank_metrics['ranking_loss'] * n_valid
                    rank_pair_count += n_valid

            pos_per, _, _, _ = _grouped_sample_scores(prediction, target, confidence, GROUPS, 0.0)
            vel_per, phase_per, dynamics_per, _ = temporal_grouped_scores(prediction, target, confidence)
            score_per = pos_per + DYNAMIC_WEIGHT * dynamics_per
            sample_positions.extend(pos_per.cpu().numpy().astype(np.float64).tolist())
            sample_velocities.extend(vel_per.cpu().numpy().astype(np.float64).tolist())
            sample_phases.extend(phase_per.cpu().numpy().astype(np.float64).tolist())
            sample_scores.extend(score_per.cpu().numpy().astype(np.float64).tolist())
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

            pred_raw = p.cpu().numpy().astype(np.float32) * v5_std[None,None] + v5_mean[None,None]
            ref_raw = y.cpu().numpy().astype(np.float32) * v5_std[None,None] + v5_mean[None,None]
            frame_max_abs_xy.append(np.abs(pred_raw[..., :2]).max(axis=(2,3)))
            frame_max_abs_xyz.append(np.abs(pred_raw).max(axis=(2,3)))
            valid_xy = (confidence.cpu().numpy() >= MIN_ANCHOR_CONF)[...,None]
            valid_abs_xy_gt5 += int(((np.abs(pred_raw[..., :2]) > 5.0) & valid_xy).sum())
            speed = np.linalg.norm(np.diff(pred_raw, axis=1), axis=-1).mean(axis=2)
            end_speed = np.linalg.norm(np.diff(pred_raw, axis=1)[:,-1], axis=-1).mean(axis=1)
            median_speed = np.median(speed, axis=1)
            endpoint_to_median_speed.extend((end_speed / np.maximum(median_speed, 1e-8)).astype(np.float64).tolist())
            pred_bones_clip = np.linalg.norm(pred_raw[:,:,BONE_EDGES[:,0]] - pred_raw[:,:,BONE_EDGES[:,1]], axis=-1)
            ref_bones_clip = np.linalg.norm(ref_raw[:,:,BONE_EDGES[:,0]] - ref_raw[:,:,BONE_EDGES[:,1]], axis=-1)
            conf_np = confidence.detach().cpu().numpy()
            endpoint_conf = np.minimum(conf_np[:,:,BONE_EDGES[:,0]], conf_np[:,:,BONE_EDGES[:,1]])
            edge_valid = (endpoint_conf >= MIN_ANCHOR_CONF) & np.isfinite(pred_bones_clip) & np.isfinite(ref_bones_clip)
            bone_group_edges = {'body':(0,12),'left_hand':(12,32),'right_hand':(32,52)}
            for edge_ix in range(len(BONE_EDGES)):
                edge_mask = edge_valid[:,:,edge_ix]
                bone_pred_by_edge[edge_ix].append(pred_bones_clip[:,:,edge_ix][edge_mask])
                bone_ref_by_edge[edge_ix].append(ref_bones_clip[:,:,edge_ix][edge_mask])
            for group_name,(edge_start,edge_end) in bone_group_edges.items():
                group_valid = edge_valid[:,:,edge_start:edge_end]
                bone_group_support[group_name] += int(group_valid.sum())
                bone_group_pred[group_name].append(pred_bones_clip[:,:,edge_start:edge_end][group_valid])
                bone_group_ref[group_name].append(ref_bones_clip[:,:,edge_start:edge_end][group_valid])
                for local_clip in range(len(batch_uids)):
                    clip_mask = group_valid[local_clip]
                    if int(clip_mask.sum()) < BONE_MIN_CLIP_GROUP_SUPPORT:
                        bone_unscorable_clip_groups += 1
                        continue
                    clip_ref = ref_bones_clip[local_clip,:,edge_start:edge_end][clip_mask]
                    clip_pred = pred_bones_clip[local_clip,:,edge_start:edge_end][clip_mask]
                    ref_p95 = float(np.percentile(clip_ref,95))
                    if not np.isfinite(ref_p95) or ref_p95 < BONE_REFERENCE_P95_MIN:
                        bone_unscorable_clip_groups += 1
                        continue
                    clip_bone_group_ratios.append(float(np.percentile(clip_pred,95)/ref_p95))

    pos_group = {name: totals[name][0]/max(totals[name][1], 1.0) for name in GROUPS}
    vel_group = {name: totals[name][2]/max(totals[name][3], 1.0) for name in GROUPS}
    pos = sum(GROUPS[name][2] * pos_group[name] for name in GROUPS)
    vel = sum(GROUPS[name][2] * vel_group[name] for name in GROUPS)
    pred_motion = {name: motion[name][0]/max(motion[name][2], 1.0) for name in GROUPS}
    ref_motion = {name: motion[name][1]/max(motion[name][2], 1.0) for name in GROUPS}
    ratios = {name: pred_motion[name]/max(ref_motion[name], 1e-8) for name in GROUPS}
    frame_max = np.concatenate(frame_max_abs_xy, axis=0)
    frame_max_xyz = np.concatenate(frame_max_abs_xyz, axis=0)
    bone_edge_diagnostics = []
    bone_unscorable_edges = 0
    edge_group_name = lambda edge_ix: ('body' if edge_ix < 12 else ('left_hand' if edge_ix < 32 else 'right_hand'))
    for edge_ix,(pred_parts,ref_parts) in enumerate(zip(bone_pred_by_edge,bone_ref_by_edge)):
        pred_values = np.concatenate(pred_parts) if pred_parts else np.empty(0,dtype=np.float32)
        ref_values = np.concatenate(ref_parts) if ref_parts else np.empty(0,dtype=np.float32)
        support = min(len(pred_values),len(ref_values))
        ref_p95 = float(np.percentile(ref_values,95)) if support else 0.0
        scorable = support >= BONE_MIN_EDGE_SUPPORT and ref_p95 >= BONE_REFERENCE_P95_MIN
        if not scorable:
            bone_unscorable_edges += 1
        bone_edge_diagnostics.append({
            'group':edge_group_name(edge_ix),'edge_index':edge_ix,
            'valid_pair_count':int(support),'scorable':bool(scorable),
            'prediction_p50':float(np.percentile(pred_values,50)) if support else None,
            'prediction_p95':float(np.percentile(pred_values,95)) if support else None,
            'reference_p50':float(np.percentile(ref_values,50)) if support else None,
            'reference_p95':ref_p95 if support else None,
            'p95_ratio':float(np.percentile(pred_values,95)/ref_p95) if scorable else None,
        })
    bone_group_diagnostics = {}
    for group_name in ('body','left_hand','right_hand'):
        pred_values = np.concatenate(bone_group_pred[group_name])
        ref_values = np.concatenate(bone_group_ref[group_name])
        ref_p95 = float(np.percentile(ref_values,95)) if len(ref_values) else 0.0
        scorable = (bone_group_support[group_name] >= BONE_MIN_EDGE_SUPPORT
                    and ref_p95 >= BONE_REFERENCE_P95_MIN)
        bone_group_diagnostics[group_name] = {
            'valid_pair_count':int(bone_group_support[group_name]),'scorable':bool(scorable),
            'prediction_p50':float(np.percentile(pred_values,50)) if len(pred_values) else None,
            'prediction_p95':float(np.percentile(pred_values,95)) if len(pred_values) else None,
            'reference_p50':float(np.percentile(ref_values,50)) if len(ref_values) else None,
            'reference_p95':ref_p95 if len(ref_values) else None,
            'p95_ratio':float(np.percentile(pred_values,95)/ref_p95) if scorable else None,
        }
    nonempty_pred_groups = [np.concatenate(parts) for parts in bone_group_pred.values() if parts and any(len(x) for x in parts)]
    nonempty_ref_groups = [np.concatenate(parts) for parts in bone_group_ref.values() if parts and any(len(x) for x in parts)]
    all_pred_bones = np.concatenate(nonempty_pred_groups) if nonempty_pred_groups else np.empty(0,dtype=np.float32)
    all_ref_bones = np.concatenate(nonempty_ref_groups) if nonempty_ref_groups else np.empty(0,dtype=np.float32)
    stability = {
        'frame_max_abs_xy_p99_by_frame': np.percentile(frame_max, 99, axis=0).round(6).tolist(),
        'frame_max_abs_xy_global_max': float(frame_max.max()),
        'frame_max_abs_xyz_p99_by_frame': np.percentile(frame_max_xyz, 99, axis=0).round(6).tolist(),
        'frame_max_abs_xyz_global_max': float(frame_max_xyz.max()),
        'valid_xy_coordinate_count_abs_gt5': int(valid_abs_xy_gt5),
        'endpoint_to_median_transition_speed_p50': float(np.percentile(endpoint_to_median_speed, 50)),
        'endpoint_to_median_transition_speed_p95': float(np.percentile(endpoint_to_median_speed, 95)),
        'endpoint_to_median_transition_speed_max': float(np.max(endpoint_to_median_speed)),
        'confidence_filtered_bone_edges':bone_edge_diagnostics,
        'confidence_filtered_bone_groups':bone_group_diagnostics,
        'bone_unscorable_edge_count':int(bone_unscorable_edges),
        'bone_unscorable_clip_group_count':int(bone_unscorable_clip_groups),
        'bone_valid_pair_count_by_group':bone_group_support,
        'pred_bone_length_p50': float(np.percentile(all_pred_bones, 50)) if len(all_pred_bones) else None,
        'pred_bone_length_p95': float(np.percentile(all_pred_bones, 95)) if len(all_pred_bones) else None,
        'ref_bone_length_p50': float(np.percentile(all_ref_bones, 50)) if len(all_ref_bones) else None,
        'ref_bone_length_p95': float(np.percentile(all_ref_bones, 95)) if len(all_ref_bones) else None,
        'bone_length_p95_ratio_to_reference': float(np.percentile(all_pred_bones, 95) / max(np.percentile(all_ref_bones, 95), 1e-8)) if len(all_pred_bones) and len(all_ref_bones) else None,
        'per_clip_bone_length_p95_ratio_max': float(np.max(clip_bone_group_ratios)) if clip_bone_group_ratios else None,
        'bone_filter_policy':{'both_endpoint_confidence_gte':MIN_ANCHOR_CONF,
                              'minimum_edge_support':BONE_MIN_EDGE_SUPPORT,
                              'minimum_clip_group_support':BONE_MIN_CLIP_GROUP_SUPPORT,
                              'minimum_reference_p95':BONE_REFERENCE_P95_MIN},
    }
    sample_mean_position = float(np.mean(sample_positions))
    sample_mean_velocity = float(np.mean(sample_velocities))
    sample_mean_phase = float(np.mean(sample_phases))
    sample_mean_dynamics = sample_mean_velocity + PHASE_WEIGHT * sample_mean_phase
    selection_pose_score = sample_mean_position
    selection_rank_loss = rank_loss_numerator / max(rank_pair_count, 1)
    result = {
        'position': float(pos), 'velocity': float(vel), 'phase_profile_error':sample_mean_phase,
        'dynamic_error':float(sample_mean_dynamics),
        'sample_mean_position': sample_mean_position,
        'sample_mean_velocity': sample_mean_velocity,
        'sample_mean_phase_profile_error':sample_mean_phase,
        'sample_mean_dynamic_error':float(sample_mean_dynamics),
        'selection_pose_score': float(selection_pose_score),
        'selection_ranking_loss': float(selection_rank_loss),
        'selection_score': float(selection_pose_score + DYNAMIC_WEIGHT * sample_mean_dynamics + RANK_WEIGHT * selection_rank_loss),
        'selection_valid_rank_pairs': int(rank_pair_count),
        'rank_filter_counts': rank_filter_totals,
        'pooled_selection_score': float(pos + DYNAMIC_WEIGHT * (vel + PHASE_WEIGHT * sample_mean_phase)),
        'position_by_group': pos_group, 'velocity_by_group': vel_group,
        'prediction_motion_by_group': pred_motion, 'reference_motion_by_group': ref_motion,
        'motion_ratio_by_group': ratios, 'stability': stability,
    }
    return result, np.asarray(sample_positions), np.asarray(sample_velocities), np.asarray(sample_phases), np.asarray(sample_scores)

FIT_FRAME_MEAN = targets[np.asarray([uid_to_index[uid] for uid in fit_uids],dtype=np.int64)].mean(axis=0)
def evaluate_frame_mean_position(uids, batch_size=EVAL_BATCH):
    scores=[]
    baseline=torch.as_tensor(FIT_FRAME_MEAN,dtype=torch.float32,device=device)
    with torch.inference_mode():
        for start in range(0,len(uids),batch_size):
            batch_uids=list(uids[start:start+batch_size])
            ids=np.asarray([uid_to_index[uid] for uid in batch_uids],dtype=np.int64)
            target=torch.from_numpy(targets[ids]).to(device)
            confidence=torch.from_numpy(CONFIDENCE[ids]).to(device)
            prediction=baseline[None].expand(len(batch_uids),-1,-1,-1)
            pos_per,_,_,_=_grouped_sample_scores(prediction,target,confidence,GROUPS,0.0)
            scores.extend(pos_per.cpu().numpy().astype(np.float64).tolist())
    return float(np.mean(scores))

STOP_FRAME_MEAN_POSITION=evaluate_frame_mean_position(early_uids)
POSE_GUARDRAIL=1.05*STOP_FRAME_MEAN_POSITION
print('Inner-stop frame-mean position baseline:',f'{STOP_FRAME_MEAN_POSITION:.6f}',
      '| pose guardrail <=',f'{POSE_GUARDRAIL:.6f}')
print('V9 objective-aligned evaluation and aggregate diagnostics initialized.')
'''

TRAIN = r'''# V9 controlled same-role experiment: calibrated coordinate means + caption ranking.
# Training target audit: aggregate only over the fit role; no per-row values are persisted.
fit_ids_for_audit = np.asarray([uid_to_index[uid] for uid in fit_uids], dtype=np.int64)
fit_raw_targets = targets[fit_ids_for_audit] * v5_std[None,None] + v5_mean[None,None]
fit_valid = (CONFIDENCE[fit_ids_for_audit] >= MIN_ANCHOR_CONF)[...,None]
fit_valid_coords = np.broadcast_to(fit_valid, fit_raw_targets.shape)
fit_abs_values = np.abs(fit_raw_targets[fit_valid_coords])
fit_xy_valid = np.broadcast_to(fit_valid, fit_raw_targets[..., :2].shape)
fit_abs_xy = np.abs(fit_raw_targets[..., :2][fit_xy_valid])
assert np.isfinite(fit_abs_values).all() and len(fit_abs_values) > 0
FIT_TARGET_AUDIT = {
    'fit_rows':len(fit_uids),'valid_coordinate_count':int(len(fit_abs_values)),
    'abs_coordinate_p50_p95_p99_max':[float(x) for x in np.percentile(fit_abs_values,[50,95,99,100])],
    'valid_xy_coordinate_count_abs_gt5':int((fit_abs_xy>5.0).sum()),
}
print('Fit-only target range audit:', json.dumps(FIT_TARGET_AUDIT, sort_keys=True))

decoder_core = MotionTransformerDecoder(
    text_dim=384, model_dim=256, num_heads=8, num_layers=4,
    feedforward_dim=1024, dropout=0.1, max_frames=TMAX,
    points=POINTS, coordinates=COORDS
).to(device=device, dtype=torch.float32)
# The pose head is a Sequential stack. Capture and validate its final Linear
# layer explicitly so the synchronized gradient probe never targets the stack.
pose_head_linears = [layer for layer in decoder_core.pose_head if isinstance(layer, nn.Linear)]
assert pose_head_linears, 'V9 pose head has no Linear layer for the GPU gradient probe.'
probe_gradient_target = pose_head_linears[-1].weight
assert tuple(probe_gradient_target.shape) == (POSE_DIM, decoder_core.model_dim), (
    f'Unexpected V9 final pose projection shape: {tuple(probe_gradient_target.shape)}; '
    f'expected {(POSE_DIM, decoder_core.model_dim)}.'
)
parallel = nn.DataParallel(decoder_core, device_ids=[0, 1])

# Synchronized fresh GPU gate. Epoch training starts only after both devices pass.
probe_uids = None
probe_shift = 0
probe_valid_pairs = None
probe_filter_counts = None
for probe_start in range(0, min(len(fit_uids), 512), BATCH_SIZE):
    candidate_uids = fit_uids[probe_start:probe_start+BATCH_SIZE]
    if len(candidate_uids) < 2:
        continue
    candidate_shift, candidate_valid, candidate_counts = choose_rank_pairing(candidate_uids, start_shift=1)
    if candidate_counts['valid_pairs'] > 0:
        probe_uids, probe_shift = candidate_uids, candidate_shift
        probe_valid_pairs, probe_filter_counts = candidate_valid, candidate_counts
        break
if probe_uids is None:
    raise RuntimeError('No source- and semantic-filtered ranking pair is available in the first 512 fit rows; refusing to start V9.')
probe_h, probe_m, probe_y, probe_c, probe_uids = collate_uids(probe_uids, text_mode='true')
probe_h = probe_h.to(device); probe_m = probe_m.to(device); probe_y = probe_y.to(device); probe_c = probe_c.to(device)
seen_devices = set()
hook = decoder_core.text_projection.register_forward_pre_hook(lambda module, inputs: seen_devices.add(inputs[0].device.index))
optimizer = torch.optim.AdamW(parallel.parameters(), lr=3e-4, weight_decay=1e-4)
parallel.train(); optimizer.zero_grad(set_to_none=True)
for gpu_ix in range(torch.cuda.device_count()): torch.cuda.synchronize(gpu_ix)
t0 = time.perf_counter()
assert int(probe_valid_pairs.sum()) > 0, 'GPU rank smoke batch has no source/caption/embedding-distinct negative pair.'
probe_neg_h = torch.roll(probe_h, shifts=probe_shift, dims=0).contiguous()
probe_neg_m = torch.roll(probe_m, shifts=probe_shift, dims=0).contiguous()
probe_pred = parallel(probe_h, probe_m)
probe_neg_pred = parallel(probe_neg_h, probe_neg_m)
assert probe_pred.shape == probe_y.shape and torch.isfinite(probe_pred).all() and torch.isfinite(probe_neg_pred).all()
probe_mask = torch.as_tensor(probe_valid_pairs, dtype=torch.bool, device=device)
probe_loss, probe_metrics = compute_motion_losses(
    probe_pred, probe_y, probe_c, GROUPS, negative_prediction=probe_neg_pred, ranking_mask=probe_mask,
    velocity_weight=VELOCITY_WEIGHT, rank_weight=RANK_WEIGHT, rank_margin=RANK_MARGIN,
    phase_weight=PHASE_WEIGHT
)
assert torch.isfinite(probe_loss)
probe_components = compute_motion_loss_components(
    probe_pred, probe_y, probe_c, GROUPS, negative_prediction=probe_neg_pred,
    ranking_mask=probe_mask, velocity_weight=VELOCITY_WEIGHT,
    phase_weight=PHASE_WEIGHT, rank_margin=RANK_MARGIN
)
probe_gradient_terms = {
    'effective_pose':probe_components['pose'],
    'effective_dynamics':DYNAMIC_WEIGHT*probe_components['dynamics'],
    'weighted_phase_profile_auxiliary':DYNAMIC_WEIGHT*PHASE_WEIGHT*probe_components['phase_profile'],
    'weighted_rank_hinge':RANK_WEIGHT*probe_components['rank'],
}
probe_component_grad_norms = {}
for component_name, component_tensor in probe_gradient_terms.items():
    component_grad = torch.autograd.grad(
        component_tensor, probe_gradient_target,
        retain_graph=True, allow_unused=True
    )[0]
    probe_component_grad_norms[component_name] = (
        float(component_grad.norm().detach().item()) if component_grad is not None else 0.0
    )
assert all(np.isfinite(value) and value < 1e4 for value in probe_component_grad_norms.values())
assert probe_component_grad_norms['effective_pose'] > 0, 'Pose objective has no measurable gradient.'
assert probe_component_grad_norms['effective_dynamics'] > 0, 'Weighted dynamics objective has no measurable gradient.'
assert probe_component_grad_norms['weighted_phase_profile_auxiliary'] > 0, 'Weighted phase-profile auxiliary has no measurable gradient.'
probe_rank_hinge_active = float(probe_metrics['ranking_loss']) > 0.0
if probe_rank_hinge_active:
    assert probe_component_grad_norms['weighted_rank_hinge'] > 0, 'Active weighted ranking hinge has no measurable gradient.'
probe_gradient_ratios_to_pose = {
    key:float(value/probe_component_grad_norms['effective_pose'])
    for key,value in probe_component_grad_norms.items()
}
assert all(np.isfinite(value) for value in probe_gradient_ratios_to_pose.values())
probe_rank_zero_gradient_margin_satisfied = (
    not probe_rank_hinge_active and probe_component_grad_norms['weighted_rank_hinge'] == 0.0
)
probe_loss.backward()
probe_grad = torch.nn.utils.clip_grad_norm_(parallel.parameters(), 1.0)
assert torch.isfinite(probe_grad) and float(probe_grad) > 0
for gpu_ix in range(torch.cuda.device_count()): torch.cuda.synchronize(gpu_ix)
probe_seconds = time.perf_counter() - t0
hook.remove(); optimizer.zero_grad(set_to_none=True)
assert {0, 1}.issubset(seen_devices), f'Multi-GPU probe failed; seen devices={seen_devices}'
print(f'V9 synchronized 2-GPU forward/backward gate passed: {probe_seconds:.3f}s; grad_norm={float(probe_grad):.4f}; rank={probe_metrics["ranking_loss"]:.5f}; valid_pairs={probe_filter_counts["valid_pairs"]}')
print('V9 effective component gradient norms on pose head:',json.dumps(probe_component_grad_norms,sort_keys=True))
print('V9 effective component gradient ratios to pose:',json.dumps(probe_gradient_ratios_to_pose,sort_keys=True))
print('V9 ranking hinge active:',probe_rank_hinge_active,
      '| zero hinge gradient with satisfied margin:',probe_rank_zero_gradient_margin_satisfied)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
train_loader = make_loader(fit_uids, BATCH_SIZE, shuffle=True)
signature_payload = {
    'model_arm': 'motion_transformer_minilm_v9_temporal_ranked',
    'module_sha256': 'MODULE_SHA256_VALUE', 'builder_sha256': 'BUILDER_SHA256_VALUE',
    'v5_checkpoint_sha256': V5_CHECKPOINT_SHA256,
    'v5_manifest_sha256': V5_MANIFEST_SHA256,
    'v5_metrics_sha256': V5_METRICS_SHA256,
    'target_cache_sha256': TARGET_CACHE_SHA256,
    'caption_data_sha256': texts_sha256.upper(),
    'normalizer_sha256': NORMALIZER_SHA256,
    'text_encoder_name': TEXT_ENCODER_NAME,
    'text_encoder_revision': ACTUAL_TEXT_ENCODER_REVISION,
    'text_encoder_state_sha256': TEXT_ENCODER_STATE_SHA256,
    'tokenizer_vocab_sha256': TOKENIZER_VOCAB_SHA256,
    'text_embeddings_sha256': TEXT_EMBEDDINGS_SHA256,
    'token_hidden_cache_sha256': TOKEN_HIDDEN_CACHE_SHA256,
    'fit_uid_order_sha256': FIT_UID_ORDER_SHA256,
    'stop_uid_order_sha256': hashlib.sha256('\n'.join(early_uids).encode()).hexdigest(),
    'development_only_uid_order_sha256': TEST_UID_ORDER_SHA256,
    'split_role_sha256': SPLIT_ROLE_SHA256,
    'fixed_inner_stop_negative_mapping_sha256': STOP_RANK_MAPPING_SHA256,
    'unsupported_source_filter': {
        'fit_rows_removed': unsupported_fit_source_rows_removed,
        'inner_stop_rows_removed': unsupported_stop_source_rows_removed,
    },
    'model_dim': 256, 'layers': 4, 'heads': 8, 'max_epochs': MAX_EPOCHS,
    'patience': PATIENCE, 'seed': SEED, 'velocity_weight': VELOCITY_WEIGHT,
    'phase_profile_weight': PHASE_WEIGHT, 'dynamic_weight': DYNAMIC_WEIGHT,
    'rank_weight': RANK_WEIGHT, 'rank_margin': RANK_MARGIN,
    'near_duplicate_cosine_threshold': NEAR_DUPLICATE_COSINE_THRESHOLD,
    'fit_stop_near_caption_filter': {
        'caption_key_normalizer': 'v5_ascii_tokens_lower_v1',
        'encoder_revision': ACTUAL_TEXT_ENCODER_REVISION,
        'threshold': NEAR_DUPLICATE_COSINE_THRESHOLD,
        'detected_cross_role_edges': cross_role_near_edge_count,
        'removed_inner_stop_rows': stop_rows_removed_for_near_duplicate,
        'remaining_cross_role_edges': remaining_cross_role_edges,
    },
    'pose_guardrail_position_max': POSE_GUARDRAIL,
    'pose_guardrail_baseline': 'fit-only frame mean evaluated on inner-stop role',
    'loss': 'group-balanced confidence-weighted pose MSE + 1.0 signed XYZ velocity MSE + 0.25 scale-normalized phase-profile error + 0.20 same-target filtered temporal hinge',
}
RUN_SIGNATURE = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
RESUME_PATH = WORK_ROOT / 'isl_v9_ranked_resume.pt'
history = []; start_epoch = 0; stale = 0; best_score = float('inf'); best_epoch = 0; best_state = None
if RESUME_PATH.is_file():
    saved = torch.load(RESUME_PATH, map_location=device, weights_only=False)
    if saved.get('signature') == RUN_SIGNATURE:
        decoder_core.load_state_dict(saved['current_state']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); history = saved['history']
        start_epoch = int(saved['epoch']); stale = int(saved['stale'])
        best_score = float(saved['best_score']); best_epoch = int(saved['best_epoch']); best_state = saved['best_state']
        print(f'Resumed matching V9 run at epoch {start_epoch}; signature={RUN_SIGNATURE}')
    else:
        raise RuntimeError('A V9 resume file exists but its source/split/objective signature differs; refusing implicit reuse.')

for epoch in range(start_epoch + 1, MAX_EPOCHS + 1):
    parallel.train(); started = time.perf_counter()
    train_losses = defaultdict(float); train_samples = 0; valid_rank_pairs = 0
    train_rank_filter_counts = {'candidate_pairs':0,'same_source_rejects':0,'same_caption_rejects':0,
                                'near_duplicate_rejects':0,'valid_pairs':0}
    for text_hidden, text_mask, target, confidence, batch_uids in train_loader:
        text_hidden = text_hidden.to(device, non_blocking=True); text_mask = text_mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True); confidence = confidence.to(device, non_blocking=True)
        batch_size = len(batch_uids)
        if batch_size > 1:
            shift, valid_pair_array, filter_counts = choose_rank_pairing(
                batch_uids, start_shift=int(np.random.randint(1,batch_size))
            )
            for key,value in filter_counts.items(): train_rank_filter_counts[key] += value
            negative_hidden = torch.roll(text_hidden, shifts=shift, dims=0).contiguous()
            negative_mask = torch.roll(text_mask, shifts=shift, dims=0).contiguous()
            ranking_mask = torch.as_tensor(valid_pair_array, dtype=torch.bool, device=device)
            negative_prediction = (parallel(negative_hidden, negative_mask)
                                   if int(ranking_mask.sum().item()) > 0 else None)
        else:
            ranking_mask = torch.zeros(1, dtype=torch.bool, device=device)
            negative_prediction = None
        optimizer.zero_grad(set_to_none=True)
        prediction = parallel(text_hidden, text_mask)
        loss, step_metrics = compute_motion_losses(
            prediction, target, confidence, GROUPS, negative_prediction=negative_prediction,
            ranking_mask=ranking_mask, velocity_weight=VELOCITY_WEIGHT,
            rank_weight=RANK_WEIGHT, rank_margin=RANK_MARGIN, phase_weight=PHASE_WEIGHT
        )
        if not torch.isfinite(loss): raise RuntimeError('Non-finite V9 training objective.')
        loss.backward(); torch.nn.utils.clip_grad_norm_(parallel.parameters(), 1.0); optimizer.step()
        batch_valid_rank_pairs = int(ranking_mask.sum().item())
        valid_rank_pairs += batch_valid_rank_pairs
        for k,v in step_metrics.items():
            if k == 'valid_rank_pairs':
                continue
            if k in ('ranking_loss','rank_true_win_fraction'):
                # These are means over valid pairs, so aggregate by the true pair count.
                train_losses[k] += v * batch_valid_rank_pairs
            else:
                train_losses[k] += v * batch_size
        train_samples += batch_size
    if valid_rank_pairs == 0:
        raise RuntimeError(f'V9 epoch {epoch} had no valid source/semantic-filtered ranking pairs; refusing to select a checkpoint.')
    scheduler.step()
    stop, _, _, _, _ = evaluate_cohort(early_uids, text_mode='true', include_ranking=True)
    if stop['selection_valid_rank_pairs'] <= 0:
        raise RuntimeError(f'V9 epoch {epoch} inner-stop evaluation had no valid source/semantic-filtered ranking pairs; refusing to select a checkpoint.')
    row = {'epoch':epoch, 'lr':float(optimizer.param_groups[0]['lr']),
           **{'fit_'+k:v/(valid_rank_pairs if k in ('ranking_loss','rank_true_win_fraction') else max(train_samples,1))
              for k,v in train_losses.items()},
           'fit_valid_rank_pairs':valid_rank_pairs,
           **{'early_'+k:v for k,v in stop.items()},
           'valid_rank_pairs':valid_rank_pairs, 'train_rank_filter_counts':train_rank_filter_counts,
           'pose_guardrail_pass':bool(stop['sample_mean_position'] <= POSE_GUARDRAIL),
           'seconds':round(time.perf_counter()-started,1)}
    history.append(row)
    current_score = stop['selection_score']
    pose_guardrail_pass = stop['sample_mean_position'] <= POSE_GUARDRAIL
    if pose_guardrail_pass and current_score < best_score:
        best_score=current_score; best_epoch=epoch; stale=0
        best_state={k:v.detach().cpu().clone() for k,v in decoder_core.state_dict().items()}
    else: stale += 1
    print(f"[V9 Epoch {epoch:02d}/{MAX_EPOCHS}] fit_objective={row.get('fit_loss',0):.5f}; "
          f"rank={row.get('fit_ranking_loss',0):.5f}; true-win={row.get('fit_rank_true_win_fraction',0):.1%}; "
          f"stop-score={stop['selection_score']:.5f}; pose={stop['sample_mean_position']:.5f}; "
          f"rank={stop['selection_ranking_loss']:.5f}; valid-ranks={stop['selection_valid_rank_pairs']}; "
          f"pose-guard={'PASS' if pose_guardrail_pass else 'FAIL'}; {row['seconds']}s", flush=True)
    torch.save({'signature':RUN_SIGNATURE,'epoch':epoch,'stale':stale,
                'current_state':{k:v.detach().cpu().clone() for k,v in decoder_core.state_dict().items()},
                'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'history':history,'best_score':best_score,'best_epoch':best_epoch,'best_state':best_state}, RESUME_PATH)
    if stale >= PATIENCE:
        print(f'V9 early stop at epoch {epoch}; best epoch={best_epoch}; score={best_score:.5f}')
        break
print(f'V9 training finished; best epoch={best_epoch}; selection_score={best_score:.5f}; signature={RUN_SIGNATURE}')
'''

EXPORT = r'''# Paired development-cohort comparison, hard geometry gates, and aggregate-only private export.
EXPORT_ROOT = WORK_ROOT / 'isl_v9_ranked_export'
assert EXPORT_ROOT.resolve().parent == WORK_ROOT.resolve(), 'V9 export path escaped Kaggle working directory.'
if EXPORT_ROOT.exists():
    shutil.rmtree(EXPORT_ROOT)
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = EXPORT_ROOT / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def paired_summary(control_scores, true_scores):
    control_scores = np.asarray(control_scores, dtype=np.float64)
    true_scores = np.asarray(true_scores, dtype=np.float64)
    assert control_scores.shape == true_scores.shape and control_scores.ndim == 1 and len(true_scores) > 1
    differences = control_scores - true_scores
    rng = np.random.default_rng(SEED)
    boot = np.empty(5000, dtype=np.float64)
    for i in range(len(boot)):
        boot[i] = rng.choice(differences, size=len(differences), replace=True).mean()
    se = differences.std(ddof=1) / np.sqrt(len(differences))
    return {'paired_rows':int(len(differences)),
            'mean_control_minus_true':float(differences.mean()),
            'fraction_true_better':float((differences>0).mean()),
            'standard_error':float(se),
            't_statistic':float(differences.mean()/max(se,1e-12)),
            'bootstrap_unit':'source_video_cluster; exactly one development row per cluster',
            'bootstrap_95pct_ci':[float(x) for x in np.percentile(boot,[2.5,97.5])]}

if best_state is None:
    report = {
        'schema_version':1,'model_arm':'motion_transformer_minilm_v9_temporal_ranked',
        'status':'no_checkpoint_passed_inner_stop_pose_guardrail',
        'checkpoint_exported':False,'development_cohort_evaluation_performed':False,
        'source_signature':signature_payload,'run_signature':RUN_SIGNATURE,
        'pose_guardrail_position_max':POSE_GUARDRAIL,'history':history,
        'role_counts':{'fit':len(fit_uids),'inner_stop':len(early_uids),
                       'development_only_rows_reused':len(test_uids)},
        'independent_final_test_available':False,
        'license_boundary':'Aggregate-only report; gated iSign clip-level material remains in mounted input and is not exported.'}
    (EXPORT_ROOT / 'metrics_summary_aggregate.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    archive_path = shutil.make_archive(str(WORK_ROOT / 'hackcessible_isl_v9_ranked_aggregate'), 'zip', root_dir=EXPORT_ROOT)
    print('No V9 checkpoint passed the inner-stop pose guardrail. Development cohort was not evaluated; no checkpoint was exported.')
    print('Aggregate status report:', EXPORT_ROOT / 'metrics_summary_aggregate.json')
    print('Aggregate archive:', archive_path, 'bytes:', Path(archive_path).stat().st_size)
else:
    decoder_core.load_state_dict(best_state)
    decoder_core.eval()

    # This repeatedly examined 498-row cohort is development-only, never a final test.
    dev_true, true_pos, true_vel, true_phase, true_score = evaluate_cohort(test_uids, text_mode='true')
    dev_shuffled, shuf_pos, shuf_vel, shuf_phase, shuf_score = evaluate_cohort(test_uids, text_mode='shuffled')
    dev_no_text, null_pos, null_vel, null_phase, null_score = evaluate_cohort(test_uids, text_mode='no_text')

    def evaluate_saved_v5_baseline(prediction_key):
        totals = {name:np.zeros(4,dtype=np.float64) for name in GROUPS}
        positions, velocities, phases, scores = [], [], [], []
        cache_hasher = hashlib.sha256()
        with torch.inference_mode():
            for uid in test_uids:
                relative_path = Path(str(sample_by_uid[uid]['predictions'][prediction_key]))
                prediction_path = (V5_ROOT / relative_path).resolve()
                if not prediction_path.is_relative_to(V5_ROOT.resolve()):
                    raise RuntimeError('V5 baseline path escaped the attached artifact root.')
                if not prediction_path.is_file():
                    raise RuntimeError(f'V5 paired development baseline cache is missing: {prediction_path.name}')
                raw_prediction = np.asarray(np.load(prediction_path,allow_pickle=False),dtype=np.float32)
                if raw_prediction.shape != (TMAX,POINTS,COORDS) or not np.isfinite(raw_prediction).all():
                    raise RuntimeError('V5 held-out baseline prediction has the wrong shape or non-finite values.')
                cache_hasher.update(np.ascontiguousarray(raw_prediction).tobytes(order='C'))
                pred_norm = ((raw_prediction-v5_mean)/v5_std).astype(np.float32)
                index = uid_to_index[uid]
                prediction = torch.from_numpy(pred_norm[None]).to(device)
                target = torch.from_numpy(targets[index:index+1]).to(device)
                confidence = torch.from_numpy(CONFIDENCE[index:index+1]).to(device)
                pos_per, _, _, _ = _grouped_sample_scores(prediction,target,confidence,GROUPS,0.0)
                vel_per, phase_per, dynamic_per, _ = temporal_grouped_scores(prediction,target,confidence)
                score_per = pos_per + DYNAMIC_WEIGHT * dynamic_per
                positions.extend(pos_per.cpu().numpy().astype(np.float64).tolist())
                velocities.extend(vel_per.cpu().numpy().astype(np.float64).tolist())
                phases.extend(phase_per.cpu().numpy().astype(np.float64).tolist())
                scores.extend(score_per.cpu().numpy().astype(np.float64).tolist())
                for name,(pn,pd,vn,vd,_) in grouped_terms(prediction,target,confidence)[2].items():
                    totals[name] += [pn.item(),pd.item(),vn.item(),vd.item()]
        pooled_position = sum(GROUPS[name][2]*totals[name][0]/max(totals[name][1],1.0) for name in GROUPS)
        pooled_velocity = sum(GROUPS[name][2]*totals[name][2]/max(totals[name][3],1.0) for name in GROUPS)
        return ({'position':float(pooled_position),'velocity':float(pooled_velocity),
                 'sample_mean_position':float(np.mean(positions)),
                 'sample_mean_velocity':float(np.mean(velocities)),
                 'sample_mean_phase_profile_error':float(np.mean(phases)),
                 'sample_mean_selection_score':float(np.mean(scores)),
                 'prediction_key':prediction_key},
                np.asarray(positions),np.asarray(velocities),np.asarray(phases),np.asarray(scores),cache_hasher.hexdigest().upper())

    v5_true, v5_pos, v5_vel, v5_phase, v5_score, v5_true_cache_sha = evaluate_saved_v5_baseline('true_text')
    frame_mean, mean_pos, mean_vel, mean_phase, mean_score, frame_mean_cache_sha = evaluate_saved_v5_baseline('train_frame_mean')
    v5_reported_position = float(v5_metrics['true_text']['group_balanced_normalized_mse'])
    mean_reported_position = float(v5_metrics['baselines']['train_frame_mean']['group_balanced_normalized_mse'])
    assert abs(v5_true['position']-v5_reported_position) <= 1e-4, 'Recomputed V5 checkpoint metric does not match its published aggregate.'
    assert abs(frame_mean['position']-mean_reported_position) <= 1e-4, 'Recomputed V5 frame-mean metric does not match its published aggregate.'
    v5_baseline_cache_sha = hashlib.sha256((v5_true_cache_sha+frame_mean_cache_sha).encode('ascii')).hexdigest().upper()

    # Positive differences mean V9 true text has lower error than the paired control.
    paired_shuffle_position = paired_summary(shuf_pos,true_pos)
    paired_shuffle_velocity = paired_summary(shuf_vel,true_vel)
    paired_shuffle_phase = paired_summary(shuf_phase,true_phase)
    paired_shuffle_selection = paired_summary(shuf_score,true_score)
    paired_no_text_position = paired_summary(null_pos,true_pos)
    paired_no_text_velocity = paired_summary(null_vel,true_vel)
    paired_no_text_phase = paired_summary(null_phase,true_phase)
    paired_v5_position = paired_summary(v5_pos,true_pos)
    paired_v5_velocity = paired_summary(v5_vel,true_vel)
    paired_v5_phase = paired_summary(v5_phase,true_phase)
    paired_frame_mean_position = paired_summary(mean_pos,true_pos)
    paired_frame_mean_velocity = paired_summary(mean_vel,true_vel)
    paired_frame_mean_phase = paired_summary(mean_phase,true_phase)

    # These bounds reject gross coordinate explosions, terminal jumps, and collapsed/stretched skeletons.
    EXPORT_THRESHOLDS = {
        'frame_max_abs_xy_global_max_lte':10.0,
        'frame_max_abs_xyz_global_max_lte':10.0,
        'endpoint_to_median_transition_speed_p95_lte':5.0,
        'endpoint_to_median_transition_speed_max_lte':10.0,
        'bone_length_p95_ratio_to_reference_range':[0.50,2.00],
        'per_clip_bone_length_p95_ratio_max_lte':3.00,
        'meaningful_true_vs_shuffled_position_mse_gain_gte':0.02,
        'true_text_motion_ratio_by_group_min':0.10,
        'true_vs_shuffled_velocity_ci_lower_gt':0.0,
        'true_vs_shuffled_phase_ci_lower_gt':0.0,
    }
    stability = dev_true['stability']
    bone_global_ratio = stability['bone_length_p95_ratio_to_reference']
    bone_clip_ratio_max = stability['per_clip_bone_length_p95_ratio_max']
    bone_group_checks = {
        name:bool(stats['scorable'] and
                  EXPORT_THRESHOLDS['bone_length_p95_ratio_to_reference_range'][0] <= stats['p95_ratio'] <= EXPORT_THRESHOLDS['bone_length_p95_ratio_to_reference_range'][1])
        for name,stats in stability['confidence_filtered_bone_groups'].items()
    }
    bone_edge_checks = {
        str(stats['edge_index']):bool(stats['scorable'] and
                  EXPORT_THRESHOLDS['bone_length_p95_ratio_to_reference_range'][0] <= stats['p95_ratio'] <= EXPORT_THRESHOLDS['bone_length_p95_ratio_to_reference_range'][1])
        for stats in stability['confidence_filtered_bone_edges']
    }
    geometry_checks = {
        'frame_magnitude':stability['frame_max_abs_xy_global_max'] <= EXPORT_THRESHOLDS['frame_max_abs_xy_global_max_lte'],
        'three_dimensional_coordinate_magnitude':stability['frame_max_abs_xyz_global_max'] <= EXPORT_THRESHOLDS['frame_max_abs_xyz_global_max_lte'],
        'endpoint_p95':stability['endpoint_to_median_transition_speed_p95'] <= EXPORT_THRESHOLDS['endpoint_to_median_transition_speed_p95_lte'],
        'endpoint_max':stability['endpoint_to_median_transition_speed_max'] <= EXPORT_THRESHOLDS['endpoint_to_median_transition_speed_max_lte'],
        'confidence_filtered_bone_support':stability['bone_unscorable_edge_count']==0 and stability['bone_unscorable_clip_group_count']==0,
        'confidence_filtered_bone_edges':all(bone_edge_checks.values()),
        'confidence_filtered_bone_groups':all(bone_group_checks.values()),
        'bone_length_global_p95':bone_global_ratio is not None and EXPORT_THRESHOLDS['bone_length_p95_ratio_to_reference_range'][0] <= bone_global_ratio <= EXPORT_THRESHOLDS['bone_length_p95_ratio_to_reference_range'][1],
        'bone_length_per_clip_p95':bone_clip_ratio_max is not None and bone_clip_ratio_max <= EXPORT_THRESHOLDS['per_clip_bone_length_p95_ratio_max_lte'],
    }
    geometry_gate_pass = all(geometry_checks.values())
    best_stop_row = next(row for row in history if int(row['epoch']) == int(best_epoch))
    pose_guardrail_pass = bool(best_stop_row['pose_guardrail_pass'])

    shuffle_position_ci = paired_shuffle_position['bootstrap_95pct_ci']
    shuffle_velocity_ci = paired_shuffle_velocity['bootstrap_95pct_ci']
    shuffle_phase_ci = paired_shuffle_phase['bootstrap_95pct_ci']
    v5_position_ci = paired_v5_position['bootstrap_95pct_ci']
    frame_position_ci = paired_frame_mean_position['bootstrap_95pct_ci']
    v5_velocity_ci = paired_v5_velocity['bootstrap_95pct_ci']
    frame_velocity_ci = paired_frame_mean_velocity['bootstrap_95pct_ci']
    quality_checks = {
        'pooled_position_better_than_v5':dev_true['position'] < v5_true['position'],
        'pooled_position_better_than_frame_mean':dev_true['position'] < frame_mean['position'],
        'pooled_velocity_better_than_v5':dev_true['velocity'] < v5_true['velocity'],
        'pooled_velocity_better_than_frame_mean':dev_true['velocity'] < frame_mean['velocity'],
        'paired_position_ci_better_than_v5':v5_position_ci[0] > 0.0,
        'paired_position_ci_better_than_frame_mean':frame_position_ci[0] > 0.0,
        'paired_velocity_ci_better_than_v5':v5_velocity_ci[0] > 0.0,
        'paired_velocity_ci_better_than_frame_mean':frame_velocity_ci[0] > 0.0,
        'true_vs_shuffled_position_gain_meaningful':paired_shuffle_position['mean_control_minus_true'] >= EXPORT_THRESHOLDS['meaningful_true_vs_shuffled_position_mse_gain_gte'],
        'true_vs_shuffled_position_ci_positive':shuffle_position_ci[0] > 0.0,
        'true_vs_shuffled_velocity_ci_positive':shuffle_velocity_ci[0] > EXPORT_THRESHOLDS['true_vs_shuffled_velocity_ci_lower_gt'],
        'true_vs_shuffled_phase_ci_positive':shuffle_phase_ci[0] > EXPORT_THRESHOLDS['true_vs_shuffled_phase_ci_lower_gt'],
        'true_text_motion_not_collapsed_all_groups':all(
            dev_true['motion_ratio_by_group'][name] >= EXPORT_THRESHOLDS['true_text_motion_ratio_by_group_min']
            for name in GROUPS
        ),
    }
    quantitative_acceptance_pass = all(quality_checks.values())
    checkpoint_exported = bool(quantitative_acceptance_pass and geometry_gate_pass and pose_guardrail_pass)

    report = {
        'schema_version':1,'model_arm':'motion_transformer_minilm_v9_temporal_ranked',
        'status':('experimental_candidate_only_not_a_working_model' if checkpoint_exported else 'checkpoint_export_blocked_by_quantitative_pose_or_geometry_gate'),
        'quantitative_acceptance_pass':bool(quantitative_acceptance_pass),
        'model_is_signer_validated':False,
        'architecture':'V7 non-autoregressive 48-frame Transformer decoder, retained unchanged',
        'objective':'group-balanced confidence-weighted coordinate-mean pose MSE + signed XYZ velocity MSE + 0.25 independently normalized aligned speed-profile error + 0.20 paired true-vs-filtered-negative temporal hinge; hinge excludes pose error',
        'selected_epoch':best_epoch,'best_inner_stop_selection_score':best_score,
        'inner_stop_pose_guardrail':{'baseline':'fit-only frame mean evaluated on inner-stop role','position_max':POSE_GUARDRAIL,'selected_epoch_position':float(best_stop_row['early_sample_mean_position']),'passed':pose_guardrail_pass},
        'source_signature':signature_payload,'run_signature':RUN_SIGNATURE,
        'fit_only_target_audit':FIT_TARGET_AUDIT,
        'role_counts':{'fit':len(fit_uids),'inner_stop':len(early_uids),
                       'development_only_rows_reused':len(test_uids)},
        'independent_final_test_available':False,
        'development_cohort':{'prior_evidence':'Previously evaluated by V7 and V8; development-only.',
                              'true_text':dev_true,'shuffled_text':dev_shuffled,
                              'no_text_ood_control_not_calibrated_unconditional_baseline':dev_no_text},
        'paired_development_uncertainty':{
            'shuffled_minus_true_position':paired_shuffle_position,
            'shuffled_minus_true_velocity':paired_shuffle_velocity,
            'shuffled_minus_true_phase_profile':paired_shuffle_phase,
            'shuffled_minus_true_selection_score':paired_shuffle_selection,
            'no_text_ood_minus_true_position':paired_no_text_position,
            'no_text_ood_minus_true_velocity':paired_no_text_velocity,
            'no_text_ood_minus_true_phase_profile':paired_no_text_phase,
            'v5_true_minus_v9_true_position':paired_v5_position,
            'v5_true_minus_v9_true_velocity':paired_v5_velocity,
            'v5_true_minus_v9_true_phase_profile':paired_v5_phase,
            'frame_mean_minus_v9_true_position':paired_frame_mean_position,
            'frame_mean_minus_v9_true_velocity':paired_frame_mean_velocity,
            'frame_mean_minus_v9_true_phase_profile':paired_frame_mean_phase,
        },
        'v5_reference_aggregate':{'true_text_recomputed_on_same_development_cohort':v5_true,
                                  'frame_mean_recomputed_on_same_development_cohort':frame_mean,
                                  'reported_v5_true_position':v5_reported_position,
                                  'reported_v5_frame_mean_position':mean_reported_position,
                                  'paired_prediction_cache_sha256':v5_baseline_cache_sha},
        'quality_checks':quality_checks,'geometry_export_thresholds':EXPORT_THRESHOLDS,
        'geometry_export_checks':geometry_checks,'confidence_filtered_bone_edge_checks':bone_edge_checks,
        'confidence_filtered_bone_group_checks':bone_group_checks,
        'geometry_export_gate_pass':bool(geometry_gate_pass),
        'checkpoint_exported':checkpoint_exported,
        'checkpoint_export_gates':{'quantitative_acceptance':bool(quantitative_acceptance_pass),
                                   'inner_stop_pose_guardrail':pose_guardrail_pass,
                                   'development_cohort_geometry':bool(geometry_gate_pass)},
        'follow_up_required':['fresh untouched source/caption cohort','fluent signer review','target-free inference path validation'],
        'history':history,
        'license_boundary':'Aggregate-only metrics and gated checkpoint. Do not redistribute gated iSign clip-level poses, video, captions, UIDs, aliases, or secrets.'
    }
    if checkpoint_exported:
        checkpoint_path = CHECKPOINT_DIR / 'motion_transformer_minilm_v9_temporal_ranked.pt'
        torch.save({'format':'Hackcessible ISL V9 ranked text-conditioning experiment',
                    'state_dict':best_state,'normalizer_mean':v5_mean,'normalizer_std':v5_std,
                    'config':signature_payload,'run_signature':RUN_SIGNATURE,
                    'status':'experimental_candidate_only_not_a_working_model'},checkpoint_path)
        report['checkpoint_sha256'] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest().upper()
    else:
        report['checkpoint_sha256'] = None
    (EXPORT_ROOT / 'metrics_summary_aggregate.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    archive_path = shutil.make_archive(str(WORK_ROOT / 'hackcessible_isl_v9_ranked_aggregate'), 'zip', root_dir=EXPORT_ROOT)
    print('=== V9 AGGREGATE DEVELOPMENT-COHORT EVALUATION ===')
    print('Independent final test available: false; this 498-row cohort was previously examined by V7/V8.')
    print('V9 pooled true-text position/velocity/phase:',dev_true['position'],dev_true['velocity'],dev_true['phase_profile_error'])
    print('V5 paired pooled position/velocity:',v5_true['position'],v5_true['velocity'])
    print('V5 frame-mean paired pooled position/velocity:',frame_mean['position'],frame_mean['velocity'])
    print('paired development uncertainty:',json.dumps(report['paired_development_uncertainty'],sort_keys=True))
    print('no-text OOD control:',json.dumps(dev_no_text,sort_keys=True))
    print('quality checks:',json.dumps(quality_checks,sort_keys=True))
    print('geometry checks:',json.dumps(geometry_checks,sort_keys=True))
    print('Status:',report['status'],'| checkpoint exported:',checkpoint_exported,'| signer validated: false')
    print('Aggregate-only report and archive; no per-row identifiers, captions, poses, or predictions were written.')
    print('Checkpoint SHA256:',report['checkpoint_sha256'])
    print('Aggregate archive:',archive_path,'bytes:',Path(archive_path).stat().st_size)
'''


def main() -> None:
    check_canonicalization_fixture()
    builder_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()
    setup = SETUP.replace('BUILDER_SHA256_VALUE', builder_sha)
    module_cell = MODULE_SOURCE + "\n\nprint('V9 Motion Transformer module SHA256:', '" + MODULE_SHA256 + "')\nprint('Module CPU preflight:', run_cpu_smoke_checks())\n"
    train = TRAIN.replace('MODULE_SHA256_VALUE', MODULE_SHA256).replace('BUILDER_SHA256_VALUE', builder_sha)
    export = EXPORT.replace('MODULE_SHA256_VALUE', MODULE_SHA256).replace('BUILDER_SHA256_VALUE', builder_sha)

    cells = [
        md('''# Hackcessible ISL text-to-pose — V9 Motion Transformer

This controlled same-role experiment retains the V7 non-autoregressive decoder and uses V5's source-alias-disjoint fit and inner-stop roles. V9 minimizes confidence-weighted coordinate-mean pose error plus signed XYZ velocity error and a separately normalized aligned speed-profile term. The true-caption versus filtered-negative hinge compares temporal error on the same target only. Stop selection uses that same full objective, with a separate frame-mean pose guardrail.

The 498-row cohort was already evaluated by V7 and V8, so V9 treats it as development evidence only. The attached cache has no untouched final cohort. The notebook reports paired uncertainty against V5 and its frame-mean baseline, shuffled-text motion, no-text OOD control, and confidence-filtered geometry checks. No signer has reviewed these generated signs.
'''),
        code(setup),
        # Fail on decoder/loss contract errors before reconstructing the large
        # pose cache or downloading MiniLM. The model module is self-contained
        # and the CPU smoke checks need only the setup imports above.
        code(module_cell),
        code(LOAD_V5_AND_POSES),
        code(TOKEN_CACHE),
        code(DATA_AND_LOSS),
        code(train),
        code(export),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "gpu",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Notebook SHA256: {hashlib.sha256(OUTPUT.read_bytes()).hexdigest().upper()}")
    print(f"Embedded V9 module SHA256: {MODULE_SHA256}")
    print(f"Total cells: {len(cells)}")


if __name__ == "__main__":
    main()
