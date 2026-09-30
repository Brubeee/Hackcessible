"""Build the self-contained Kaggle notebook for the controlled iSign V3 run."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "hackcessible-isl-text-to-pose-v3.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(r"""# Hackcessible ISL text-to-pose — controlled V3

This notebook tests whether a compact 75-point body-and-hands target, per-component supervision, and a frozen English sentence encoder improve held-out iSign outputs. It preserves the V2 sampled videos and split. The cached poses are loaded first; missing examples are recovered from the official gated archive using HTTP byte ranges, never by downloading the 159 GiB archive.

The three arms are: (1) trainable word encoder + position loss, (2) the same model + a small velocity term, and (3) frozen `all-MiniLM-L6-v2` text features + the same position and velocity losses. A frozen-embedding nearest-training-clip system, zero pose, and training frame-mean are comparison baselines. Every final held-out caption is paired with a deterministic different held-out caption for text-sensitivity evaluation.

Outputs are canonicalized body-and-hands poses, not full ISL. This model omits the face mesh and other nonmanual expression cues, and no coordinate metric or motion ratio proves that signs are intelligible. The iSign dataset is CC-BY-NC-SA-4.0 and restricted to research/non-commercial use; keep resulting model artifacts under those terms. Enable Kaggle Internet, select **GPU T4 x2**, attach `brubee/hackcessible-isign-pose-cache-v2`, and reuse the existing `HF_TOKEN` Kaggle secret.
"""),
    code(r"""# Environment, gated metadata access, and two-GPU preflight.
import gc, hashlib, json, math, os, random, re, shutil, subprocess, sys, time, zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

try:
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'])
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

try:
    from pose_format import Pose
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pose-format'])
    from pose_format import Pose

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers'])
    from transformers import AutoModel, AutoTokenizer

from kaggle_secrets import UserSecretsClient

SEED, LIMIT, TMAX, LMAX, BATCH = 17, 2048, 48, 40, 32
MAX_EPOCHS, PATIENCE, VELOCITY_WEIGHT = 30, 6, 0.05
REPO_ID = 'Exploration-Lab/iSign'
RUN_KEY_EXPECTED = '711ff3100b0f'
HF_TOKEN = UserSecretsClient().get_secret('HF_TOKEN')
assert HF_TOKEN and HF_TOKEN.startswith('hf_'), 'HF_TOKEN Kaggle secret is missing.'
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, 'Select Kaggle GPU T4 x2 before running.'
device = torch.device('cuda:0')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

repo_info = HfApi(token=HF_TOKEN).dataset_info(REPO_ID, files_metadata=True)
metadata_path = hf_hub_download(repo_id=REPO_ID, filename='iSign_v1.1.csv', repo_type='dataset', token=HF_TOKEN)
metadata = pd.read_csv(metadata_path)
POSE_PARTS = [f'iSign-poses_v1.1_part_{suffix}' for suffix in ('aa', 'ab', 'ac', 'ad')]
file_sizes = {item.rfilename: item.size for item in repo_info.siblings}
assert all(file_sizes.get(name) for name in POSE_PARTS), 'The official multipart pose archive listing is incomplete.'
print('Official iSign metadata:', metadata.shape, '| CUDA:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print('Archive parts will be read by byte range only; sizes GiB:', [round(file_sizes[n] / 1024**3, 2) for n in POSE_PARTS])
"""),
    code(r"""# Recreate the V2 2,048-video selection and load the exact 2,046 usable cache roster.
class HttpRangeFile:
    BLOCK = 1024 * 1024
    def __init__(self, url, token, size, session):
        self.url, self.token, self.size, self.pos, self.session = url, token, size, 0, session
        self.cache = {}
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset if whence == 2 else None)
        if pos is None or pos < 0: raise ValueError('Invalid ZIP seek.')
        self.pos = pos; return pos
    def _block(self, index):
        if index not in self.cache:
            start = index * self.BLOCK; end = min(start + self.BLOCK, self.size) - 1
            response = self.session.get(self.url, headers={'Authorization': f'Bearer {self.token}', 'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}, allow_redirects=True, stream=True, timeout=(30, 120))
            if response.status_code != 206:
                status = response.status_code; response.close()
                raise RuntimeError(f'Range request returned HTTP {status}; refusing a full-shard download.')
            data = b''.join(response.iter_content(chunk_size=256 * 1024)); response.close()
            if len(data) != end - start + 1: raise RuntimeError(f'Short ZIP range: wanted {end-start+1} bytes, got {len(data)}.')
            self.cache[index] = data
        return self.cache[index]
    def read(self, size=-1):
        if size is None or size < 0: size = self.size - self.pos
        size = min(size, max(0, self.size - self.pos)); result = bytearray()
        while size:
            index, offset = divmod(self.pos, self.BLOCK); block = self._block(index)
            take = min(size, len(block) - offset); result.extend(block[offset:offset+take]); self.pos += take; size -= take
        return bytes(result)

class ConcatRangeFile:
    def __init__(self, files):
        self.files = files; self.sizes = [item.size for item in files]; self.starts = []; total = 0
        for size in self.sizes: self.starts.append(total); total += size
        self.size, self.pos = total, 0
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset if whence == 2 else None)
        if pos is None or pos < 0: raise ValueError('Invalid concatenated ZIP seek.')
        self.pos = pos; return pos
    def read(self, size=-1):
        if size is None or size < 0: size = self.size - self.pos
        size = min(size, max(0, self.size - self.pos)); result = bytearray()
        while size:
            i = max(j for j, start in enumerate(self.starts) if start <= self.pos)
            local = self.pos - self.starts[i]; take = min(size, self.sizes[i] - local)
            self.files[i].seek(local); result.extend(self.files[i].read(take)); self.pos += take; size -= take
        return bytes(result)

range_session = requests.Session()
range_files = [HttpRangeFile(hf_hub_url(REPO_ID, name, repo_type='dataset', revision=repo_info.sha), HF_TOKEN, file_sizes[name], range_session) for name in POSE_PARTS]
archive = ConcatRangeFile(range_files)
with zipfile.ZipFile(archive, 'r') as zf:
    pose_names = [name for name in zf.namelist() if name.lower().endswith('.pose')]
pose_by_file = {Path(name).name: name for name in pose_names}

candidates = metadata.copy()
candidates['uid'] = candidates['uid'].astype(str)
candidates['text'] = candidates['text'].fillna('').astype(str).str.strip()
candidates['video_id'] = candidates['uid'].str.replace(r'-\d+$', '', regex=True)
candidates = candidates[candidates['text'].str.split().str.len().between(3, LMAX)]
candidates = candidates[~candidates['text'].str.match(r'(?i)^(page|unit|chapter)\s+\d+\s*$')]
candidates = candidates[candidates['uid'].map(lambda uid: uid + '.pose' in pose_by_file)]
candidates = candidates.drop_duplicates('video_id').reset_index(drop=True)
assert len(candidates) >= LIMIT, f'Need {LIMIT} distinct videos in the official metadata; found {len(candidates)}.'
selected = candidates.sample(n=LIMIT, random_state=SEED).reset_index(drop=True)
run_key = hashlib.sha256('|'.join(selected['uid']).encode()).hexdigest()[:12]
assert run_key == RUN_KEY_EXPECTED, f'V2 sampled UID hash changed: expected {RUN_KEY_EXPECTED}, got {run_key}.'

cache_candidates = list(Path('/kaggle/input').rglob('isign_pose_cache_' + run_key))
assert cache_candidates, 'Attach brubee/hackcessible-isign-pose-cache-v2 before running.'
CACHE_DIR = cache_candidates[0]
cached_paths = list(CACHE_DIR.rglob('*.npz'))
cached_uid_set = {path.stem for path in cached_paths}
cache_is_exact_roster = len(cached_uid_set) == 2046
selected_uids = set(selected['uid'])
if cache_is_exact_roster:
    assert cached_uid_set.issubset(selected_uids), 'The attached cache does not match the V2 selected UID set.'
    assert len(cached_uid_set) == 2046
print('Selected:', len(selected), '| cache files:', len(cached_uid_set), '| exact V2 cache roster:', cache_is_exact_roster, '| run key:', run_key)

def clear_range_cache():
    for ranged in range_files: ranged.cache.clear()

def decode_member(zf, uid):
    pose = Pose.read(zf.read(pose_by_file[uid + '.pose']))
    data = np.asarray(pose.body.data, dtype=np.float32)
    conf = np.asarray(pose.body.confidence, dtype=np.float32)
    if data.ndim == 4 and conf.ndim == 3:
        if data.shape[1] != conf.shape[1]: raise ValueError('person_axis')
        person = int(np.nanmean(conf, axis=(0, 2)).argmax()); data = data[:, person]; conf = conf[:, person]
    elif data.ndim == 4 and data.shape[1] == 1:
        data = data[:, 0]
        if conf.ndim == 3: conf = conf[:, 0]
    if data.ndim != 3 or conf.ndim != 2 or data.shape[0] < 2 or data.shape[-1] != 3 or data.shape[-2] != 576:
        raise ValueError('shape')
    if conf.shape != data.shape[:2]: raise ValueError('confidence_shape')
    t = data.shape[0]; positions = np.linspace(0.0, 1.0, TMAX, dtype=np.float32); q = positions * (t - 1)
    lo = np.floor(q).astype(np.int64); hi = np.minimum(lo + 1, t - 1); blend = (q - lo)[:, None]
    flat = np.nan_to_num(data.reshape(t, -1), nan=0.0, posinf=0.0, neginf=0.0)
    confidence = np.clip(np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    return (flat[lo] * (1.0 - blend) + flat[hi] * blend).astype(np.float32), (confidence[lo] * (1.0 - blend) + confidence[hi] * blend).astype(np.float32), int(t)

working_cache = Path('/kaggle/working/isign_pose_cache_' + run_key); working_cache.mkdir(parents=True, exist_ok=True)
usable_records, flat_poses, raw_confidences = [], [], []
skipped = Counter(); cache_hits = 0; range_decodes = 0
with zipfile.ZipFile(archive, 'r') as zf:
    for row in selected.itertuples(index=False):
        uid = str(row.uid)
        if cache_is_exact_roster and uid not in cached_uid_set:
            skipped['not_in_exact_v2_cache_roster'] += 1
            continue
        cache_path = CACHE_DIR / (uid + '.npz')
        if not cache_path.exists():
            matches = list(CACHE_DIR.rglob(uid + '.npz'))
            cache_path = matches[0] if matches else cache_path
        target_cache = working_cache / (uid + '.npz')
        try:
            raw_frame_count = None
            if cache_path.exists():
                with np.load(cache_path) as saved:
                    flat = np.asarray(saved['pose'], dtype=np.float32).copy()
                    conf = np.asarray(saved['confidence'], dtype=np.float32).copy()
                    if 'raw_frame_count' in saved: raw_frame_count = int(saved['raw_frame_count'])
                if flat.shape != (TMAX, 1728) or conf.shape != (TMAX, 576): raise ValueError('cached_shape')
                cache_hits += 1
            elif target_cache.exists():
                with np.load(target_cache) as saved:
                    flat = np.asarray(saved['pose'], dtype=np.float32).copy(); conf = np.asarray(saved['confidence'], dtype=np.float32).copy()
                    if 'raw_frame_count' in saved: raw_frame_count = int(saved['raw_frame_count'])
                if flat.shape != (TMAX, 1728) or conf.shape != (TMAX, 576): raise ValueError('working_cache_shape')
                cache_hits += 1
            else:
                flat, conf, raw_frame_count = decode_member(zf, uid); np.savez_compressed(target_cache, pose=flat, confidence=conf, raw_frame_count=np.asarray(raw_frame_count)); range_decodes += 1
            record = row._asdict(); record['raw_frame_count'] = raw_frame_count
            usable_records.append(record); flat_poses.append(flat); raw_confidences.append(conf)
        except Exception as exc:
            skipped[type(exc).__name__ + ':' + str(exc)[:80]] += 1
        finally:
            clear_range_cache()
        if len(usable_records) % 256 == 0:
            print(f'Loaded {len(usable_records)} usable poses; cache hits={cache_hits}, range decodes={range_decodes}', flush=True)

if len(usable_records) != 2046:
    raise RuntimeError(f'Expected the exact V2 usable roster of 2,046; got {len(usable_records)}. Skips={dict(skipped)}. Refusing to change the deterministic split.')
rows = pd.DataFrame(usable_records).reset_index(drop=True)
poses = np.stack(flat_poses).reshape(len(rows), TMAX, 576, 3).astype(np.float32)
confs = np.stack(raw_confidences).astype(np.float32)
del flat_poses, raw_confidences; gc.collect()
assert hashlib.sha256('|'.join(selected['uid']).encode()).hexdigest()[:12] == run_key
assert cache_is_exact_roster is False or set(rows['uid']) == cached_uid_set
print('Loaded exact V2 roster:', len(rows), '| 2,046 expected | skipped:', dict(skipped))
"""),
    code(r"""# Canonicalize the official 75-point body-and-hands representation and preserve the V2 split.
POINT_INDICES = np.asarray(list(range(33)) + list(range(501, 543)), dtype=np.int64)
assert len(POINT_INDICES) == 75 and len(set(POINT_INDICES.tolist())) == 75
TMAX, POINTS = 48, 75

def canonicalize_one(raw_pose, raw_confidence, min_anchor_conf=0.15):
    raw_pose = np.asarray(raw_pose, dtype=np.float32)
    raw_confidence = np.asarray(raw_confidence, dtype=np.float32)
    valid_coords = np.isfinite(raw_pose).all(axis=-1)
    safe_pose = np.nan_to_num(raw_pose, nan=0.0, posinf=0.0, neginf=0.0)
    point_conf = np.clip(np.nan_to_num(raw_confidence, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    selected_pose = safe_pose[:, POINT_INDICES, :].copy()
    selected_conf = point_conf[:, POINT_INDICES].copy() * valid_coords[:, POINT_INDICES]
    center = (safe_pose[:, 11, :] + safe_pose[:, 12, :]) * 0.5
    nose = safe_pose[:, 0, :]
    scale = np.linalg.norm(nose[:, :2] - center[:, :2], axis=-1)
    anchors_ok = (point_conf[:, 0] >= min_anchor_conf) & (point_conf[:, 11] >= min_anchor_conf) & (point_conf[:, 12] >= min_anchor_conf)
    anchors_ok &= valid_coords[:, 0] & valid_coords[:, 11] & valid_coords[:, 12] & np.isfinite(scale) & (scale > 1e-4)
    good = np.flatnonzero(anchors_ok)
    if len(good) < 2: return None
    frame_ix = np.arange(raw_pose.shape[0])
    center_interp = np.stack([np.interp(frame_ix, good, center[good, axis]) for axis in range(3)], axis=-1)
    scale_interp = np.exp(np.interp(frame_ix, good, np.log(scale[good].clip(1e-4))))
    reliability = np.where(anchors_ok, 1.0, 0.5).astype(np.float32)
    canonical = (selected_pose - center_interp[:, None, :]) / scale_interp[:, None, None]
    selected_conf *= reliability[:, None]
    canonical = np.nan_to_num(canonical, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return canonical, selected_conf.astype(np.float32), int(len(raw_pose) - len(good))

canonical_rows, canonical_conf_rows = [], []
anchor_interpolations = 0
for pose, confidence in zip(poses, confs):
    result = canonicalize_one(pose, confidence)
    if result is None: raise RuntimeError('A V2 cached pose has fewer than two reliable nose/shoulder frames.')
    canonical, canonical_conf, interpolated = result
    canonical_rows.append(canonical); canonical_conf_rows.append(canonical_conf); anchor_interpolations += interpolated
canonical_poses = np.stack(canonical_rows).astype(np.float32)
canonical_confs = np.stack(canonical_conf_rows).astype(np.float32)
del canonical_rows, canonical_conf_rows, poses, confs; gc.collect()

permutation = np.random.default_rng(SEED).permutation(len(rows))
val_count = max(1, int(round(len(rows) * 0.10)))
val_ix = permutation[:val_count]; outer_train_ix = permutation[val_count:]
assert len(outer_train_ix) == 1841 and len(val_ix) == 205
assert set(rows.iloc[outer_train_ix].video_id).isdisjoint(set(rows.iloc[val_ix].video_id))
assert 'onNSvHmicw0--0' in set(rows.iloc[val_ix].uid.astype(str)), 'Known V2 held-out reference UID is missing.'

token_pattern = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", re.IGNORECASE)
def text_key(text): return ' '.join(token_pattern.findall(str(text).lower()))

validation_text_keys = {text_key(rows.iloc[i].text) for i in val_ix}
caption_overlap_ix = [int(i) for i in outer_train_ix if text_key(rows.iloc[i].text) in validation_text_keys]
train_pool_ix = [int(i) for i in outer_train_ix if text_key(rows.iloc[i].text) not in validation_text_keys]
grouped_by_text = defaultdict(list)
for i in train_pool_ix: grouped_by_text[text_key(rows.iloc[i].text)].append(i)
caption_groups = list(grouped_by_text)
np.random.default_rng(SEED + 41).shuffle(caption_groups)
early_target = max(1, int(round(len(train_pool_ix) * 0.10)))
early_stop_ix, early_keys = [], set()
for key in caption_groups:
    if len(early_stop_ix) >= early_target: break
    early_stop_ix.extend(grouped_by_text[key]); early_keys.add(key)
fit_ix = [i for i in train_pool_ix if text_key(rows.iloc[i].text) not in early_keys]
assert fit_ix and early_stop_ix and set(fit_ix).isdisjoint(early_stop_ix)
assert not ({text_key(rows.iloc[i].text) for i in fit_ix} & validation_text_keys)
assert not ({text_key(rows.iloc[i].text) for i in fit_ix} & early_keys)
assert set(rows.iloc[fit_ix].video_id).isdisjoint(set(rows.iloc[early_stop_ix].video_id))

print(f'Canonical points: body 0:33, left hand 33:54, right hand 54:75; scale is 2D nose-to-shoulder-midpoint distance in x/y, applied to x/y/z; reliable-anchor frames interpolated={anchor_interpolations}')
print(f'Outer V2 split: {len(outer_train_ix)} train / {len(val_ix)} final holdout; exact train captions removed={len(caption_overlap_ix)}')
print(f'Effective gradient fit: {len(fit_ix)}; text-group early stop: {len(early_stop_ix)}; final holdout remains untouched for model selection.')
"""),
    code(r"""# Fit train-only coordinate statistics, word vocabulary, and frozen MiniLM sentence features.
fit_conf = canonical_confs[fit_ix]
fit_pose = canonical_poses[fit_ix]
weights = fit_conf[..., None]
den = weights.sum(axis=(0, 1))
normalizer_mean = (fit_pose * weights).sum(axis=(0, 1)) / np.maximum(den, 1e-6)
variance = (((fit_pose - normalizer_mean[None, None]) ** 2) * weights).sum(axis=(0, 1)) / np.maximum(den, 1e-6)
normalizer_std = np.sqrt(np.maximum(variance, 0.0)).clip(0.05).astype(np.float32)
normalizer_mean = normalizer_mean.astype(np.float32)
targets = np.nan_to_num((canonical_poses - normalizer_mean[None, None]) / normalizer_std[None, None], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
training_uids_used = rows.iloc[fit_ix].uid.astype(str).tolist()
normalizer_fit_uid_sha256 = hashlib.sha256('\n'.join(sorted(training_uids_used)).encode()).hexdigest()

train_conf = canonical_confs[fit_ix]
train_pose = canonical_poses[fit_ix]
frame_weight = train_conf[..., None]
frame_denom = frame_weight.sum(axis=0)
train_frame_mean = (train_pose * frame_weight).sum(axis=0) / np.maximum(frame_denom, 1e-6)
train_frame_mean = np.nan_to_num(train_frame_mean, nan=0.0).astype(np.float32)

MAX_TOKENS = LMAX
train_tokens = [token_pattern.findall(str(rows.iloc[i].text).lower())[:MAX_TOKENS] for i in fit_ix]
token_counts = Counter(token for sequence in train_tokens for token in sequence)
vocab = {'<pad>': 0, '<unk>': 1}
for token, _ in token_counts.most_common():
    if token not in vocab: vocab[token] = len(vocab)
word_ids = np.zeros((len(rows), MAX_TOKENS), dtype=np.int64)
word_masks = np.zeros((len(rows), MAX_TOKENS), dtype=np.float32)
for i, text in enumerate(rows.text):
    sequence = token_pattern.findall(str(text).lower())[:MAX_TOKENS]
    encoded = [vocab.get(token, 1) for token in sequence]
    word_ids[i, :len(encoded)] = encoded; word_masks[i, :len(encoded)] = 1.0
assert set(vocab) >= {'<pad>', '<unk>'}

MINILM_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
mini_tokenizer = AutoTokenizer.from_pretrained(MINILM_NAME)
mini_encoder = AutoModel.from_pretrained(MINILM_NAME).to(device).eval()
for parameter in mini_encoder.parameters(): parameter.requires_grad_(False)
mini_features = []
with torch.no_grad():
    for start in range(0, len(rows), 96):
        text_batch = rows.iloc[start:start+96].text.astype(str).tolist()
        encoded = mini_tokenizer(text_batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = mini_encoder(**encoded).last_hidden_state
        mask = encoded['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        pooled = nn.functional.normalize(pooled, p=2, dim=-1)
        mini_features.append(pooled.cpu().numpy().astype(np.float32))
mini_features = np.concatenate(mini_features, axis=0)
assert mini_features.shape[1] == 384 and np.isfinite(mini_features).all()
del mini_encoder, mini_tokenizer; gc.collect(); torch.cuda.empty_cache()

print('Vocab from gradient-fit text:', len(vocab), '| frozen MiniLM features:', mini_features.shape)
print('Normalizer shape:', normalizer_mean.shape, '| train UID SHA256:', normalizer_fit_uid_sha256)
"""),
    code(r"""# Define a direct sequence decoder and group-balanced position/velocity objective.
GROUPS = {'body': (0, 33, 0.20), 'left_hand': (33, 54, 0.40), 'right_hand': (54, 75, 0.40)}
HIDDEN, TIME_DIM = 256, 128

class TextToPose(nn.Module):
    def __init__(self, mode, vocab_size=None, feature_dim=384):
        super().__init__(); self.mode = mode
        if mode == 'random':
            self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
            self.text_encoder = nn.GRU(128, HIDDEN // 2, batch_first=True, bidirectional=True)
            self.context_projection = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh())
        else:
            self.context_projection = nn.Sequential(nn.Linear(feature_dim, HIDDEN), nn.Tanh())
        self.time_embedding = nn.Embedding(TMAX, TIME_DIM)
        self.initial_state = nn.Linear(HIDDEN, HIDDEN)
        self.decoder = nn.GRU(HIDDEN + TIME_DIM, HIDDEN, batch_first=True)
        self.output = nn.Linear(HIDDEN, POINTS * 3)

    def forward(self, text_input, text_mask):
        if self.mode == 'random':
            embedded = self.embedding(text_input)
            lengths = text_mask.sum(1).to(dtype=torch.long).clamp_min(1).cpu()
            packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.text_encoder(packed)
            encoded, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=text_input.shape[1])
            mask = text_mask.unsqueeze(-1)
            pooled = (encoded * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            context = self.context_projection(pooled)
        else:
            context = self.context_projection(text_input)
        temporal = self.time_embedding.weight.unsqueeze(0).expand(text_input.shape[0], -1, -1)
        decoder_input = torch.cat([context.unsqueeze(1).expand(-1, TMAX, -1), temporal], dim=-1)
        decoded, _ = self.decoder(decoder_input, torch.tanh(self.initial_state(context)).unsqueeze(0))
        return self.output(decoded)

def grouped_terms(prediction, target, confidence):
    prediction = prediction.reshape(-1, TMAX, POINTS, 3)
    target = target.reshape(-1, TMAX, POINTS, 3)
    confidence = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
    terms = {}
    for name, (start, end, group_weight) in GROUPS.items():
        pos_err = (prediction[:, :, start:end] - target[:, :, start:end]).square()
        pos_w = confidence[:, :, start:end, None].expand_as(pos_err)
        pos_num, pos_den = (pos_err * pos_w).sum(), pos_w.sum().clamp_min(1.0)
        vel_err = (prediction[:, 1:, start:end] - prediction[:, :-1, start:end] - (target[:, 1:, start:end] - target[:, :-1, start:end])).square()
        vel_w_point = torch.minimum(confidence[:, 1:, start:end], confidence[:, :-1, start:end])
        vel_w = vel_w_point[..., None].expand_as(vel_err)
        vel_num, vel_den = (vel_err * vel_w).sum(), vel_w.sum().clamp_min(1.0)
        terms[name] = (pos_num, pos_den, vel_num, vel_den, group_weight)
    position = sum(weight * (num / den) for num, den, _, _, weight in terms.values())
    velocity = sum(weight * (vnum / vden) for _, _, vnum, vden, weight in terms.values())
    return position, velocity, terms

@torch.no_grad()
def evaluate_loader(model, loader, velocity_weight):
    model.eval(); totals = {name: np.zeros(4, dtype=np.float64) for name in GROUPS}; motion_totals = {name: np.zeros(3, dtype=np.float64) for name in GROUPS}
    for text_input, text_mask, target, confidence in loader:
        text_input = text_input.to(device, non_blocking=True); text_mask = text_mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True); confidence = confidence.to(device, non_blocking=True)
        prediction = model(text_input, text_mask)
        terms = grouped_terms(prediction, target, confidence)[2]
        prediction = prediction.reshape(-1, TMAX, POINTS, 3); target = target.reshape(-1, TMAX, POINTS, 3)
        for name, (pn, pd, vn, vd, _) in terms.items():
            totals[name] += [pn.item(), pd.item(), vn.item(), vd.item()]
            start, end, _ = GROUPS[name]
            conf_pair = torch.minimum(confidence[:, 1:, start:end], confidence[:, :-1, start:end])
            pred_speed = torch.linalg.vector_norm(prediction[:, 1:, start:end] - prediction[:, :-1, start:end], dim=-1)
            ref_speed = torch.linalg.vector_norm(target[:, 1:, start:end] - target[:, :-1, start:end], dim=-1)
            motion_totals[name] += [float((pred_speed * conf_pair).sum()), float((ref_speed * conf_pair).sum()), float(conf_pair.sum())]
    pos_parts = {name: totals[name][0] / max(totals[name][1], 1.0) for name in GROUPS}
    vel_parts = {name: totals[name][2] / max(totals[name][3], 1.0) for name in GROUPS}
    position = sum(GROUPS[name][2] * pos_parts[name] for name in GROUPS)
    velocity = sum(GROUPS[name][2] * vel_parts[name] for name in GROUPS)
    pred_motion = {name: float(motion_totals[name][0] / max(motion_totals[name][2], 1.0)) for name in GROUPS}
    ref_motion = {name: float(motion_totals[name][1] / max(motion_totals[name][2], 1.0)) for name in GROUPS}
    motion_ratio = {name: float(pred_motion[name] / max(ref_motion[name], 1e-8)) for name in GROUPS}
    selection_score = position + VELOCITY_WEIGHT * velocity
    return {'position': float(position), 'velocity': float(velocity), 'objective': float(position + velocity_weight * velocity), 'selection_score': float(selection_score), 'position_by_group': pos_parts, 'velocity_by_group': vel_parts, 'prediction_motion_by_group': pred_motion, 'reference_motion_by_group': ref_motion, 'motion_ratio_by_group': motion_ratio}

def feature_arrays(mode):
    if mode == 'random': return word_ids, word_masks
    return mini_features, np.ones((len(rows), 1), dtype=np.float32)

def make_loader(indices, mode, shuffle):
    input_array, mask_array = feature_arrays(mode)
    dataset = TensorDataset(torch.from_numpy(input_array[indices]), torch.from_numpy(mask_array[indices]), torch.from_numpy(targets[indices].reshape(len(indices), TMAX, POINTS * 3)), torch.from_numpy(canonical_confs[indices]))
    generator = torch.Generator().manual_seed(SEED + 919)
    return DataLoader(dataset, batch_size=BATCH, shuffle=shuffle, num_workers=0, pin_memory=True, drop_last=False, generator=generator)

print('Loss group weights:', {name: value[2] for name, value in GROUPS.items()}, '| low velocity coefficient:', VELOCITY_WEIGHT)
"""),
    code(r"""# Train three controlled arms with internal text-group early stopping and per-epoch recovery checkpoints.
ARMS = [
    {'name': 'random_position', 'mode': 'random', 'velocity_weight': 0.0, 'seed': SEED + 101},
    {'name': 'random_position_motion', 'mode': 'random', 'velocity_weight': VELOCITY_WEIGHT, 'seed': SEED + 202},
    {'name': 'minilm_position_motion', 'mode': 'minilm', 'velocity_weight': VELOCITY_WEIGHT, 'seed': SEED + 303},
]
training_ix = np.asarray(fit_ix, dtype=np.int64); early_ix = np.asarray(early_stop_ix, dtype=np.int64)
runs = {}

def cpu_state(module): return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

def train_arm(config):
    arm_name, mode, velocity_weight = config['name'], config['mode'], config['velocity_weight']
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    core = TextToPose(mode, vocab_size=len(vocab)).to(device)
    model = nn.DataParallel(core, device_ids=[0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    train_loader = make_loader(training_ix, mode, True); stop_loader = make_loader(early_ix, mode, False)
    baseline = evaluate_loader(model, stop_loader, velocity_weight)
    exp_signature = hashlib.sha256(json.dumps({'code_version':'compact75-timepos-balanced-v2-common-selection','run':run_key,'arm':config,'points':POINT_INDICES.tolist(),'groups':GROUPS,'hidden':HIDDEN,'time_dim':TIME_DIM,'max_epochs':MAX_EPOCHS,'selection_velocity_weight':VELOCITY_WEIGHT,'fit_hash':normalizer_fit_uid_sha256},sort_keys=True).encode()).hexdigest()[:16]
    resume_path = Path(f'/kaggle/working/isl_v3_{arm_name}_{exp_signature}_resume.pt')
    history, start_epoch, stale = [], 0, 0
    best_selection_score, best_state = float('inf'), None
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved.get('signature') == exp_signature:
            core.load_state_dict(saved['current_state']); optimizer.load_state_dict(saved['optimizer'])
            history = saved['history']; start_epoch = int(saved['epoch']); stale = int(saved['stale'])
            best_selection_score = float(saved['best_selection_score']); best_state = saved['best_state']
            print(f'{arm_name}: resuming at epoch {start_epoch}/{MAX_EPOCHS}; best common internal score={best_selection_score:.5f}', flush=True)
    devices_seen = set()
    hook = core.register_forward_pre_hook(lambda module, inputs: devices_seen.add(inputs[0].device.index))
    probe = next(iter(train_loader)); model.eval()
    probe_x=probe[0].to(device); probe_m=probe[1].to(device); probe_y=probe[2].to(device); probe_c=probe[3].to(device)
    model.train(); optimizer.zero_grad(set_to_none=True)
    probe_prediction=model(probe_x,probe_m)
    assert probe_prediction.shape==probe_y.shape==(probe_x.shape[0],TMAX,POINTS*3), f'{arm_name}: minibatch output shape mismatch {tuple(probe_prediction.shape)}'
    assert torch.isfinite(probe_prediction).all(), f'{arm_name}: minibatch output contains NaN/Inf.'
    probe_position,probe_velocity,_=grouped_terms(probe_prediction,probe_y,probe_c)
    probe_loss=probe_position+velocity_weight*probe_velocity
    assert torch.isfinite(probe_loss), f'{arm_name}: minibatch loss is not finite.'
    probe_loss.backward()
    probe_grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    assert torch.isfinite(probe_grad_norm) and float(probe_grad_norm)>0.0, f'{arm_name}: minibatch gradient is zero or non-finite.'
    optimizer.zero_grad(set_to_none=True); model.eval()
    assert {0,1}.issubset(devices_seen), f'{arm_name}: minibatch forward/backward did not use both GPUs: {sorted(devices_seen)}'
    print(f'{arm_name}: minibatch smoke check OK; shape={tuple(probe_prediction.shape)} position={float(probe_position):.5f} velocity={float(probe_velocity):.5f} grad_norm={float(probe_grad_norm):.5f}',flush=True)
    started = time.time()
    for epoch in range(start_epoch + 1, MAX_EPOCHS + 1):
        model.train(); train_losses = []
        for text_input, text_mask, target, confidence in train_loader:
            text_input = text_input.to(device, non_blocking=True); text_mask = text_mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True); confidence = confidence.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            position, velocity, _ = grouped_terms(model(text_input, text_mask), target, confidence)
            loss = position + velocity_weight * velocity
            if not torch.isfinite(loss): raise RuntimeError(f'{arm_name}: non-finite loss.')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_losses.append((float(position.detach()), float(velocity.detach())))
        train_position = float(np.mean([item[0] for item in train_losses]))
        validation = evaluate_loader(model, stop_loader, velocity_weight)
        record = {'epoch': epoch, 'train_position': train_position, 'train_velocity': float(np.mean([item[1] for item in train_losses])), **{'early_' + key: value for key, value in validation.items()}}
        history.append(record)
        if validation['selection_score'] < best_selection_score:
            best_selection_score = validation['selection_score']; stale = 0; best_state = cpu_state(core)
        else:
            stale += 1
        payload = {'signature': exp_signature, 'epoch': epoch, 'current_state': cpu_state(core), 'optimizer': optimizer.state_dict(), 'history': history, 'stale': stale, 'best_selection_score': best_selection_score, 'best_state': best_state}
        tmp_path = resume_path.with_suffix('.pt.tmp'); torch.save(payload, tmp_path); tmp_path.replace(resume_path)
        print(f'{arm_name} epoch {epoch:02d}/{MAX_EPOCHS}: train_pos={train_position:.5f} early_pos={validation["position"]:.5f} early_vel={validation["velocity"]:.5f} common_score={validation["selection_score"]:.5f} hand_motion_ratio={validation["motion_ratio_by_group"]}', flush=True)
        if stale >= PATIENCE: break
    hook.remove()
    if best_state is None: raise RuntimeError(f'{arm_name} failed to save a best state.')
    if len(devices_seen) < 2: raise RuntimeError(f'{arm_name} did not use both GPUs: {sorted(devices_seen)}')
    core.load_state_dict(best_state); model.eval()
    best_epoch = min(history, key=lambda item: item['early_selection_score'])['epoch'] if history else start_epoch
    result = {'config': config, 'best_state': best_state, 'history': history, 'best_epoch': best_epoch, 'best_internal_selection_score': best_selection_score, 'best_internal_position': history[best_epoch - 1]['early_position'] if history and best_epoch <= len(history) else float('nan'), 'best_internal_velocity': history[best_epoch - 1]['early_velocity'] if history and best_epoch <= len(history) else float('nan'), 'baseline_internal': baseline, 'gpus_used': sorted(devices_seen), 'seconds': round(time.time() - started, 1), 'signature': exp_signature}
    del model, core, optimizer, train_loader, stop_loader; gc.collect(); torch.cuda.empty_cache()
    return result

for config in ARMS:
    runs[config['name']] = train_arm(config)

selected_arm = min(runs, key=lambda name: runs[name]['best_internal_selection_score'])
print('Internal-validation candidate selected for follow-up:', selected_arm)
print('Predeclared common score = group-balanced normalized position MSE + 0.05 x group-balanced normalized velocity MSE.')
print('Selection used only text-group early-stop data; final 205-video holdout was not used for selection.')
"""),
    code(r"""# Tiny 8-example optimization diagnostic; save outputs and component motion ratios.
tiny_ix = sorted(fit_ix, key=lambda i: (hashlib.sha256(f'{SEED}:{rows.iloc[i].uid}'.encode()).hexdigest(), str(rows.iloc[i].uid)))[:8]
assert len(tiny_ix) == 8
torch.manual_seed(SEED + 404); torch.cuda.manual_seed_all(SEED + 404)
tiny_core = TextToPose('minilm', vocab_size=len(vocab)).to(device)
tiny_model = nn.DataParallel(tiny_core, device_ids=[0, 1]); tiny_opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3, weight_decay=0.0)
tiny_loader = make_loader(np.asarray(tiny_ix, dtype=np.int64), 'minilm', False)

def tiny_eval():
    batch = next(iter(tiny_loader)); x, m, y, c = [item.to(device) for item in batch]
    with torch.no_grad():
        pos, vel, terms = grouped_terms(tiny_model(x, m), y, c)
        detail = {name: float(terms[name][0] / terms[name][1]) for name in GROUPS}
    return float(pos), detail

tiny_start, tiny_start_parts = tiny_eval(); tiny_history = []
for step in range(100):
    tiny_model.train()
    for x, m, y, c in tiny_loader:
        x=x.to(device); m=m.to(device); y=y.to(device); c=c.to(device)
        tiny_opt.zero_grad(set_to_none=True)
        pos, vel, _ = grouped_terms(tiny_model(x, m), y, c)
        loss = pos + VELOCITY_WEIGHT * vel; loss.backward(); torch.nn.utils.clip_grad_norm_(tiny_model.parameters(), 1.0); tiny_opt.step()
    if step in (0, 9, 24, 49, 99):
        score, parts = tiny_eval(); tiny_history.append({'step': step + 1, 'position': score, 'position_by_group': parts})
tiny_end, tiny_end_parts = tiny_eval()
with torch.no_grad():
    tx=torch.from_numpy(mini_features[tiny_ix]).to(device); tm=torch.ones((len(tiny_ix),1),device=device)
    tiny_prediction = tiny_core(tx, tm).cpu().numpy().reshape(len(tiny_ix), TMAX, POINTS, 3)
tiny_prediction = (tiny_prediction * normalizer_std[None,None] + normalizer_mean[None,None]).astype(np.float32)
tiny_reference = canonical_poses[tiny_ix]
tiny_confidence = canonical_confs[tiny_ix]
tiny_motion = {}
for name, (start, end, _) in GROUPS.items():
    w = np.minimum(tiny_confidence[:,1:,start:end], tiny_confidence[:,:-1,start:end])
    pred_v = np.linalg.norm(np.diff(tiny_prediction[:,:,start:end], axis=1), axis=-1)
    ref_v = np.linalg.norm(np.diff(tiny_reference[:,:,start:end], axis=1), axis=-1)
    tiny_motion[name] = {'prediction_weighted_mean': float((pred_v*w).sum()/max(w.sum(),1e-8)), 'reference_weighted_mean': float((ref_v*w).sum()/max(w.sum(),1e-8))}
tiny_report = {'uids':[str(rows.iloc[i].uid) for i in tiny_ix], 'steps':100, 'start_position':tiny_start, 'end_position':tiny_end, 'start_by_group':tiny_start_parts, 'end_by_group':tiny_end_parts, 'history':tiny_history, 'motion':tiny_motion, 'interpretation':'Optimization/pipeline check on eight fit examples only; not evidence of generalization or sign correctness.'}
np.savez_compressed('/kaggle/working/isl_v3_tiny_overfit.npz', uids=np.asarray(tiny_report['uids']), reference=tiny_reference, confidence=tiny_confidence, prediction=tiny_prediction)
Path('/kaggle/working/isl_v3_tiny_overfit.json').write_text(json.dumps(tiny_report, indent=2))
print('Tiny diagnostic position loss:', round(tiny_start,5), '->', round(tiny_end,5), '| component motion:', tiny_motion)
del tiny_model, tiny_core, tiny_opt; gc.collect(); torch.cuda.empty_cache()
"""),
    code(r"""# Evaluate all 205 unseen videos under true and shuffled text; export model and evaluator bundles.
EXPORT_ROOT = Path('/kaggle/working/isl_v3_export')
if EXPORT_ROOT.exists(): shutil.rmtree(EXPORT_ROOT)
(EXPORT_ROOT / 'references').mkdir(parents=True)
(EXPORT_ROOT / 'baselines').mkdir(parents=True)
(EXPORT_ROOT / 'predictions').mkdir(parents=True)
(EXPORT_ROOT / 'checkpoints').mkdir(parents=True)
(EXPORT_ROOT / 'diagnostics').mkdir(parents=True)

val_unique = {}
for i in val_ix:
    val_unique.setdefault(text_key(rows.iloc[int(i)].text), int(i))
unique_val_keys = sorted(val_unique)
assert len(unique_val_keys) >= 2, 'Need at least two distinct held-out caption groups for a text-shuffle counterfactual.'
counter_ix = []
for i in val_ix:
    uid = str(rows.iloc[int(i)].uid); own = text_key(rows.iloc[int(i)].text)
    choices = [key for key in unique_val_keys if key != own]
    rank = int(hashlib.sha256(f'{SEED}:{uid}'.encode()).hexdigest(), 16) % len(choices)
    counter_ix.append(val_unique[choices[rank]])
counter_ix = np.asarray(counter_ix, dtype=np.int64)
assert all(text_key(rows.iloc[int(i)].text) != text_key(rows.iloc[int(j)].text) for i,j in zip(val_ix,counter_ix))
fit_keys = {text_key(rows.iloc[int(i)].text) for i in fit_ix}
assert all(text_key(rows.iloc[int(j)].text) not in fit_keys for j in counter_ix), 'A counterfactual caption leaked into gradient-fit text.'

ranked_val = sorted((str(rows.iloc[int(i)].uid) for i in val_ix), key=lambda uid: (hashlib.sha256(f'{SEED}:{uid}'.encode()).hexdigest(), uid))
fixed_uids = ranked_val[:12]
assert len(fixed_uids) == 12
ref_dir = EXPORT_ROOT / 'references'; pred_root = EXPORT_ROOT / 'predictions'
safe_names = {}
for i in val_ix:
    uid = str(rows.iloc[int(i)].uid); slug = re.sub(r'[^A-Za-z0-9._-]+', '_', uid)
    safe_names[uid] = slug
    np.savez_compressed(ref_dir / (slug + '.npz'), pose=canonical_poses[int(i)], confidence=canonical_confs[int(i)])
np.save(EXPORT_ROOT / 'point_indices.npy', POINT_INDICES)

header = json.load(open('/kaggle/input/hackcessible-isign-pose-cache-v2/onNSvHmicw0_header.json', encoding='utf-8')) if Path('/kaggle/input/hackcessible-isign-pose-cache-v2/onNSvHmicw0_header.json').exists() else None
if header is None:
    local_headers = list(Path('/kaggle/input').rglob('onNSvHmicw0_header.json'))
    if local_headers: header = json.load(open(local_headers[0], encoding='utf-8'))
if header is None:
    try:
        header_path = Path('/kaggle/working/onNSvHmicw0_header.json')
        header_url = hf_hub_download(repo_id=REPO_ID, filename='pose_header.json', repo_type='dataset', token=HF_TOKEN)
        header = json.load(open(header_url, encoding='utf-8'))
    except Exception:
        header = None
if header is not None:
    edges = []
    for component in header.get('components', []):
        name = component.get('name', '')
        if name == 'POSE_LANDMARKS': dst_start, source_start, count = 0, 0, 33
        elif name == 'LEFT_HAND': dst_start, source_start, count = 33, 501, 21
        elif name == 'RIGHT_HAND': dst_start, source_start, count = 54, 522, 21
        else: continue
        for a,b in component.get('limbs', []):
            if 0 <= int(a) < count and 0 <= int(b) < count: edges.append((dst_start + int(a), dst_start + int(b)))
    edges = np.asarray(sorted(set(tuple(sorted(edge)) for edge in edges)), dtype=np.int64)
else:
    # MediaPipe body and hand landmark connections, matching the 75-point source topology.
    body_edges_local = [(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),(11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(27,31),(24,26),(26,28),(28,30),(30,32),(28,32),(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10)]
    hand_edges_local = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
    edges = np.asarray(body_edges_local + [(33+a,33+b) for a,b in hand_edges_local] + [(54+a,54+b) for a,b in hand_edges_local], dtype=np.int64)
np.save(EXPORT_ROOT / 'skeleton_edges.npy', edges)

zero_pose = np.zeros((TMAX, POINTS, 3), dtype=np.float32)
np.save(EXPORT_ROOT / 'baselines' / 'zero_pose.npy', zero_pose)
np.save(EXPORT_ROOT / 'baselines' / 'train_frame_mean.npy', train_frame_mean)
np.savez_compressed(EXPORT_ROOT / 'normalizer.npz', mean=normalizer_mean, std=normalizer_std)

def predict_for_indices(core, model, arm, indices):
    inputs, masks = feature_arrays(arm)
    out = []
    for start in range(0, len(indices), BATCH):
        batch_ix = indices[start:start+BATCH]
        x=torch.from_numpy(inputs[batch_ix]).to(device); m=torch.from_numpy(masks[batch_ix]).to(device)
        with torch.no_grad(): y=model(x,m).cpu().numpy().reshape(len(batch_ix),TMAX,POINTS,3)
        out.append(y)
    normalized = np.concatenate(out, axis=0)
    return (normalized * normalizer_std[None,None] + normalizer_mean[None,None]).astype(np.float32)

def export_metrics(prediction, reference, confidence):
    z_pred=(prediction-normalizer_mean[None,None])/normalizer_std[None,None]
    z_ref=(reference-normalizer_mean[None,None])/normalizer_std[None,None]
    result={}
    for name,(start,end,weight) in GROUPS.items():
        conf=confidence[:,:,start:end,None]
        sq=(z_pred[:,:,start:end]-z_ref[:,:,start:end])**2
        raw_sq=(prediction[:,:,start:end]-reference[:,:,start:end])**2
        den=max(float(conf.sum()*3),1e-8)
        vconf=np.minimum(confidence[:,1:,start:end],confidence[:,:-1,start:end])
        pm=np.linalg.norm(np.diff(prediction[:,:,start:end],axis=1),axis=-1)
        rm=np.linalg.norm(np.diff(reference[:,:,start:end],axis=1),axis=-1)
        motion_den=max(float(vconf.sum()),1e-8)
        pred_motion=float((pm*vconf).sum()/motion_den); ref_motion=float((rm*vconf).sum()/motion_den)
        result[name]={'normalized_mse':float((sq*conf).sum()/den),'canonical_rmse':float(np.sqrt((raw_sq*conf).sum()/den)),'prediction_motion':pred_motion,'reference_motion':ref_motion,'motion_ratio':float(pred_motion/max(ref_motion,1e-8))}
    result['group_balanced_normalized_mse']=float(sum(GROUPS[name][2]*result[name]['normalized_mse'] for name in GROUPS))
    return result

reference_all=canonical_poses[val_ix]
confidence_all=canonical_confs[val_ix]
retrieval_train_ix=np.asarray(fit_ix,dtype=np.int64)
similarity=mini_features[val_ix] @ mini_features[retrieval_train_ix].T
retrieval_ix=retrieval_train_ix[similarity.argmax(axis=1)]
retrieval_all=canonical_poses[retrieval_ix]
retrieval_uid=[str(rows.iloc[int(i)].uid) for i in retrieval_ix]
for i,uid in enumerate([str(rows.iloc[int(j)].uid) for j in val_ix]):
    np.save(EXPORT_ROOT / 'baselines' / (safe_names[uid] + '_semantic_retrieval.npy'), retrieval_all[i])

def row_record(i):
    raw_count=rows.iloc[int(i)].raw_frame_count
    return {'uid':str(rows.iloc[int(i)].uid),'source_video_id':str(rows.iloc[int(i)].video_id),'text':str(rows.iloc[int(i)].text),'raw_frame_count':None if pd.isna(raw_count) else int(raw_count),'resampled_frame_count':TMAX,'resampling_policy':'uniform_linear_48_endpoints_included' if not pd.isna(raw_count) else 'uniform_linear_48_endpoints_included; original_count_unavailable_in_v2_cache'}
outer_train_records=[row_record(i) for i in outer_train_ix]
outer_val_records=[row_record(i) for i in val_ix]
early_stop_uids=[str(rows.iloc[int(i)].uid) for i in early_stop_ix]
training_uids_used=[str(rows.iloc[int(i)].uid) for i in fit_ix]
topology={'components':[{'name':'body','start':0,'count':33},{'name':'left_hand','start':33,'count':21},{'name':'right_hand','start':54,'count':21}]}

all_metrics={'dataset':'Exploration-Lab/iSign v1.1','run_key':run_key,'usable_examples':len(rows),'outer_train':len(outer_train_ix),'final_validation':len(val_ix),'effective_fit':len(fit_ix),'internal_early_stop':len(early_stop_ix),'removed_exact_train_test_caption_rows':len(caption_overlap_ix),'fixed_qualitative_uids':fixed_uids,'selected_primary_arm':selected_arm,'candidate_selection':'Lowest common internal text-group validation score: group-balanced normalized position MSE + 0.05 times group-balanced normalized velocity MSE; the V2 205-video outer holdout is development evidence because it has been inspected across arms.','counterfactual_text_rule':'For each held-out UID, choose a different normalized caption from the held-out caption set using SHA256(SEED:UID); no final-held-out caption text occurs in the gradient-fit set.','resampling':{'resampled_frame_count':TMAX,'policy':'uniform_linear_48_endpoints_included','raw_frame_count':'Unavailable for original V2 cached examples because the cache contains only already-resampled pose and confidence arrays. Raw count is recorded when ranged fallback decoding exposes it.'},'arms':{},'baselines':{}}
all_metrics['baselines']['zero_pose']=export_metrics(np.broadcast_to(zero_pose,(len(val_ix),TMAX,POINTS,3)),reference_all,confidence_all)
all_metrics['baselines']['train_frame_mean']=export_metrics(np.broadcast_to(train_frame_mean,(len(val_ix),TMAX,POINTS,3)),reference_all,confidence_all)
all_metrics['baselines']['semantic_nearest_training_clip']=export_metrics(retrieval_all,reference_all,confidence_all)

for arm_name, run in runs.items():
    config=run['config']; mode=config['mode']; state=run['best_state']
    model_core=TextToPose(mode,vocab_size=len(vocab)).to(device); model_core.load_state_dict(state); model=nn.DataParallel(model_core,device_ids=[0,1]).eval()
    true_all=predict_for_indices(model_core,model,mode,val_ix)
    shuffled_all=predict_for_indices(model_core,model,mode,counter_ix)
    arm_dir=pred_root / arm_name; arm_dir.mkdir(parents=True,exist_ok=True)
    counter_text_by_uid={}
    sample_records=[]
    for j,idx in enumerate(val_ix):
        uid=str(rows.iloc[int(idx)].uid); slug=safe_names[uid]; cf_idx=int(counter_ix[j])
        counter_text=str(rows.iloc[cf_idx].text); counter_text_by_uid[uid]=counter_text
        donor_uid=str(rows.iloc[cf_idx].uid)
        true_path=arm_dir / (slug + '_true.npy'); shuffled_path=arm_dir / (slug + '_shuffled.npy')
        np.save(true_path,true_all[j]); np.save(shuffled_path,shuffled_all[j])
        raw_count=rows.iloc[int(idx)].raw_frame_count
        sample_records.append({'uid':uid,'source_video_id':str(rows.iloc[int(idx)].video_id),'text':str(rows.iloc[int(idx)].text),'counterfactual_text':counter_text,'counterfactual_source_uid':donor_uid,'raw_frame_count':None if pd.isna(raw_count) else int(raw_count),'resampled_frame_count':TMAX,'resampling_policy':'uniform_linear_48_endpoints_included' if not pd.isna(raw_count) else 'uniform_linear_48_endpoints_included; original_count_unavailable_in_v2_cache','semantic_retrieval_source_uid':retrieval_uid[j],'reference':{'path':'references/' + slug + '.npz','pose_key':'pose','confidence_key':'confidence'},'predictions':{'true_text':'predictions/' + arm_name + '/' + slug + '_true.npy','shuffled_text':'predictions/' + arm_name + '/' + slug + '_shuffled.npy','zero_pose':'baselines/zero_pose.npy','train_frame_mean':'baselines/train_frame_mean.npy','semantic_retrieval':'baselines/' + slug + '_semantic_retrieval.npy'}})
    true_metrics=export_metrics(true_all,reference_all,confidence_all)
    shuffled_metrics=export_metrics(shuffled_all,reference_all,confidence_all)
    mean_text_delta=np.mean(np.abs(true_all-shuffled_all),axis=(1,2,3))
    for j,sample in enumerate(sample_records): sample['true_vs_shuffled_mean_abs_delta']=float(mean_text_delta[j])
    best_history=next(item for item in run['history'] if item['epoch']==run['best_epoch'])
    all_metrics['arms'][arm_name]={'config':config,'best_epoch':run['best_epoch'],'best_internal_selection_score':run['best_internal_selection_score'],'best_internal_position':run['best_internal_position'],'best_internal_velocity':run['best_internal_velocity'],'best_internal_motion_ratio_by_group':best_history['early_motion_ratio_by_group'],'gpus_used':run['gpus_used'],'seconds':run['seconds'],'true_text':true_metrics,'shuffled_text':shuffled_metrics,'true_vs_shuffled_mean_abs_delta':{'median':float(np.median(mean_text_delta)),'mean':float(np.mean(mean_text_delta)),'per_uid':{sample_records[j]['uid']:float(mean_text_delta[j]) for j in range(len(sample_records))}},'text_sensitivity_gain_vs_shuffled_group_balanced_mse':float(shuffled_metrics['group_balanced_normalized_mse']-true_metrics['group_balanced_normalized_mse'])}
    checkpoint={'format':'Hackcessible iSign V3 compact body+hands','arm':arm_name,'config':config,'state_dict':state,'normalizer_mean':normalizer_mean,'normalizer_std':normalizer_std,'normalizer_fit_uid_sha256':normalizer_fit_uid_sha256,'point_indices':POINT_INDICES,'time_steps':TMAX,'topology':topology,'vocab':vocab if mode=='random' else None,'frozen_text_encoder':MINILM_NAME if mode=='minilm' else None,'license':'iSign CC-BY-NC-SA-4.0; research/non-commercial use only'}
    torch.save(checkpoint,EXPORT_ROOT / 'checkpoints' / (arm_name + '.pt'))
    manifest={'schema_version':1,'dataset':'Exploration-Lab/iSign v1.1','model_arm':arm_name,'evaluation_status':'development_evidence_not_final_test; the V2 outer holdout was inspected across candidate arms','selected_for_follow_up':arm_name==selected_arm,'selection_metric':'group-balanced normalized position MSE + 0.05 x group-balanced normalized velocity MSE on text-group internal validation','objective':{'group_weights':{'body':0.20,'left_hand':0.40,'right_hand':0.40},'velocity_coefficient':VELOCITY_WEIGHT,'training_velocity_coefficient':config['velocity_weight'],'confidence':'point confidence clipped to [0,1]; velocity confidence is the minimum adjacent-frame confidence','normalization':'train-only per-point/per-axis z-score fit on training_uids_used'},'training_uids_used':training_uids_used,'early_stop_uids':early_stop_uids,'split':{'strategy':'V2 seed 17 video-disjoint selection; exact normalized-caption overlap removed from gradient fit; internal early-stop split by caption groups','group_field':'source_video_id','train':outer_train_records,'validation':outer_val_records,'training_uids_used':training_uids_used,'early_stop_uids':early_stop_uids,'caption_overlap_removed_uids':[str(rows.iloc[int(i)].uid) for i in caption_overlap_ix]},'fixed_unseen_set':{'seed':SEED,'count':len(fixed_uids),'selection_rule':'sha256_rank_by_uid','uids':fixed_uids},'topology':topology,'normalizer':{'path':'normalizer.npz','mean_key':'mean','std_key':'std','fit_split':'train','fit_uid_sha256':normalizer_fit_uid_sha256,'fit_uid_hash_rule':'sha256(sorted training UIDs joined by newline)'},'coordinate_space':'canonical shoulder-centered pose; scale is 2D nose-to-shoulder-midpoint distance in x/y and is applied to x/y/z; predictions are denormalized canonical values, not z-scores','resampling':all_metrics['resampling'],'samples':sample_records,'metrics_path':'metrics_summary.json','license':'iSign CC-BY-NC-SA-4.0; research/non-commercial use only'}
    (EXPORT_ROOT / ('manifest_' + arm_name + '.json')).write_text(json.dumps(manifest,indent=2))
    del model, model_core; gc.collect(); torch.cuda.empty_cache()
    print(arm_name, '| internal best epoch', run['best_epoch'], '| true group-balanced MSE', round(true_metrics['group_balanced_normalized_mse'],5), '| shuffled', round(shuffled_metrics['group_balanced_normalized_mse'],5), '| text delta', round(float(np.mean(mean_text_delta)),5))

(EXPORT_ROOT / 'metrics_summary.json').write_text(json.dumps(all_metrics,indent=2))
tiny_src=Path('/kaggle/working/isl_v3_tiny_overfit.npz')
tiny_json_src=Path('/kaggle/working/isl_v3_tiny_overfit.json')
if tiny_src.exists(): shutil.copy2(tiny_src,EXPORT_ROOT/'diagnostics'/'tiny_overfit.npz')
if tiny_json_src.exists(): shutil.copy2(tiny_json_src,EXPORT_ROOT/'diagnostics'/'tiny_overfit.json')
(EXPORT_ROOT / 'README.txt').write_text('V3 outputs contain canonicalized body and hands only (75 points; 48 frames). Face mesh and nonmanual facial grammar are not modeled. Predictions are denormalized canonical coordinates. The 2,046 usable videos reproduce the V2 seed-17 video split. The 12 SHA-ranked fixed unseen UIDs are a subset of the full 205 held-out videos. Motion and coordinate scores do not demonstrate sign intelligibility; have an ISL-fluent signer review rendered outputs. Dataset license: CC-BY-NC-SA-4.0, research/non-commercial use only.\n')
archive_path=shutil.make_archive('/kaggle/working/hackcessible_isl_v3_artifacts','zip',root_dir=EXPORT_ROOT)
print('V3 artifacts:',archive_path,'| bytes:',Path(archive_path).stat().st_size)
print('Selected by internal validation:',selected_arm,'| all held-out rows scored:',len(val_ix),'| fixed qualitative set:',len(fixed_uids))
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
        "kaggle": {"accelerator": "GPU", "dataSources": [], "isInternetEnabled": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH} with {len(cells)} cells.")
