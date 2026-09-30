"""Build the controlled, source-disjoint iSign V5 Kaggle notebook."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "hackcessible-isl-text-to-pose-v3.ipynb"
OUTPUT = ROOT / "hackcessible-isl-text-to-pose-v5.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


v3 = json.loads(SOURCE.read_text(encoding="utf-8"))
v3_code = ["".join(cell["source"]) for cell in v3["cells"]]


V5_DATA = r'''# Freeze a source-disjoint fresh test and load only the selected pose files.
from collections import OrderedDict

SEED, V2_LIMIT, FRESH_TRAIN_TARGET, FRESH_TEST_TARGET = 17, 2048, 8000, 500
FRESH_MIN_COVERAGE = 0.98
TMAX, LMAX, BATCH = 48, 40, 32
MAX_EPOCHS, PATIENCE, VELOCITY_WEIGHT = 30, 6, 0.05
RUN_KEY_EXPECTED = '711ff3100b0f'

class HttpRangeFile:
    BLOCK = 1024 * 1024
    def __init__(self, url, token, size, session, max_blocks=96):
        self.url, self.token, self.size, self.pos, self.session = url, token, size, 0, session
        self.cache, self.max_blocks = OrderedDict(), int(max_blocks)
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
            while len(self.cache) > self.max_blocks: self.cache.popitem(last=False)
        self.cache.move_to_end(index)
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
print('Official pose archive members:', len(pose_names), '| ranged-read LRU:', 96, 'MiB per shard.')

def source_group_from_uid(uid):
    """Conservative iSign source ID parser; returns (source_id, rule, valid)."""
    uid = str(uid).strip()
    match = re.fullmatch(r'(?P<core>.+)--(?P<seq>\d+)', uid)
    if match:
        core = match.group('core')
        return (core, 'inferred_youtube11_double_dash_sequence', True) if len(core) == 11 and re.fullmatch(r'[A-Za-z0-9_-]{11}', core) else (None, 'invalid_double_dash_core', False)
    match = re.fullmatch(r'(?P<core>.+)-(?P<seq>\d+)', uid)
    if match:
        core = match.group('core')
        return (core, 'documented_hex12_id-sequence', True) if len(core) == 12 and re.fullmatch(r'[0-9a-fA-F]{12}', core) else (None, 'invalid_hex12_sequence_core', False)
    match = re.fullmatch(r'(?P<core>.+)_e(?P<seq>\d+)', uid)
    if match:
        core = match.group('core')
        return (core, 'inferred_def_example_suffix', True) if len(core) == 11 and re.fullmatch(r'[A-Za-z0-9_-]{11}', core) else (None, 'invalid_example_core', False)
    match = re.fullmatch(r'(?P<core>.+)_d', uid)
    if match:
        core = match.group('core')
        return (core, 'inferred_def_description_suffix', True) if len(core) == 11 and re.fullmatch(r'[A-Za-z0-9_-]{11}', core) else (None, 'invalid_description_core', False)
    # `_w` may be part of an 11-character YouTube ID, so it is never stripped.
    if len(uid) == 11 and re.fullmatch(r'[A-Za-z0-9_-]{11}', uid):
        return uid, 'bare_youtube_id', True
    if len(uid) == 12 and re.fullmatch(r'[0-9a-fA-F]{12}', uid):
        return uid, 'bare_hex12_id', True
    return None, 'unsupported_uid_source_id_grammar', False

def conservative_source_aliases(uid, source_id=None):
    """Keep whole UIDs and every length-validated plausible source core reserved."""
    uid = str(uid).strip(); aliases = {uid}
    if source_id is not None and str(source_id).strip(): aliases.add(str(source_id).strip())
    parsed, _, valid = source_group_from_uid(uid)
    if valid:
        aliases.add(parsed)
        # For id--N, the official generic id-N grammar also admits the
        # literal 12-character source candidate id-. Reserve it as an alias
        # without asserting that it is the authoritative source ID.
        if re.fullmatch(r'.+--\d+', uid) and len(parsed) == 11:
            aliases.add(parsed + '-')
    # Unsupported 11-character-core-N rows are not eligible primary IDs, but
    # reserve their plausible core when protecting the legacy V2 roster.
    single_dash = re.fullmatch(r'(?P<core>.+)-\d+', uid)
    if single_dash:
        core = single_dash.group('core')
        if len(core) == 11 and re.fullmatch(r'[A-Za-z0-9_-]{11}', core):
            aliases.add(core)
    match = re.fullmatch(r'(?P<core>.+)_e\d+', uid) or re.fullmatch(r'(?P<core>.+)_d', uid)
    if match and len(match.group('core')) == 11 and re.fullmatch(r'[A-Za-z0-9_-]{11}', match.group('core')):
        aliases.add(match.group('core'))
    # `_w` is ambiguous; reserve a stripped alias only when its source ID validates.
    if uid.endswith('_w'):
        candidate = uid[:-2]
        if (len(candidate) == 11 and re.fullmatch(r'[A-Za-z0-9_-]{11}', candidate)) or (len(candidate) == 12 and re.fullmatch(r'[0-9a-fA-F]{12}', candidate)):
            aliases.add(candidate)
    return aliases

_uid_parser_examples = {
    'onNSvHmicw0--0': 'onNSvHmicw0',
    '9zQleTPz3oQ--1': '9zQleTPz3oQ',
    '-W3Sefpg9l4_e2': '-W3Sefpg9l4',
    'SEKRNyZ9Sgg_d': 'SEKRNyZ9Sgg',
    '1782bea75c7d-7': '1782bea75c7d',
    'yGr2Iv6Lf_w--0': 'yGr2Iv6Lf_w',
}
for _uid, _expected in _uid_parser_examples.items():
    _actual, _rule, _valid = source_group_from_uid(_uid)
    assert _valid and _actual == _expected, (_uid, _actual, _rule)
print('UID parser examples passed:', _uid_parser_examples)

metadata = metadata.copy()
assert {'uid', 'text'}.issubset(metadata.columns), f'Official CSV columns changed: {list(metadata.columns)}'
metadata['uid'] = metadata['uid'].astype(str).str.strip()
metadata['text'] = metadata['text'].fillna('').astype(str).str.strip()
if metadata['uid'].duplicated().any():
    raise RuntimeError(f'Official metadata contains {int(metadata.uid.duplicated().sum())} duplicate UIDs; refusing ambiguous joins.')

source_parse = [source_group_from_uid(uid) for uid in metadata['uid']]
metadata['parsed_source_video_id'] = [item[0] for item in source_parse]
metadata['source_uid_rule'] = [item[1] for item in source_parse]
metadata['source_uid_supported'] = [item[2] for item in source_parse]
if 'video_id' in metadata.columns:
    metadata['source_video_id'] = metadata['video_id'].fillna('').astype(str).str.strip()
    metadata['source_video_id_provenance'] = 'official_csv_video_id'
    metadata['source_uid_mismatch'] = metadata['source_uid_supported'] & (metadata['parsed_source_video_id'] != metadata['source_video_id'])
    metadata['source_uid_supported'] = metadata['source_video_id'].ne('')
    metadata.loc[metadata['source_uid_supported'], 'source_uid_rule'] = 'official_csv_video_id'
else:
    metadata['source_video_id'] = metadata['parsed_source_video_id']
    metadata['source_video_id_provenance'] = 'uid_suffix_parser_v1'
    metadata['source_uid_mismatch'] = False
if metadata['source_video_id'].fillna('').eq('').any():
    metadata.loc[metadata['source_video_id'].fillna('').eq(''), 'source_uid_supported'] = False
if metadata['source_uid_mismatch'].any():
    mismatch_examples = metadata.loc[metadata['source_uid_mismatch'], ['uid','parsed_source_video_id','source_video_id']].head(10).to_dict(orient='records')
    print('Explicit official video_id overrides UID inference after conflict audit; mismatch rows:', int(metadata['source_uid_mismatch'].sum()), '| examples:', mismatch_examples)

token_pattern = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
def text_key(text): return ' '.join(token_pattern.findall(str(text).lower()))

eligible = metadata.copy()
eligible = eligible[eligible['text'].str.split().str.len().between(3, LMAX)]
eligible = eligible[~eligible['text'].str.match(r'(?i)^(page|unit|chapter)\s+\d+\s*$')]
eligible = eligible[eligible['uid'].map(lambda uid: uid + '.pose' in pose_by_file)].copy()
eligible['text_key'] = eligible['text'].map(text_key)
eligible['source_video_id'] = eligible['source_video_id'].fillna('').astype(str)

# Reproduce the old V2 sampled roster exactly, including its historical grouping rule.
v2_candidates = eligible.copy()
v2_candidates['legacy_video_id'] = v2_candidates['uid'].str.replace(r'-\d+$', '', regex=True)
v2_candidates = v2_candidates.drop_duplicates('legacy_video_id').reset_index(drop=True)
if len(v2_candidates) < V2_LIMIT:
    raise RuntimeError(f'Only {len(v2_candidates)} legacy V2 candidates are available; need {V2_LIMIT}.')
selected_v2 = v2_candidates.sample(n=V2_LIMIT, random_state=SEED).reset_index(drop=True)
run_key = hashlib.sha256('|'.join(selected_v2['uid']).encode()).hexdigest()[:12]
assert run_key == RUN_KEY_EXPECTED, f'V2 roster hash changed: {run_key} != {RUN_KEY_EXPECTED}'
selected_v2['source_video_id'] = [
    str(source_id) if supported and str(source_id).strip() else 'unresolved_uid:' + str(uid)
    for source_id, supported, uid in zip(selected_v2['source_video_id'], selected_v2['source_uid_supported'], selected_v2['uid'])
]

cache_candidates = list(Path('/kaggle/input').rglob('isign_pose_cache_' + run_key))
assert cache_candidates, 'Attach the verified Hackcessible iSign Pose Cache V2 dataset.'
CACHE_DIR = cache_candidates[0]
cached_paths = list(CACHE_DIR.rglob('*.npz'))
cached_by_uid = {path.stem: path for path in cached_paths}
selected_v2_uids = set(selected_v2['uid'])
assert len(cached_by_uid) == 2046 and set(cached_by_uid).issubset(selected_v2_uids), 'V2 cache must be the verified exact 2,046-file roster.'
old_source_ids = set(selected_v2['source_video_id'])
old_text_keys = set(selected_v2['text_key'])
old_source_aliases = set().union(*(conservative_source_aliases(row.uid, row.source_video_id) for row in selected_v2.itertuples()))
old_uid_to_row = selected_v2.set_index('uid').to_dict(orient='index')
print('Official CSV shape/columns:', metadata.shape, list(metadata.columns))
print('Source ID provenance:', metadata['source_video_id_provenance'].iloc[0], '| UID parse families:', metadata['source_uid_rule'].value_counts().to_dict())
print('Unsupported metadata UIDs quarantined:', int((~metadata['source_uid_supported']).sum()), '| V2 source groups reserved:', len(old_source_ids), '| V2 captions reserved:', len(old_text_keys))
print('Exact V2 roster:', len(selected_v2), '| V2 cache:', len(cached_by_uid), '| run key:', run_key)

# Select test first: one caption per source, unique captions, and no source/text overlap with any V2 row.
if 'video_id' in metadata.columns:
    test_family_mask = eligible['source_uid_supported']
else:
    test_family_mask = eligible['source_uid_rule'].isin({'inferred_youtube11_double_dash_sequence','documented_hex12_id-sequence'})
fresh_candidates = eligible[eligible['source_uid_supported'] & test_family_mask & ~eligible['text_key'].isin(old_text_keys)].copy()
fresh_candidates['test_rank'] = [hashlib.sha256(f'{SEED}:fresh-test:{row.source_video_id}:{row.uid}'.encode()).hexdigest() for row in fresh_candidates.itertuples()]
fresh_candidates = fresh_candidates.sort_values(['test_rank', 'uid'], kind='mergesort')
fresh_test_rows, test_sources, test_texts, test_source_aliases = [], set(), set(), set()
for record in fresh_candidates.to_dict(orient='records'):
    source_id, caption = record['source_video_id'], record['text_key']
    aliases = conservative_source_aliases(record['uid'], source_id)
    if aliases & old_source_aliases or aliases & test_source_aliases or source_id in test_sources or caption in test_texts:
        continue
    record['source_aliases'] = sorted(aliases)
    fresh_test_rows.append(record); test_sources.add(source_id); test_texts.add(caption); test_source_aliases.update(aliases)
    if len(fresh_test_rows) == FRESH_TEST_TARGET: break
if len(fresh_test_rows) != FRESH_TEST_TARGET:
    raise RuntimeError(f'Only {len(fresh_test_rows)} fresh source/caption-disjoint test rows exist; need {FRESH_TEST_TARGET}.')

# Pick 8,000 fresh train clips after freezing test groups. Round-robin across source videos, max 8 clips/source.
train_candidates = eligible[
    eligible['source_uid_supported']
    & ~eligible['text_key'].isin(old_text_keys | test_texts)
].copy()
train_candidates = train_candidates[
    train_candidates.apply(lambda row: not (conservative_source_aliases(row['uid'], row['source_video_id']) & (old_source_aliases | test_source_aliases)), axis=1)
].copy()
by_source = defaultdict(list)
for record in train_candidates.to_dict(orient='records'):
    record['source_aliases'] = sorted(conservative_source_aliases(record['uid'], record['source_video_id']))
    by_source[record['source_video_id']].append(record)
source_order = sorted(by_source, key=lambda source: (hashlib.sha256(f'{SEED}:fresh-train-source:{source}'.encode()).hexdigest(), source))
for source in source_order:
    by_source[source].sort(key=lambda record: (hashlib.sha256(f'{SEED}:fresh-train-uid:{record["uid"]}'.encode()).hexdigest(), record['uid']))
fresh_train_rows = []
for depth in range(8):
    for source in source_order:
        if depth < len(by_source[source]):
            fresh_train_rows.append(by_source[source][depth])
            if len(fresh_train_rows) == FRESH_TRAIN_TARGET: break
    if len(fresh_train_rows) == FRESH_TRAIN_TARGET: break
if len(fresh_train_rows) != FRESH_TRAIN_TARGET:
    raise RuntimeError(f'Only {len(fresh_train_rows)} fresh source-disjoint train rows are available under the 8/source cap; need {FRESH_TRAIN_TARGET}.')
fresh_train_sources = {record['source_video_id'] for record in fresh_train_rows}
fresh_train_texts = {record['text_key'] for record in fresh_train_rows}
fresh_train_aliases = set().union(*(set(record['source_aliases']) for record in fresh_train_rows))
assert not (old_source_aliases & test_source_aliases or old_source_aliases & fresh_train_aliases or fresh_train_aliases & test_source_aliases)
assert not (old_text_keys & test_texts or fresh_train_texts & test_texts)
assert len({record['uid'] for record in fresh_train_rows}) == FRESH_TRAIN_TARGET
print('Fresh selection:', len(fresh_train_rows), 'train /', len(fresh_test_rows), 'test | sources:', len(fresh_train_sources), '/', len(test_sources))
print('Selection audit: source overlap old/train/test = 0; exact normalized caption overlap old/train/test = 0')
predecode_test_uids=sorted(str(record['uid']) for record in fresh_test_rows)
fixed_uids=sorted(predecode_test_uids,key=lambda uid:(hashlib.sha256(f'{SEED}:{uid}'.encode()).hexdigest(),uid))[:12]

# Persist the immutable selected roster before any network decoding. Decode failures
# are recorded against these rows; selections are never silently replaced.
def selection_record(record):
    return {'uid':str(record['uid']),'source_video_id':str(record['source_video_id']),'source_aliases':sorted(set(record['source_aliases'])),'source_uid_rule':str(record['source_uid_rule']),'text':str(record['text']),'normalized_caption_key':str(record['text_key'])}
PREDECODE_SELECTION_PATH=Path('/kaggle/working/isl_v5_selection_manifest_predecode.json')
PREDECODE_SELECTION_PATH.write_text(json.dumps({'seed':SEED,'v2_run_key':run_key,'fresh_test_rule':'primary UID families only when official video_id is absent; candidate aliases and normalized captions disjoint; deterministic UID hash rank','fresh_train_rule':'source/caption disjoint from reserved V2 and fresh test; deterministic source/UID hash order; max 8 clips per source','minimum_fresh_coverage':FRESH_MIN_COVERAGE,'fixed_unseen_set':{'selection_rule':'sha256_rank_by_uid','uids':fixed_uids},'selected_v2':[selection_record(record) for record in selected_v2.to_dict(orient='records')],'fresh_train':[selection_record(record) for record in fresh_train_rows],'fresh_test':[selection_record(record) for record in fresh_test_rows]},indent=2))
print('Persisted fixed selection roster before decoding:',PREDECODE_SELECTION_PATH,'| rows:',V2_LIMIT+FRESH_TRAIN_TARGET+FRESH_TEST_TARGET)

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

working_cache = Path('/kaggle/working/isign_pose_cache_v5'); working_cache.mkdir(parents=True, exist_ok=True)
original_train_records = []
original_test_records = []
load_exclusions = []
rows_loaded, pose_paths = [], []
for record in selected_v2.to_dict(orient='records'):
    enriched = dict(record); enriched['split_source'] = 'v2_reservoir'; enriched['source_uid_rule'] = str(record['source_uid_rule'])
    original_train_records.append(enriched)
for record in fresh_train_rows: original_train_records.append({**record, 'split_source': 'fresh_train'})
for record in fresh_test_rows: original_test_records.append({**record, 'split_source': 'fresh_test'})

def exclusion_record(record, side, reason):
    return {'uid': str(record['uid']), 'source_video_id': str(record['source_video_id']), 'source_aliases': sorted(conservative_source_aliases(record['uid'], record.get('source_video_id'))), 'text': str(record['text']), 'original_split': side,
            'anchor_mode': 'unavailable_before_canonicalization', 'threshold': 0.15,
            'finite_anchor_counts': {'nose': 0, 'left_shoulder': 0, 'right_shoulder': 0},
            'confident_anchor_counts': {'nose': 0, 'left_shoulder': 0, 'right_shoulder': 0},
            'primary_scale_valid_count': 0, 'direct_anchor_frames': 0, 'fallback_anchor_frames': 0,
            'interpolated_anchor_frames': 0, 'unresolved_anchor_frames': TMAX, 'total_anchor_frames': 0,
            'resampled_frame_count': TMAX, 'reason': reason}

for record in selected_v2.to_dict(orient='records'):
    if record['uid'] not in cached_by_uid:
        load_exclusions.append(exclusion_record(record, 'train', 'selected_v2_pose_missing_from_verified_cache'))
        continue
    path = cached_by_uid[record['uid']]
    try:
        with np.load(path) as saved:
            if np.asarray(saved['pose']).shape != (TMAX, 1728) or np.asarray(saved['confidence']).shape != (TMAX, 576): raise ValueError('cached_shape')
            raw_count = int(saved['raw_frame_count']) if 'raw_frame_count' in saved else None
    except Exception as exc:
        load_exclusions.append(exclusion_record(record, 'train', 'selected_v2_cache_error:' + type(exc).__name__))
        continue
    rows_loaded.append({**record, 'source_aliases': sorted(conservative_source_aliases(record['uid'], record['source_video_id'])), 'raw_frame_count': raw_count, 'resampling_policy': 'uniform_linear_48_endpoints_included; original_count_unavailable_in_v2_cache', 'split_source': 'v2_reservoir'})
    pose_paths.append(path)

fresh_decode_rows = [
    *[{**record, 'split_source': 'fresh_train'} for record in fresh_train_rows],
    *[{**record, 'split_source': 'fresh_test'} for record in fresh_test_rows],
]
decode_session = requests.Session()
decode_errors = Counter(); range_decodes = 0; reused_v5_cache = 0; transient_retry_attempts = 0
def decode_member_with_retry(zf, uid, max_attempts=3):
    global transient_retry_attempts
    for attempt in range(1,max_attempts+1):
        try: return decode_member(zf,uid)
        except Exception as exc:
            message=str(exc)
            transient=isinstance(exc,(requests.RequestException,TimeoutError,ConnectionError)) or any(f'HTTP {status}' in message for status in (429,500,502,503,504))
            if not transient or attempt==max_attempts: raise
            transient_retry_attempts+=1
            wait_seconds=min(2**(attempt-1),4)
            print(f'Transient pose-range failure for {uid}; retry {attempt+1}/{max_attempts} in {wait_seconds}s: {type(exc).__name__}',flush=True)
            time.sleep(wait_seconds)
with zipfile.ZipFile(archive, 'r') as zf:
    fresh_decode_rows.sort(key=lambda row: (zf.getinfo(pose_by_file[row['uid'] + '.pose']).header_offset, row['uid']))
    for n, record in enumerate(fresh_decode_rows, start=1):
        uid = str(record['uid']); side = 'train' if record['split_source'] == 'fresh_train' else 'validation'
        cache_path = working_cache / (uid + '.npz')
        try:
            if cache_path.exists():
                with np.load(cache_path) as saved:
                    flat = np.asarray(saved['pose'], dtype=np.float32).copy(); conf = np.asarray(saved['confidence'], dtype=np.float32).copy()
                    raw_count = int(saved['raw_frame_count'])
                if flat.shape != (TMAX, 1728) or conf.shape != (TMAX, 576): raise ValueError('working_cache_shape')
                reused_v5_cache += 1
            else:
                flat, conf, raw_count = decode_member_with_retry(zf, uid)
                tmp = cache_path.with_suffix('.npz.tmp')
                with open(tmp, 'wb') as handle: np.savez_compressed(handle, pose=flat, confidence=conf, raw_frame_count=np.asarray(raw_count))
                tmp.replace(cache_path); range_decodes += 1
            rows_loaded.append({**record, 'raw_frame_count': raw_count, 'resampling_policy': 'uniform_linear_48_endpoints_included', 'split_source': 'fresh_train' if side == 'train' else 'fresh_test'})
            pose_paths.append(cache_path)
        except Exception as exc:
            decode_errors[type(exc).__name__ + ':' + str(exc)[:80]] += 1
            load_exclusions.append(exclusion_record(record, side, 'fresh_pose_decode_error:' + type(exc).__name__))
        if n % 256 == 0 or n == len(fresh_decode_rows):
            print(f'Decoded {n}/{len(fresh_decode_rows)} fresh clips; range decodes={range_decodes}, working-cache hits={reused_v5_cache}, exclusions={len(load_exclusions)}', flush=True)

rows = pd.DataFrame(rows_loaded).reset_index(drop=True)
rows['source_video_id'] = rows['source_video_id'].astype(str)
assert rows.uid.is_unique and len(rows) == len(pose_paths)
loaded_fresh_train_count = int(np.sum(rows['split_source'] == 'fresh_train'))
loaded_fresh_test_count = int(np.sum(rows['split_source'] == 'fresh_test'))
min_fresh_train = int(math.ceil(FRESH_MIN_COVERAGE * FRESH_TRAIN_TARGET))
min_fresh_test = int(math.ceil(FRESH_MIN_COVERAGE * FRESH_TEST_TARGET))
assert loaded_fresh_train_count >= min_fresh_train, f'Fresh training decode coverage {loaded_fresh_train_count}/{FRESH_TRAIN_TARGET} is below {FRESH_MIN_COVERAGE:.0%}; selection manifest and failure list are preserved; refusing to continue.'
assert loaded_fresh_test_count >= min_fresh_test, f'Fresh test decode coverage {loaded_fresh_test_count}/{FRESH_TEST_TARGET} is below {FRESH_MIN_COVERAGE:.0%}; selection manifest and failure list are preserved; refusing to continue.'
loaded_test_sources = set(rows.loc[rows['split_source'] == 'fresh_test', 'source_video_id'])
loaded_train_sources = set(rows.loc[rows['split_source'] != 'fresh_test', 'source_video_id'])
assert loaded_train_sources.isdisjoint(loaded_test_sources), 'Source-video leakage after loading.'
loaded_train_aliases = set().union(*(set(value) for value in rows.loc[rows['split_source'] != 'fresh_test', 'source_aliases']))
loaded_test_aliases = set().union(*(set(value) for value in rows.loc[rows['split_source'] == 'fresh_test', 'source_aliases']))
assert loaded_train_aliases.isdisjoint(loaded_test_aliases), 'Conservative source-alias leakage after loading.'
print('Loaded effective rows:', len(rows), '| training reservoir:', int((rows.split_source != 'fresh_test').sum()), '| fresh test:', int((rows.split_source == 'fresh_test').sum()))
print('V5 pose loads:', range_decodes, 'fresh archive decodes,', reused_v5_cache, 'working-cache hits; transient retries:',transient_retry_attempts,'; failures:', dict(decode_errors))
print('V2 cached data reused:', len(cached_by_uid), '| fresh selected original:', len(fresh_train_rows), 'train /', len(fresh_test_rows), 'test')

# Load the frozen V4 MiniLM model for a paired baseline on the same fresh test examples.
v4_checkpoint_candidates = list(Path('/kaggle/input').rglob('minilm_position_motion.pt'))
assert v4_checkpoint_candidates, 'Attach the private V4 checkpoint artifact dataset containing minilm_position_motion.pt.'
V4_CHECKPOINT_PATH = v4_checkpoint_candidates[0]
V4_CHECKPOINT_SHA256 = hashlib.sha256(V4_CHECKPOINT_PATH.read_bytes()).hexdigest()
v4_checkpoint = torch.load(V4_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
assert v4_checkpoint.get('arm') == 'minilm_position_motion' and v4_checkpoint.get('frozen_text_encoder') == 'sentence-transformers/all-MiniLM-L6-v2'
assert np.asarray(v4_checkpoint['normalizer_mean']).shape == (75, 3) and np.asarray(v4_checkpoint['normalizer_std']).shape == (75, 3)
assert len(v4_checkpoint['point_indices']) == 75 and v4_checkpoint['time_steps'] == TMAX
print('Frozen V4 checkpoint SHA256:', V4_CHECKPOINT_SHA256, '| architecture:', v4_checkpoint['format'])
'''


V5_CANONICAL = r'''# Canonicalize all loaded rows, hold out fresh test first, then source-group inner stop.
POINT_INDICES = np.asarray(list(range(33)) + list(range(501, 543)), dtype=np.int64)
TMAX, POINTS = 48, 75
MIN_ANCHOR_CONF = 0.15
assert np.array_equal(np.asarray(v4_checkpoint['point_indices']), POINT_INDICES), 'V4 and V5 target points differ.'

def canonicalize_one(raw_pose, raw_confidence, min_anchor_conf=MIN_ANCHOR_CONF):
    raw_pose = np.asarray(raw_pose, dtype=np.float32).reshape(TMAX, 576, 3)
    raw_confidence = np.asarray(raw_confidence, dtype=np.float32).reshape(TMAX, 576)
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
    anchor_names = {'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12}
    info = {
        'anchor_mode': 'nose_both_shoulders', 'threshold': float(min_anchor_conf),
        'finite_anchor_counts': {name: int(valid_coords[:, point].sum()) for name, point in anchor_names.items()},
        'confident_anchor_counts': {name: int((valid_coords[:, point] & (point_conf[:, point] >= min_anchor_conf)).sum()) for name, point in anchor_names.items()},
        'primary_scale_valid_count': int(len(good)), 'direct_anchor_frames': int(len(good)), 'fallback_anchor_frames': 0,
        'interpolated_anchor_frames': 0, 'unresolved_anchor_frames': TMAX, 'total_anchor_frames': int(len(good)),
        'resampled_frame_count': TMAX,
    }
    if len(good) < 2:
        info['anchor_mode'] = 'unusable'
        info['reason'] = 'fewer_than_two_reliable_nose_and_both_shoulder_frames'
        return None, None, info
    frame_ix = np.arange(TMAX)
    center_interp = np.stack([np.interp(frame_ix, good, center[good, axis]) for axis in range(3)], axis=-1)
    scale_interp = np.exp(np.interp(frame_ix, good, np.log(scale[good].clip(1e-4))))
    reliability = np.where(anchors_ok, 1.0, 0.5).astype(np.float32)
    canonical = (selected_pose - center_interp[:, None, :]) / scale_interp[:, None, None]
    selected_conf *= reliability[:, None]
    canonical = np.nan_to_num(canonical, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    info['interpolated_anchor_frames'] = int(TMAX - len(good)); info['unresolved_anchor_frames'] = 0
    return canonical, selected_conf.astype(np.float32), info

original_rows = rows.copy().reset_index(drop=True)
original_rows['source_aliases'] = [
    sorted(set(value) if isinstance(value, (list, tuple, set)) else conservative_source_aliases(row.uid, row.source_video_id))
    for row, value in zip(original_rows.itertuples(), original_rows['source_aliases'])
]
original_train_records = [
    {key: value for key, value in record.items() if key not in {'text_key', 'test_rank', 'legacy_video_id', 'parsed_source_video_id', 'source_uid_supported', 'source_uid_mismatch'}}
    for record in original_train_records
]
original_test_records = [
    {key: value for key, value in record.items() if key not in {'text_key', 'test_rank', 'legacy_video_id', 'parsed_source_video_id', 'source_uid_supported', 'source_uid_mismatch'}}
    for record in original_test_records
]
old_source_ids = set(selected_v2['source_video_id'])
old_text_keys = set(selected_v2['text_key'])
canonical_poses, canonical_confs, kept_positions = [], [], []
canonicalization_by_uid, canonicalization_exclusions = {}, list(load_exclusions)
for position, (record, path) in enumerate(zip(original_rows.to_dict(orient='records'), pose_paths)):
    with np.load(path) as saved:
        pose = np.asarray(saved['pose'], dtype=np.float32).copy()
        confidence = np.asarray(saved['confidence'], dtype=np.float32).copy()
    canonical, canonical_conf, info = canonicalize_one(pose, confidence)
    uid = str(record['uid'])
    if canonical is None:
        side = 'validation' if record['split_source'] == 'fresh_test' else 'train'
        canonicalization_exclusions.append({'uid':uid,'source_video_id':str(record['source_video_id']),'source_aliases':list(record['source_aliases']),'text':str(record['text']),'original_split':side,**info})
        canonicalization_by_uid[uid] = info
        continue
    kept_positions.append(position); canonical_poses.append(canonical); canonical_confs.append(canonical_conf); canonicalization_by_uid[uid] = info

kept_positions = np.asarray(kept_positions, dtype=np.int64)
rows = original_rows.iloc[kept_positions].reset_index(drop=True)
canonical_poses = np.stack(canonical_poses).astype(np.float32)
canonical_confs = np.stack(canonical_confs).astype(np.float32)
position_to_new = {int(old): new for new, old in enumerate(kept_positions)}
training_indices_original = [i for i, record in enumerate(original_rows.to_dict(orient='records')) if record['split_source'] != 'fresh_test']
test_indices_original = [i for i, record in enumerate(original_rows.to_dict(orient='records')) if record['split_source'] == 'fresh_test']
training_pool_ix = np.asarray([position_to_new[i] for i in training_indices_original if i in position_to_new], dtype=np.int64)
test_ix = np.asarray([position_to_new[i] for i in test_indices_original if i in position_to_new], dtype=np.int64)
assert set(rows.iloc[training_pool_ix].source_video_id).isdisjoint(set(rows.iloc[test_ix].source_video_id)), 'Canonicalization changed the source-disjoint split.'
assert len(training_pool_ix) + len(test_ix) + len(canonicalization_exclusions) == len(original_train_records) + len(original_test_records)

# Internal early-stop groups are stable source-video groups within the V5 training pool.
alias_parent = {int(i): int(i) for i in training_pool_ix}
def alias_find(item):
    while alias_parent[item] != item:
        alias_parent[item] = alias_parent[alias_parent[item]]
        item = alias_parent[item]
    return item
def alias_union(left, right):
    left, right = alias_find(left), alias_find(right)
    if left != right: alias_parent[max(left, right)] = min(left, right)
alias_owner = {}
for i in training_pool_ix:
    for alias in rows.iloc[int(i)].source_aliases:
        if alias in alias_owner: alias_union(int(i), alias_owner[alias])
        else: alias_owner[alias] = int(i)
alias_members = defaultdict(list)
for i in training_pool_ix: alias_members[alias_find(int(i))].append(int(i))
cluster_by_index = {}
for members in alias_members.values():
    member_ids = sorted(str(rows.iloc[i].uid) for i in members)
    cluster = 'conservative_uid_alias_cluster:' + hashlib.sha256('\n'.join(member_ids).encode()).hexdigest()[:16]
    for i in members: cluster_by_index[i] = cluster
rows['source_video_cluster'] = [cluster_by_index.get(i, 'test:' + str(rows.iloc[i].uid)) for i in range(len(rows))]
train_groups = sorted(set(rows.iloc[training_pool_ix].source_video_cluster.astype(str)))
early_group_count = max(1, int(round(len(train_groups) * 0.10)))
early_groups = set(sorted(train_groups, key=lambda source: (hashlib.sha256(f'{SEED}:inner-source:{source}'.encode()).hexdigest(), source))[:early_group_count])
early_stop_ix = [int(i) for i in training_pool_ix if str(rows.iloc[int(i)].source_video_cluster) in early_groups]
early_keys = {text_key(rows.iloc[i].text) for i in early_stop_ix}
test_keys = {text_key(row['text']) for row in original_test_records}
fit_ix = [int(i) for i in training_pool_ix if str(rows.iloc[int(i)].source_video_cluster) not in early_groups and text_key(rows.iloc[int(i)].text) not in early_keys and text_key(rows.iloc[int(i)].text) not in test_keys]
fit_ix = np.asarray(fit_ix, dtype=np.int64); early_stop_ix = np.asarray(early_stop_ix, dtype=np.int64); training_pool_ix = np.asarray(training_pool_ix, dtype=np.int64); test_ix = np.asarray(test_ix, dtype=np.int64)
assert len(fit_ix) and len(early_stop_ix) and len(test_ix) >= int(math.ceil(FRESH_MIN_COVERAGE*FRESH_TEST_TARGET)), f'Need at least {FRESH_MIN_COVERAGE:.0%} canonicalized fresh test coverage; got {len(test_ix)}/{FRESH_TEST_TARGET}.'
assert int(np.sum(rows.iloc[training_pool_ix].split_source=='fresh_train')) >= int(math.ceil(FRESH_MIN_COVERAGE*FRESH_TRAIN_TARGET)), f'Fresh training canonicalization coverage is below {FRESH_MIN_COVERAGE:.0%}.'
assert set(rows.iloc[fit_ix].uid).isdisjoint(set(rows.iloc[early_stop_ix].uid))
assert set(rows.iloc[fit_ix].source_video_cluster).isdisjoint(set(rows.iloc[early_stop_ix].source_video_cluster))
fit_aliases = set().union(*(set(value) for value in rows.iloc[fit_ix].source_aliases))
early_aliases = set().union(*(set(value) for value in rows.iloc[early_stop_ix].source_aliases))
assert fit_aliases.isdisjoint(early_aliases), 'Conservative UID alias overlap across fit and internal stop.'
assert {text_key(x) for x in rows.iloc[fit_ix].text}.isdisjoint(early_keys | test_keys)
assert set(rows.iloc[training_pool_ix].source_video_id).isdisjoint(set(rows.iloc[test_ix].source_video_id))
assert set(rows.iloc[training_pool_ix].uid).isdisjoint(set(rows.iloc[test_ix].uid))
assert {text_key(x) for x in rows.iloc[training_pool_ix].text}.isdisjoint(test_keys)

source_groups = {'body': (0, 33, 0.20), 'left_hand': (33, 54, 0.40), 'right_hand': (54, 75, 0.40)}
GROUPS = source_groups
print(f'Canonicalized train={len(training_pool_ix)} (fit={len(fit_ix)}, early-stop={len(early_stop_ix)}) / fresh test={len(test_ix)}; groups={len(train_groups)} train groups / {len(set(rows.iloc[test_ix].source_video_id))} test groups.')
print('Excluded rows by original split/reason:', dict(Counter((row['original_split'],row['reason']) for row in canonicalization_exclusions)))
print('Canonical target: 75 points, body 0:33, LH 33:54, RH 54:75; 2D nose-to-shoulder scale applied to xyz.')
'''


V5_TRAIN = r'''# Train one frozen-MiniLM arm using the predeclared position + velocity objective.
training_ix = np.asarray(fit_ix, dtype=np.int64); early_ix = np.asarray(early_stop_ix, dtype=np.int64)
torch.manual_seed(SEED + 303); torch.cuda.manual_seed_all(SEED + 303)
core = TextToPose('minilm', vocab_size=len(vocab)).to(device)
model = nn.DataParallel(core, device_ids=[0, 1])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
train_loader = make_loader(training_ix, 'minilm', True); stop_loader = make_loader(early_ix, 'minilm', False)

devices_seen = set()
hook = core.register_forward_pre_hook(lambda module, inputs: devices_seen.add(inputs[0].device.index))
probe = next(iter(train_loader)); model.eval()
probe_x, probe_m, probe_y, probe_c = [item.to(device) for item in probe]
model.train(); optimizer.zero_grad(set_to_none=True)
probe_prediction = model(probe_x, probe_m)
assert probe_prediction.shape == probe_y.shape == (probe_x.shape[0], TMAX, POINTS * 3)
assert torch.isfinite(probe_prediction).all()
probe_pos, probe_vel, _ = grouped_terms(probe_prediction, probe_y, probe_c)
probe_loss = probe_pos + VELOCITY_WEIGHT * probe_vel
assert torch.isfinite(probe_loss)
probe_loss.backward()
probe_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
assert torch.isfinite(probe_grad_norm) and float(probe_grad_norm) > 0
optimizer.zero_grad(set_to_none=True); model.eval()
assert {0, 1}.issubset(devices_seen), f'Two-GPU probe failed: {sorted(devices_seen)}'
print('Two-GPU minibatch forward/backward passed:', tuple(probe_prediction.shape), 'grad_norm=', round(float(probe_grad_norm), 5), flush=True)

baseline_internal = evaluate_loader(model, stop_loader, VELOCITY_WEIGHT)
best_score, best_state, best_epoch, stale = float('inf'), None, 0, 0
history=[]; started=time.time()
for epoch in range(1, MAX_EPOCHS + 1):
    model.train(); loss_rows=[]
    for text_input, text_mask, target, confidence in train_loader:
        text_input=text_input.to(device,non_blocking=True); text_mask=text_mask.to(device,non_blocking=True)
        target=target.to(device,non_blocking=True); confidence=confidence.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        position, velocity, _ = grouped_terms(model(text_input,text_mask),target,confidence)
        loss=position+VELOCITY_WEIGHT*velocity
        if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss.')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        loss_rows.append((float(position.detach()),float(velocity.detach())))
    validation=evaluate_loader(model,stop_loader,VELOCITY_WEIGHT)
    row={'epoch':epoch,'train_position':float(np.mean([x[0] for x in loss_rows])),'train_velocity':float(np.mean([x[1] for x in loss_rows])),**{'early_'+key:value for key,value in validation.items()}}
    history.append(row)
    if validation['selection_score'] < best_score:
        best_score=validation['selection_score']; best_epoch=epoch; stale=0
        best_state={key:value.detach().cpu().clone() for key,value in core.state_dict().items()}
        torch.save({'epoch':best_epoch,'state_dict':best_state,'best_score':best_score,'history':history,'fit_uids':[str(rows.iloc[i].uid) for i in fit_ix],'early_stop_uids':[str(rows.iloc[i].uid) for i in early_stop_ix]},'/kaggle/working/isl_v5_resume_best.pt')
    else: stale+=1
    print(f'V5 epoch {epoch:02d}/{MAX_EPOCHS}: fit_pos={row["train_position"]:.5f} stop_pos={validation["position"]:.5f} stop_vel={validation["velocity"]:.5f} score={validation["selection_score"]:.5f} motion={validation["motion_ratio_by_group"]}',flush=True)
    if stale>=PATIENCE: break
hook.remove()
if best_state is None: raise RuntimeError('V5 produced no best checkpoint.')
core.load_state_dict(best_state); model.eval()
assert {0,1}.issubset(devices_seen), f'Training did not use both GPUs: {sorted(devices_seen)}'
run={'config':{'name':'minilm_position_motion_v5','mode':'minilm','velocity_weight':VELOCITY_WEIGHT,'seed':SEED+303},'best_state':best_state,'best_epoch':best_epoch,'best_internal_selection_score':best_score,'best_internal_position':history[best_epoch-1]['early_position'],'best_internal_velocity':history[best_epoch-1]['early_velocity'],'baseline_internal':baseline_internal,'history':history,'gpus_used':sorted(devices_seen),'seconds':round(time.time()-started,1)}
print('Selected V5 epoch:',best_epoch,'| fit rows:',len(fit_ix),'| early-stop rows:',len(early_stop_ix),'| time:',run['seconds'],'s')
'''


V5_EXPORT = r'''# Export the fixed fresh test, true/shuffled predictions, frozen V4 baseline, and full audit manifest.
EXPORT_ROOT=Path('/kaggle/working/isl_v5_export')
if EXPORT_ROOT.exists(): shutil.rmtree(EXPORT_ROOT)
for directory in ('references','baselines','predictions','checkpoints','diagnostics'): (EXPORT_ROOT/directory).mkdir(parents=True,exist_ok=True)
shutil.copy2(PREDECODE_SELECTION_PATH,EXPORT_ROOT/'selection_manifest_predecode.json')

test_unique={}
for i in test_ix: test_unique.setdefault(text_key(rows.iloc[int(i)].text),int(i))
test_keys_sorted=sorted(test_unique)
assert len(test_keys_sorted)>=2
counter_ix=[]
for i in test_ix:
    uid=str(rows.iloc[int(i)].uid); own=text_key(rows.iloc[int(i)].text)
    choices=[key for key in test_keys_sorted if key!=own]
    rank=int(hashlib.sha256(f'{SEED}:{uid}'.encode()).hexdigest(),16)%len(choices)
    counter_ix.append(test_unique[choices[rank]])
counter_ix=np.asarray(counter_ix,dtype=np.int64)
assert all(text_key(rows.iloc[int(i)].text)!=text_key(rows.iloc[int(j)].text) for i,j in zip(test_ix,counter_ix))
fit_text_keys={text_key(rows.iloc[int(i)].text) for i in fit_ix}
assert all(text_key(rows.iloc[int(j)].text) not in fit_text_keys for j in counter_ix)

slug_by_uid={str(rows.iloc[int(i)].uid):re.sub(r'[^A-Za-z0-9._-]+','_',str(rows.iloc[int(i)].uid)) for i in test_ix}
for i in test_ix:
    uid=str(rows.iloc[int(i)].uid)
    np.savez_compressed(EXPORT_ROOT/'references'/(slug_by_uid[uid]+'.npz'),pose=canonical_poses[int(i)],confidence=canonical_confs[int(i)])

# Fit train-only per-point statistics and baseline frame mean on gradient-fit rows.
fit_pose=canonical_poses[fit_ix]; fit_conf=canonical_confs[fit_ix]; weights=fit_conf[...,None]
den=weights.sum(axis=(0,1)); normalizer_mean=((fit_pose*weights).sum(axis=(0,1))/np.maximum(den,1e-6)).astype(np.float32)
variance=(((fit_pose-normalizer_mean[None,None])**2)*weights).sum(axis=(0,1))/np.maximum(den,1e-6)
normalizer_std=np.sqrt(np.maximum(variance,0)).clip(0.05).astype(np.float32)
targets=np.nan_to_num((canonical_poses-normalizer_mean[None,None])/normalizer_std[None,None],nan=0,posinf=0,neginf=0).astype(np.float32)
training_uids_used=[str(rows.iloc[int(i)].uid) for i in fit_ix]
normalizer_fit_uid_sha256=hashlib.sha256('\n'.join(sorted(training_uids_used)).encode()).hexdigest()
frame_den=fit_conf[...,None].sum(axis=0)
train_frame_mean=np.nan_to_num((fit_pose*fit_conf[...,None]).sum(axis=0)/np.maximum(frame_den,1e-6),nan=0).astype(np.float32)
np.savez_compressed(EXPORT_ROOT/'normalizer.npz',mean=normalizer_mean,std=normalizer_std)
np.save(EXPORT_ROOT/'baselines'/'zero_pose.npy',np.zeros((TMAX,POINTS,3),dtype=np.float32))
np.save(EXPORT_ROOT/'baselines'/'train_frame_mean.npy',train_frame_mean)
np.save(EXPORT_ROOT/'point_indices.npy',POINT_INDICES)

header_path=next(iter(Path('/kaggle/input').rglob('onNSvHmicw0_header.json')),None)
header=json.load(open(header_path,encoding='utf-8')) if header_path else None
if header is not None:
    edges=[]
    for component in header.get('components',[]):
        name=component.get('name','')
        if name=='POSE_LANDMARKS': dst,source,count=0,0,33
        elif name in {'LEFT_HAND','LEFT_HAND_LANDMARKS'}: dst,source,count=33,501,21
        elif name in {'RIGHT_HAND','RIGHT_HAND_LANDMARKS'}: dst,source,count=54,522,21
        else: continue
        for a,b in component.get('limbs',[]):
            if 0<=int(a)<count and 0<=int(b)<count: edges.append((dst+int(a),dst+int(b)))
    edges=np.asarray(sorted(set(tuple(sorted(edge)) for edge in edges)),dtype=np.int64)
else:
    body_edges=[(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),(11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(27,31),(24,26),(26,28),(28,30),(30,32),(28,32),(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10)]
    hand_edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
    edges=np.asarray(body_edges+[(33+a,33+b) for a,b in hand_edges]+[(54+a,54+b) for a,b in hand_edges],dtype=np.int64)
np.save(EXPORT_ROOT/'skeleton_edges.npy',edges)

def predict(core_model,indices):
    out=[]
    for start in range(0,len(indices),BATCH):
        batch=indices[start:start+BATCH]
        x=torch.from_numpy(mini_features[batch]).to(device); mask=torch.ones((len(batch),1),dtype=torch.float32,device=device)
        with torch.no_grad(): out.append(core_model(x,mask).cpu().numpy().reshape(len(batch),TMAX,POINTS,3))
    z=np.concatenate(out,axis=0)
    return (z*normalizer_std[None,None]+normalizer_mean[None,None]).astype(np.float32)

def calc_metrics(prediction,reference,confidence):
    zpred=(prediction-normalizer_mean[None,None])/normalizer_std[None,None]
    zref=(reference-normalizer_mean[None,None])/normalizer_std[None,None]
    out={}
    for name,(start,end,weight) in GROUPS.items():
        conf=confidence[:,:,start:end,None]; err=(zpred[:,:,start:end]-zref[:,:,start:end])**2
        raw=(prediction[:,:,start:end]-reference[:,:,start:end])**2; den=max(float(conf.sum()*3),1e-8)
        pair=np.minimum(confidence[:,1:,start:end],confidence[:,:-1,start:end])
        pred_speed=np.linalg.norm(np.diff(prediction[:,:,start:end],axis=1),axis=-1); ref_speed=np.linalg.norm(np.diff(reference[:,:,start:end],axis=1),axis=-1)
        md=max(float(pair.sum()),1e-8); pm=float((pred_speed*pair).sum()/md); rm=float((ref_speed*pair).sum()/md)
        out[name]={'normalized_mse':float((err*conf).sum()/den),'canonical_rmse':float(np.sqrt((raw*conf).sum()/den)),'prediction_motion':pm,'reference_motion':rm,'motion_ratio':float(pm/max(rm,1e-8))}
    out['group_balanced_normalized_mse']=float(sum(GROUPS[name][2]*out[name]['normalized_mse'] for name in GROUPS))
    return out

reference_all=canonical_poses[test_ix]; confidence_all=canonical_confs[test_ix]
mini_similarity=mini_features[test_ix] @ mini_features[fit_ix].T
retrieval_ix=fit_ix[mini_similarity.argmax(axis=1)]
retrieval_all=canonical_poses[retrieval_ix]
v4_core=TextToPose('minilm',vocab_size=len(vocab)).to(device)
v4_core.load_state_dict(v4_checkpoint['state_dict']); v4_core.eval()
v4_mean=np.asarray(v4_checkpoint['normalizer_mean'],dtype=np.float32); v4_std=np.asarray(v4_checkpoint['normalizer_std'],dtype=np.float32)
def predict_v4(indices):
    out=[]
    for start in range(0,len(indices),BATCH):
        batch=indices[start:start+BATCH]; x=torch.from_numpy(mini_features[batch]).to(device); mask=torch.ones((len(batch),1),device=device)
        with torch.no_grad(): out.append(v4_core(x,mask).cpu().numpy().reshape(len(batch),TMAX,POINTS,3))
    z=np.concatenate(out,axis=0)
    return (z*v4_std[None,None]+v4_mean[None,None]).astype(np.float32)

true_v5=predict(core,test_ix); shuffle_v5=predict(core,counter_ix)
true_v4=predict_v4(test_ix); shuffle_v4=predict_v4(counter_ix)
zero=np.zeros_like(reference_all); mean_pose=np.broadcast_to(train_frame_mean,reference_all.shape).copy()
np.save(EXPORT_ROOT/'baselines'/'zero_pose.npy',zero[0])

v5_dir=EXPORT_ROOT/'predictions'/'v5_minilm'; v4_dir=EXPORT_ROOT/'predictions'/'v4_frozen'; v5_dir.mkdir(); v4_dir.mkdir()
sample_records=[]
for j,i in enumerate(test_ix):
    uid=str(rows.iloc[int(i)].uid); slug=slug_by_uid[uid]; donor_idx=int(counter_ix[j]); donor_uid=str(rows.iloc[donor_idx].uid)
    paths={
        'true_text':v5_dir/(slug+'_true.npy'),'shuffled_text':v5_dir/(slug+'_shuffled.npy'),
        'v4_model_true':v4_dir/(slug+'_true.npy'),'v4_model_shuffled':v4_dir/(slug+'_shuffled.npy'),
        'zero_pose':EXPORT_ROOT/'baselines'/(slug+'_zero.npy'),'train_frame_mean':EXPORT_ROOT/'baselines'/(slug+'_mean.npy'),
        'semantic_retrieval':EXPORT_ROOT/'baselines'/(slug+'_retrieval.npy')}
    np.save(paths['true_text'],true_v5[j]); np.save(paths['shuffled_text'],shuffle_v5[j])
    np.save(paths['v4_model_true'],true_v4[j]); np.save(paths['v4_model_shuffled'],shuffle_v4[j])
    np.save(paths['zero_pose'],zero[j]); np.save(paths['train_frame_mean'],mean_pose[j]); np.save(paths['semantic_retrieval'],retrieval_all[j])
    rec=rows.iloc[int(i)]; raw=rec.raw_frame_count
    sample_records.append({'uid':uid,'source_video_id':str(rec.source_video_id),'source_aliases':sorted(set(rec.source_aliases)),'source_uid_rule':str(rec.source_uid_rule),'source_uid_supported':bool(rec.source_uid_supported),'source_video_id_provenance':str(rec.source_video_id_provenance),'source_video_cluster':str(rec.source_video_cluster),'normalized_caption_key':text_key(rec.text),'text':str(rec.text),'counterfactual_text':str(rows.iloc[donor_idx].text),'counterfactual_source_uid':donor_uid,'raw_frame_count':None if pd.isna(raw) else int(raw),'resampled_frame_count':TMAX,'resampling_policy':str(rec.resampling_policy),'canonicalization_anchor_audit':canonicalization_by_uid[uid],'semantic_retrieval_source_uid':str(rows.iloc[int(retrieval_ix[j])].uid),'reference':{'path':'references/'+slug+'.npz','pose_key':'pose','confidence_key':'confidence'},'predictions':{key:str(path.relative_to(EXPORT_ROOT)).replace('\\','/') for key,path in paths.items()}})

effective_test_uid_set={str(rows.iloc[int(i)].uid) for i in test_ix}
fixed_present_uids=[uid for uid in fixed_uids if uid in effective_test_uid_set]
fixed_missing_uids=[uid for uid in fixed_uids if uid not in effective_test_uid_set]

metrics={
    'dataset':'Exploration-Lab/iSign v1.1','v2_run_key':run_key,'training_rows':len(training_pool_ix),'effective_fit_rows':len(fit_ix),'early_stop_rows':len(early_stop_ix),'fresh_test_rows_selected':len(fresh_test_rows),'fresh_test_rows_effective':len(test_ix),'fresh_train_rows_selected':len(fresh_train_rows),'fresh_train_rows_effective':int(np.sum(rows.split_source=='fresh_train')),'fresh_coverage_threshold':FRESH_MIN_COVERAGE,'fixed_qualitative_uids_selected_before_decode':fixed_uids,'fixed_qualitative_uids_effective':fixed_present_uids,'fixed_qualitative_uids_missing':fixed_missing_uids,
    'v2_cache_rows_selected':V2_LIMIT,'v2_cache_rows_loaded':int(np.sum(rows.split_source=='v2_reservoir')),'load_and_canonicalization_exclusions':canonicalization_exclusions,
    'v5_training':{'selected_arm':'minilm_position_motion_v5','best_epoch':best_epoch,'selection_score':best_score,'best_internal_position':run['best_internal_position'],'best_internal_velocity':run['best_internal_velocity'],'gpus_used':run['gpus_used'],'seconds':run['seconds'],'history':history,'selection_rule':'minimum source-grouped inner validation group-balanced position MSE + 0.05 * group-balanced velocity MSE; fresh test never used for selection'},
    'v4_frozen_checkpoint_sha256':V4_CHECKPOINT_SHA256,'v4_frozen_checkpoint_source':'attached private V4 artifacts dataset','text_encoder':'sentence-transformers/all-MiniLM-L6-v2','objective':{'group_weights':{'body':0.20,'left_hand':0.40,'right_hand':0.40},'velocity_coefficient':VELOCITY_WEIGHT},
    'true_text':calc_metrics(true_v5,reference_all,confidence_all),'shuffled_text':calc_metrics(shuffle_v5,reference_all,confidence_all),'v4_model_true':calc_metrics(true_v4,reference_all,confidence_all),'v4_model_shuffled':calc_metrics(shuffle_v4,reference_all,confidence_all),
    'baselines':{'zero_pose':calc_metrics(zero,reference_all,confidence_all),'train_frame_mean':calc_metrics(mean_pose,reference_all,confidence_all),'semantic_nearest_training_clip':calc_metrics(retrieval_all,reference_all,confidence_all)}}
(EXPORT_ROOT/'metrics_summary.json').write_text(json.dumps(metrics,indent=2))

source_rule_counts=metadata.groupby('source_uid_rule').size().to_dict()
source_id_provenance=str(metadata['source_video_id_provenance'].iloc[0])
def simple_record(row):
    uid=str(row['uid']); text=str(row['text']); aliases=row.get('source_aliases')
    if not isinstance(aliases,(list,tuple,set,np.ndarray)):
        aliases=conservative_source_aliases(uid,row.get('source_video_id'))
    result={'uid':uid,'source_video_id':str(row['source_video_id']),'source_aliases':sorted({str(value) for value in aliases}),'source_uid_rule':str(row.get('source_uid_rule','unknown')),'source_uid_supported':bool(row.get('source_uid_supported',False)),'source_uid_mismatch':bool(row.get('source_uid_mismatch',False)),'source_video_id_provenance':str(row.get('source_video_id_provenance','unknown')),'normalized_caption_key':text_key(text),'text':text}
    for key in ('split_source','source_video_cluster'):
        value=row.get(key)
        if value is not None and not pd.isna(value): result[key]=str(value)
    return result
old_train_records=[simple_record(record) for record in original_train_records]
old_test_records=[simple_record(record) for record in original_test_records]
effective_train_records=[simple_record(rows.iloc[int(i)]) for i in training_pool_ix]
effective_test_records=[simple_record(rows.iloc[int(i)]) for i in test_ix]
exclusions=canonicalization_exclusions
exclusion_counts={side:dict(Counter(item['reason'] for item in exclusions if item['original_split']==side)) for side in ('train','validation')}
fit_hash=normalizer_fit_uid_sha256
test_uids=predecode_test_uids
def aliases_for(records): return set().union(*(set(record.get('source_aliases',[])) for record in records)) if records else set()
v2_records=[record for record in old_train_records if record.get('split_source')=='v2_reservoir']
fresh_train_records=[record for record in old_train_records if record.get('split_source')=='fresh_train']
fit_records=[simple_record(rows.iloc[int(i)]) for i in fit_ix]
early_records=[simple_record(rows.iloc[int(i)]) for i in early_stop_ix]
test_aliases=aliases_for(effective_test_records)
alias_sets={'v2_reservoir':aliases_for(v2_records),'fresh_train':aliases_for(fresh_train_records),'fresh_test':test_aliases,'fit':aliases_for(fit_records),'early_stop':aliases_for(early_records),'training_reservoir':aliases_for(old_train_records)}
alias_pairs=[('v2_reservoir','fresh_train'),('v2_reservoir','fresh_test'),('fresh_train','fresh_test'),('training_reservoir','fresh_test'),('fit','early_stop'),('fit','fresh_test'),('early_stop','fresh_test')]
alias_intersections={f'{left}__{right}':sorted(alias_sets[left]&alias_sets[right]) for left,right in alias_pairs}
assert all(not values for values in alias_intersections.values()), f'Export source-alias overlap audit failed: { {key:value[:10] for key,value in alias_intersections.items() if value} }'
fit_clusters=sorted({str(rows.iloc[int(i)].source_video_cluster) for i in fit_ix})
early_clusters=sorted({str(rows.iloc[int(i)].source_video_cluster) for i in early_stop_ix})
assert set(fit_clusters).isdisjoint(early_clusters)
manifest={
    'schema_version':1,'dataset':'Exploration-Lab/iSign v1.1','model_arm':'minilm_position_motion_v5','evaluation_status':('official_video_id_and_caption_disjoint_fresh_test; V4 paired on the same fresh cohort' if 'video_id' in metadata.columns else 'candidate_alias_and_caption_disjoint_fresh_test; UID-derived source identity is inferred; V4 paired on the same fresh cohort'),'selected_for_follow_up':True,
    'selection_metric':'group-balanced normalized position MSE + 0.05 x group-balanced normalized velocity MSE on source-grouped inner training validation only',
    'objective':{'group_weights':{'body':0.20,'left_hand':0.40,'right_hand':0.40},'velocity_coefficient':VELOCITY_WEIGHT,'confidence':'clipped point confidence; velocity confidence is minimum adjacent-frame confidence','normalization':'train-only per-point/per-axis z-score fit on gradient-fit UIDs'},
    'training_uids_used':training_uids_used,'early_stop_uids':[str(rows.iloc[int(i)].uid) for i in early_stop_ix],
    'paired_condition_comparisons':[{'name':'V5 vs frozen V4','left_condition':'true_text','right_condition':'v4_model_true','scope':'all_samples'}],
    'comparison_design':'Practical paired comparison on a fresh source/caption-disjoint cohort. V5 also changes training-set size, source/caption filtering, and inner selection grouping relative to V4; observed differences cannot be attributed to data scale alone.',
    'split':{'strategy':'all selected V2 UIDs, source aliases, and normalized captions reserved before deterministic fresh selection; fresh test selected first; inner stop selected by alias-connected components with exact captions held out','group_field':'source_video_id','inner_group_field':'source_video_cluster','source_alias_grouping':'conservative_source_alias_connected_component','source_video_id_provenance':source_id_provenance,
      'source_uid_parser':{'version':'v2-conservative','rules':['--N -> inferred 11-character source candidate; also reserve candidate with trailing hyphen','-N -> 12-hex source candidate','_eN -> inferred 11-character source candidate','_d -> inferred 11-character source candidate','_w is never stripped as an ID rule; reserve a validated 11/12-character stripped candidate as an alias','bare 11/12-character IDs retained'],'official_video_id_column_used':bool('video_id' in metadata.columns),'uid_rule_counts':source_rule_counts,'unsupported_uid_count':int((~metadata['source_uid_supported']).sum()),'mismatch_count':int(metadata['source_uid_mismatch'].sum()),'mismatch_examples':metadata.loc[metadata['source_uid_mismatch'],['uid','parsed_source_video_id','source_video_id']].head(20).to_dict(orient='records'),'fresh_test_allowed_uid_rules':sorted(set(fresh_candidates.source_uid_rule))},
      'inner_source_selection':{'method':'union-find connected components over shared conservative source_aliases; 10 percent of components assigned to early stop using deterministic SHA-256 ordering','fit_component_ids':fit_clusters,'early_stop_component_ids':early_clusters,'fit_alias_count':len(alias_sets['fit']),'early_stop_alias_count':len(alias_sets['early_stop']),'fit_early_alias_intersection':[]},
      'source_alias_audit':{'alias_construction':'whole UID plus parser candidate, official video_id when available, inferred suffix candidates, and validated ambiguous aliases','alias_counts':{key:len(value) for key,value in alias_sets.items()},'intersections':alias_intersections},
      'original_train':old_train_records,'original_validation':old_test_records,'train':effective_train_records,'validation':effective_test_records,'canonicalization_exclusions':exclusions,'canonicalization_exclusion_counts_by_original_split':exclusion_counts},
    'canonicalization':{'policy':'nose and both shoulders must be finite, confidence >= 0.15, and define a nondegenerate 2D x/y scale; interpolate center/log-scale if at least two valid anchors exist; apply 2D scale to x/y/z','threshold':MIN_ANCHOR_CONF,'resampled_frame_count':TMAX,'exclusion_count':len(exclusions),'exclusions':exclusions},
    'fixed_unseen_set':{'seed':SEED,'count':len(fixed_uids),'selection_rule':'sha256_rank_by_uid','selected_before_decode':True,'selection_universe_count':len(test_uids),'selection_universe_uid_sha256':hashlib.sha256('\n'.join(test_uids).encode()).hexdigest(),'uids':fixed_uids,'effective_uids':fixed_present_uids,'missing_uids':fixed_missing_uids},
    'topology':{'components':[{'name':'body','start':0,'count':33},{'name':'left_hand','start':33,'count':21},{'name':'right_hand','start':54,'count':21}]},
    'normalizer':{'path':'normalizer.npz','mean_key':'mean','std_key':'std','fit_split':'train','fit_uid_sha256':fit_hash,'fit_uid_hash_rule':'sha256(sorted fit UIDs joined by newline)'},
    'coordinate_space':'canonical nose/shoulder-centered pose; scale is 2D nose-to-shoulder midpoint in x/y and applied to x/y/z; predictions are denormalized canonical values, not z-scores',
    'resampling':{'resampled_frame_count':TMAX,'policy':'uniform_linear_48_endpoints_included','raw_frame_count':'V2 cached raw frame counts unavailable; newly decoded official pose entries store source frame counts'},
    'samples':sample_records,'metrics_path':'metrics_summary.json','follow_up_requirements':['This fresh cohort is a paired development benchmark and may inform later model choices. Reserve a new untouched source/caption cohort for final evaluation after architecture selection.','Final model scope still requires face and nonmanual expression representation plus fluent ISL signer intelligibility review.'],'license':'iSign CC-BY-NC-SA-4.0; research/non-commercial use only'}
(EXPORT_ROOT/'manifest_minilm_position_motion_v5.json').write_text(json.dumps(manifest,indent=2))
checkpoint={'format':'Hackcessible iSign V5 compact body+hands','arm':'minilm_position_motion_v5','config':run['config'],'state_dict':best_state,'normalizer_mean':normalizer_mean,'normalizer_std':normalizer_std,'normalizer_fit_uid_sha256':fit_hash,'point_indices':POINT_INDICES,'time_steps':TMAX,'topology':manifest['topology'],'frozen_text_encoder':'sentence-transformers/all-MiniLM-L6-v2','license':'iSign CC-BY-NC-SA-4.0; research/non-commercial use only'}
torch.save(checkpoint,EXPORT_ROOT/'checkpoints'/'minilm_position_motion_v5.pt')
tiny_src=Path('/kaggle/working/isl_v5_tiny_overfit.npz'); tiny_json=Path('/kaggle/working/isl_v5_tiny_overfit.json')
if tiny_src.exists(): shutil.copy2(tiny_src,EXPORT_ROOT/'diagnostics'/'tiny_overfit.npz')
if tiny_json.exists(): shutil.copy2(tiny_json,EXPORT_ROOT/'diagnostics'/'tiny_overfit.json')
readme=(f'V5 uses {len(training_pool_ix)} effective training-reservoir rows ({len(fit_ix)} gradient fit, {len(early_stop_ix)} alias-component inner stop) and {len(test_ix)} effective fresh test rows. '
        f'The original selection was {len(original_train_records)} training and {len(original_test_records)} test rows; every load/canonicalization exclusion is listed with its frozen side. '
        f'All V2 selected UIDs, conservative source aliases, and normalized captions were reserved before fresh test selection. V4 is a frozen paired baseline on the same fresh test. '
        'This comparison changes training-set size, source/caption filtering, and internal selection grouping together; it does not isolate a data-scale effect. '
        'The target includes body and hands only (75 points), omitting face mesh and nonmanual expression cues. Metrics do not prove ISL intelligibility; fluent signer review remains required. '
        f'Fixed qualitative UIDs selected before decoding: {len(fixed_uids)}; present={len(fixed_present_uids)}, missing={len(fixed_missing_uids)} with no replacements. '
        'The 500-row cohort is a paired development benchmark; reserve a new untouched source/caption cohort after architecture selection. Final scope still needs face/nonmanual representation and fluent ISL signer review. '
        'iSign license: CC-BY-NC-SA-4.0; research/non-commercial use only.\n')
(EXPORT_ROOT/'README.txt').write_text(readme)
archive_path=shutil.make_archive('/kaggle/working/hackcessible_isl_v5_artifacts','zip',root_dir=EXPORT_ROOT)
print('V5 export:',archive_path,'bytes=',Path(archive_path).stat().st_size)
print('Fresh test rows scored:',len(test_ix),'| fixed qualitative cohort:',len(fixed_uids),'| V5 true/shuffle:',metrics['true_text']['group_balanced_normalized_mse'],metrics['shuffled_text']['group_balanced_normalized_mse'])
print('V4 true-text paired baseline:',metrics['v4_model_true']['group_balanced_normalized_mse'],'| V4 checkpoint SHA256:',V4_CHECKPOINT_SHA256)
'''


def main() -> None:
    cells = [
        md(r'''# Hackcessible ISL text-to-pose — V5 fresh-cohort comparison

V5 keeps the V4 MiniLM pose architecture, target points, confidence-weighted group objective, and optimizer. It reuses the full selected V2 roster as training material and adds 8,000 selected iSign pose clips. A fresh 500-clip test is selected first and held out from V2, V5 training, and internal early-stop groups using conservative source aliases and normalized captions. The frozen V4 checkpoint is evaluated on the same fresh test as a paired reference.

V4 output was close to a near-static training mean (pooled motion ratio 0.0646), despite some text sensitivity. V5 changes both training-set size and source/caption split and selection policy; any difference is combined evidence, not a data-scale-only result. The 205-row V2 set is development material and is included only in the V5 training reservoir; it is not treated as a fresh test.

Enable Kaggle Internet, select **GPU T4 x2**, attach the verified V2 pose cache, and attach the private V4 checkpoint dataset containing `minilm_position_motion.pt`. Reuse the existing Kaggle `HF_TOKEN` secret. V5 still omits face and nonmanual expression features and requires fluent ISL review before any intelligibility claim. iSign is CC-BY-NC-SA-4.0 (research/non-commercial use only).'''),
        code(v3_code[1]),
        code(V5_DATA),
        code(V5_CANONICAL),
        code(v3_code[4]),
        code(v3_code[5]),
        code(V5_TRAIN),
        code(v3_code[7].replace('isl_v3_tiny_overfit', 'isl_v5_tiny_overfit').replace('V3', 'V5')),
        code(V5_EXPORT),
    ]
    # Correct the V4 internal checkpoint structure and normalizer reuse before generation.
    cells[4]["source"] = [line for line in "".join(cells[4]["source"]).replace(
        "training_uids_used = rows.iloc[fit_ix].uid.astype(str).tolist()",
        "training_uids_used = rows.iloc[fit_ix].uid.astype(str).tolist()"
    ).splitlines(keepends=True)]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "NvidiaTeslaT4",
            "colab": {"provenance": []},
            "kaggle": {"accelerator": "NvidiaTeslaT4", "dataSources": [], "isGpuEnabled": True, "isInternetEnabled": True, "language": "python", "sourceType": "notebook"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Cells: {len(cells)}; code bytes: {sum(len(''.join(c.get('source', []))) for c in cells):,}")


if __name__ == "__main__":
    main()
