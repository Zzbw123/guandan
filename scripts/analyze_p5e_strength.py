"""Read-only recomputation of existing P5c/P5e summaries; no training or replay."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'artifacts/evaluations/p5e-balanced-v1'
C = ROOT / 'artifacts/evaluations/p5c-diagnostic-v1'
bindings = {}


def checked(path, package):
    receipt = json.loads((package / 'delivery-receipt.json').read_text(encoding='utf-8'))
    key = path.relative_to(ROOT).as_posix()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != receipt['artifact_sha256'][key]:
        raise ValueError(f'Artifact hash mismatch: {key}')
    bindings[key] = actual
    return path


def rows(path, package):
    with checked(path, package).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


batch_summary = defaultdict(Counter)
for row in rows(E / 'training-batches.csv', E):
    item = batch_summary[row['job']]
    item['updates'] += 1
    item['single_class_updates'] += row['single_class'] == 'True'
    item['samples'] += int(row['n'])
    item['positive'] += int(row['positive'])
for item in batch_summary.values():
    item['single_class_fraction'] = item['single_class_updates'] / item['updates']

training = {r['group']: r for r in rows(C / 'training-summary.csv', C)}
positive_sources = {}
for seed in (314510, 314511, 314512):
    job = f'mixed-{seed}'
    positive_sources[job] = {
        'all_positive': int(training[job]['positive']),
        'block2_positive': int(training[job + '/block-2']['positive']),
        'block2_fraction': int(training[job + '/block-2']['positive']) / int(training[job]['positive']),
    }

validation = {}
for arm in ('ordinary', 'label_balanced'):
    directory = E / 'evaluations' / f'validation-{arm}'
    result_path = checked(directory / 'results.jsonl', E)
    records = [json.loads(line) for line in result_path.read_text(encoding='utf-8').splitlines()]
    ids = {r['trial_id'] for r in records}
    assert len(ids) == len(records) == 1560
    summary = defaultdict(Counter)
    deals = defaultdict(set)
    for row in records:
        assert row['status'] == 'ok' and row['illegal_actions'] == row['timeouts'] == 0
        item = summary[row['matchup_id']]
        item['games'] += 1
        item['wins'] += row['win']
        deals[row['matchup_id']].add(row['deal_seed'])
    for matchup, item in summary.items():
        assert item['games'] == 520 and len(deals[matchup]) == 65
        item['win_rate'] = item['wins'] / item['games']
        item['original_deals'] = len(deals[matchup])
    behavior_path = checked(directory / 'behavior.jsonl', E)
    behavior = [json.loads(line) for line in behavior_path.read_text(encoding='utf-8').splitlines()]
    assert len(behavior) == len(ids) and {r['trial_id'] for r in behavior} == ids
    totals = Counter()
    for row in behavior:
        totals.update(row['counts'])
    rates = {}
    for name, numerator, denominator in (
        ('optional_pass', 'optional_passes', 'optional_pass_opportunities'),
        ('finish_miss', 'missed_finishes', 'finish_opportunities'),
        ('endgame_optional_pass', 'endgame_optional_passes', 'endgame_optional_pass_opportunities'),
    ):
        rates[name] = {'numerator': totals[numerator], 'denominator': totals[denominator],
                       'rate': totals[numerator] / totals[denominator] if totals[denominator] else None}
    validation[arm] = {'matchups': dict(summary), 'behavior_all_three_opponents': rates}

print(json.dumps({
    'scope': 'Existing artifact hash checks and arithmetic only; no new training, evaluation, bootstrap, or replay audit.',
    'batch_summary': dict(batch_summary), 'positive_sources_p5c': positive_sources,
    'validation_p5e': validation, 'verified_artifact_sha256': bindings,
}, ensure_ascii=False, indent=2))
