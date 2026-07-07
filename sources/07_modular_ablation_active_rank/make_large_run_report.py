from __future__ import annotations
import csv, json
from pathlib import Path

root = Path('ablation_runs_large3600_chunked')
items = [
    ('distance_rnn_300', root/'distance_eval_300'/'meta_plastic_paper_task_eval_summary.json'),
    ('distance_no_rnn_300', root/'distance_no_rnn_ablation'/'meta_plastic_paper_task_eval_summary.json'),
    ('distance_no_plasticity_300', root/'distance_no_plasticity_ablation'/'meta_plastic_paper_task_eval_summary.json'),
    ('distance_constant_reliability_300', root/'distance_constant_reliability_ablation'/'meta_plastic_paper_task_eval_summary.json'),
    ('distance_posterior_mean_300', root/'distance_posterior_mean_readout'/'meta_plastic_paper_task_eval_summary.json'),
    ('raw_bars_rnn_100', root/'raw_eval_100'/'meta_plastic_paper_task_eval_summary.json'),
    ('raw_bars_no_rnn_100', root/'raw_observation_no_rnn_ablation'/'meta_plastic_paper_task_eval_summary.json'),
    ('full_suite_distance_rnn_120', Path('ablation_runs_full_3600/distance_rnn_reliability/meta_plastic_paper_task_eval_summary.json')),
    ('full_suite_distance_no_rnn_120', Path('ablation_runs_full_3600/distance_no_rnn_ablation/meta_plastic_paper_task_eval_summary.json')),
]
keys = [
    'overall_accuracy','learned_pairs_accuracy','nonlearned_pairs_accuracy',
    'consistent_error_subjects_80pct_ratio','consistent_error_subjects_100pct_ratio',
    'mean_self_consistency_from_majority_choices','mean_circular_triads_from_majority_choices',
    'mean_inter_subject_kendall_tau','correct_ranking_subjects','self_consistent_incorrect_subjects',
    'self_inconsistent_subjects','mean_posterior_entropy','mean_hebb_gate','mean_edge_strength','edge_recon_loss'
]
rows = []
for name, p in items:
    if not p.exists():
        rows.append({'variant': name, 'status': 'missing'})
        continue
    s = json.loads(p.read_text())
    row = {'variant': name, 'status': 'ok'}
    sig = s.get('ablation_signature', {})
    for k, v in sig.items(): row[k] = v
    for k in keys: row[k] = s.get(k, '')
    bc = s.get('beta_pair_category_counts', {})
    row['beta_bimodal_pairs'] = bc.get('bimodal', '')
    row['distance_accuracy'] = json.dumps(s.get('distance_accuracy', {}), ensure_ascii=False)
    row['serial_position_accuracy'] = json.dumps(s.get('serial_position_accuracy', {}), ensure_ascii=False)
    rows.append(row)
fields = list(dict.fromkeys(k for r in rows for k in r.keys()))
with open(root/'large3600_summary.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def f(row, k):
    try: return float(row.get(k,''))
    except Exception: return None
by = {r['variant']: r for r in rows}
base = by['distance_rnn_300']; nor = by['distance_no_rnn_300']; nop = by['distance_no_plasticity_300']; const = by['distance_constant_reliability_300']; pm = by['distance_posterior_mean_300']; raw = by['raw_bars_rnn_100']; rawn = by['raw_bars_no_rnn_100']
lines = []
lines.append('# 3600s large-batch / larger-iteration ablation report')
lines.append('')
lines.append('## Test plan and expectations')
lines.append('')
lines.append('- Increase scale from smoke settings (30 iterations, batch 4, hidden 16) to larger settings. The stable chunked run used batch 16, hidden 32, item dim 16, subject dim 10; distance-input model reached 300 effective meta-training iterations, and raw-bar model reached 100 iterations.')
lines.append('- Expected before running: distance-input RNN reliability should improve paper-task accuracy, preserve learned >= non-learned accuracy, preserve high self-consistency and bimodal pair distributions, and show a positive symbolic-distance trend.')
lines.append('- Expected before running: no-RNN, no-plasticity, and constant-reliability ablations should drop clearly if the mechanism is truly RNN-dependent and reliability is coming from RNN activity.')
lines.append('- Expected before running: posterior-mean readout should reduce self-consistency, because stable idiosyncratic rankings require subject-level commitment.')
lines.append('- Expected before running: raw-bar input should remain harder than signed-distance input, but longer training should move it above chance; no-RNN should not outperform the trained raw-bar RNN.')
lines.append('')
lines.append('## Execution notes')
lines.append('')
lines.append('- A direct full-suite call was launched with `--timeout 3600 --big-timeout 3600`. The 120-iteration distance baseline completed, but the long shell call was killed by the environment while progressing into later commands; its completed baseline and no-RNN eval are kept under `ablation_runs_full_3600/`.')
lines.append('- I then used safe chunked continuation under the same code path to avoid losing checkpoints when the environment killed long foreground jobs. The attempted single 1200-iteration background run reached about 452/1200 before being killed and did not produce a final checkpoint because it had `save_every=0`, so it is treated only as a failed long-run attempt, not as a valid result.')
lines.append('')
lines.append('## Main results')
lines.append('')
lines.append('| variant | overall | learned | nonlearned | c80 | c100 | self-consistency | tau | bimodal pairs | entropy | edge strength |')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for name in ['distance_rnn_300','distance_no_rnn_300','distance_no_plasticity_300','distance_constant_reliability_300','distance_posterior_mean_300','raw_bars_rnn_100','raw_bars_no_rnn_100','full_suite_distance_rnn_120','full_suite_distance_no_rnn_120']:
    r = by.get(name, {'status':'missing'})
    def fmt(k):
        v = f(r,k)
        return '' if v is None else f'{v:.3f}'
    lines.append('| ' + ' | '.join([name, fmt('overall_accuracy'), fmt('learned_pairs_accuracy'), fmt('nonlearned_pairs_accuracy'), fmt('consistent_error_subjects_80pct_ratio'), fmt('consistent_error_subjects_100pct_ratio'), fmt('mean_self_consistency_from_majority_choices'), fmt('mean_inter_subject_kendall_tau'), str(r.get('beta_bimodal_pairs','')), fmt('mean_posterior_entropy'), fmt('mean_edge_strength')]) + ' |')
lines.append('')
lines.append('## Expectation vs observed')
lines.append('')
base_drop = f(base,'overall_accuracy') - f(nor,'overall_accuracy')
nop_drop = f(base,'overall_accuracy') - f(nop,'overall_accuracy')
const_delta = f(const,'overall_accuracy') - f(base,'overall_accuracy')
raw_drop = f(raw,'overall_accuracy') - f(rawn,'overall_accuracy')
lines.append(f'- Distance RNN baseline at 300 iterations: overall={f(base,"overall_accuracy"):.3f}, learned={f(base,"learned_pairs_accuracy"):.3f}, nonlearned={f(base,"nonlearned_pairs_accuracy"):.3f}. This is better than the completed 120-iteration full-suite baseline ({f(by["full_suite_distance_rnn_120"],"overall_accuracy"):.3f}) but worse than the earlier smoke run reported before (~0.638).')
lines.append(f'- RNN necessity did not hold at this checkpoint: no-RNN overall={f(nor,"overall_accuracy"):.3f}, drop={base_drop:.3f}, far below the desired >=0.15 drop.')
lines.append(f'- Plasticity necessity also did not hold: no-plasticity overall={f(nop,"overall_accuracy"):.3f}, drop={nop_drop:.3f}.')
lines.append(f'- Reliability-from-RNN did not win against the constant-reliability control: constant reliability overall={f(const,"overall_accuracy"):.3f}, which is {const_delta:.3f} above the RNN-reliability baseline.')
lines.append(f'- Commitment/readout expectation held strongly: posterior-mean readout reduced self-consistency from {f(base,"mean_self_consistency_from_majority_choices"):.3f} to {f(pm,"mean_self_consistency_from_majority_choices"):.3f}, increased circular triads to {f(pm,"mean_circular_triads_from_majority_choices"):.3f}, and eliminated bimodal pair fits.')
lines.append(f'- Raw-bar input did not improve at 100 iterations: raw RNN overall={f(raw,"overall_accuracy"):.3f}; raw no-RNN overall={f(rawn,"overall_accuracy"):.3f}; trained raw RNN is worse by {-raw_drop:.3f}.')
lines.append('')
lines.append('## Interpretation')
lines.append('')
lines.append('The larger runs did not produce the ideal mechanism. They show that subject-level rank commitment reliably generates self-consistent idiosyncratic rankings, but RNN-derived reliability is not yet the source of behavioral success. At the 300-iteration distance checkpoint, learned/non-learned accuracy and distance trend are present but weak; RNN ablation barely hurts; no-plasticity barely hurts; and constant reliability outperforms RNN reliability. This means the current bottom layer still lets the active-rank / commitment module explain much of the behavior without using the RNN in the intended way.')
lines.append('')
lines.append('The likely failure mode is visible in the diagnostics: edge strength falls very low in the trained RNN-reliability model, while posterior entropy remains high. Constant reliability restores stronger effective edge evidence and improves accuracy. In other words, the network is learning an under-confident reliability/write gate rather than a useful reliability signal.')
lines.append('')
lines.append('## Next code-level direction')
lines.append('')
lines.append('- Add an explicit auxiliary loss that makes RNN hidden activity predict signed relation and reliability separately: keep relation encoding RNN-dependent, but prevent the reliability head from collapsing toward weak edges.')
lines.append('- Replace the current constant posterior precision/readout with a calibrated evidence-temperature learned from RNN activity, then test whether no-RNN destroys that temperature calibration.')
lines.append('- For raw bars, add a small sensory-difference encoder before the RNN, trained jointly, because the current raw input does not reliably recover relational sign within 100 iterations.')
lines.append('- Keep posterior-mean readout as an important negative control: it correctly shows that commitment, not just noisy trial choices, is required for the paper-like stable/self-consistent errors.')
(root/'large3600_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(root/'large3600_summary.csv')
print(root/'large3600_report.md')
