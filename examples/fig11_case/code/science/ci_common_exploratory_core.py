"""Strict object scores plus explicitly conditional FP promotions.

Two windows stay independent. All source masks/truth stay unchanged.
No inference, data access or trained-model dependency in this numerical module.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from ci_v4_relaxed_core import components, matching, prediction_segments

MINUTE = 60_000_000_000
SUM_KEYS = ('tp', 'fn', 'fp', 'observed_objects', 'predicted_objects', 'valid_cases',
    'negative_cases', 'negative_alarm_cases', 'observed_covered', 'predicted_supported',
    'duplicate_promotions', 'new35_promotions', 'weak_promotions', 'brief_promotions',
    'topology_flagged_promotions', 'radar_candidate_FP_coverage_rejected',
    'new20_matched_FP_without_complete_profile', 'FP_in_incomplete_radar_windows',
    'observed_pixels', 'predicted_pixels', 'valid_pixels', 'predicted_pixels_near_reference',
    'predictions_near_multiple_references', 'references_near_multiple_predictions',
    'exact_core_oracle_tp', 'exact_core_oracle_fn')


def truth_fields(labels, dy, dx):
    return np.asarray([distance_transform_edt(labels != k, sampling=(dy, dx))
                       for k in range(1, int(labels.max()) + 1)])


def matrix(fields, labels, count):
    if not len(fields) or not count:
        return np.empty((len(fields), count), np.float64)
    points, starts = prediction_segments(labels)
    return np.asarray([np.minimum.reduceat(field.ravel()[points], starts) for field in fields])


def true_value(value):
    return str(value).lower() in ('true', '1', '1.0')


def finite_number(value):
    try:
        return float(value) if np.isfinite(float(value)) else np.nan
    except (TypeError, ValueError):
        return np.nan


def episode_flags(event, start, end):
    """Same permissive requested rule as the completed A3 FP promotion study.

Follow-up is the observed first 30 min after the new20 birth, which can extend
beyond the forecast bin. Its coverage/lineage limitation is always reported.
"""
    f = event.get('profile', {})
    complete = f.get('profile_status') == 'complete_observed_30min'
    birth = int(event['birth_ns'])
    onset = f.get('persistent35_birth_ns')
    persistent_in_bin = onset is not None and start < int(onset) <= end
    growth = (true_value(f.get('growth_candidate')) or true_value(f.get('weak_growth_candidate')) or
        finite_number(f.get('p90_gain_dbz')) >= 5 or
        (finite_number(f.get('native_area_growth_ratio')) >= 1.5 and
         finite_number(f.get('native_area_gain_pixels')) >= 3))
    eligible = complete and start < birth <= end
    return dict(complete=complete, new35=eligible and persistent_in_bin,
        weak=eligible and growth and not persistent_in_bin,
        brief=eligible and finite_number(f.get('max_native_dbz')) >= 35 and not persistent_in_bin,
        topology=true_value(f.get('lineage_topology_review')))


def context(core, valid, radar_coverage, new20, new35, dy, dx, start, end):
    labels, no = components(np.asarray(core, bool) & valid)
    fields = truth_fields(labels, dy, dx)
    def event_fields(events):
        result = []
        for e in events:
            binary = np.zeros(valid.shape, bool)
            binary.ravel()[np.asarray(e['flat'], int)] = True
            result.append(distance_transform_edt(~binary, sampling=(dy, dx)))
        return np.asarray(result)
    to_truth = (np.min(fields, axis=0) if no else np.full(valid.shape, np.inf))
    coverage = None if radar_coverage is None else np.array(radar_coverage, dtype=bool, copy=True)
    if coverage is None or coverage.all():
        to_bad = None
    else:
        to_bad = distance_transform_edt(coverage, sampling=(dy, dx))
    return dict(valid=np.asarray(valid, bool), labels=labels, no=no, fields=fields,
        sizes=np.bincount(labels.ravel(), minlength=no + 1)[1:],
        to_truth=to_truth, coverage=coverage, to_bad=to_bad,
        new20=new20, fields20=event_fields(new20), fields35=event_fields(new35),
        flags20=[episode_flags(e, start, end) for e in new20], dy=dy, dx=dx)


def coverage_ok(ctx, labels, original_id, tolerance):
    if ctx['coverage'] is None:
        return False
    footprint = labels == original_id
    if ctx['to_bad'] is None or (ctx['to_bad'][footprint] > tolerance + 1e-8).all():
        return True
    vicinity = distance_transform_edt(~footprint, sampling=(ctx['dy'], ctx['dx'])) <= tolerance + 1e-8
    return bool(ctx['coverage'][vicinity].mean() >= .95)


def evaluate_probability(probability, ctx, thresholds, areas=(1, 2, 3), tolerances=(5., 10., 20.)):
    valid = ctx['valid']
    if not np.isfinite(probability[valid]).all():
        raise RuntimeError('Nonfinite predicted probabilities')
    output = []
    for threshold in thresholds:
        predicted, nall = components((probability >= threshold) & valid)
        sizes = np.bincount(predicted.ravel(), minlength=nall + 1)[1:]
        d = matrix(ctx['fields'], predicted, nall)
        d20 = matrix(ctx['fields20'], predicted, nall)
        d35 = matrix(ctx['fields35'], predicted, nall)
        geometry_cache = {}
        for area in areas:
            ids = np.flatnonzero(sizes >= area) + 1
            n = len(ids)
            distances = d[:, ids - 1]
            twenty, thirtyfive = d20[:, ids - 1], d35[:, ids - 1]
            keep = np.zeros(nall + 1, bool); keep[ids] = True
            footprint = keep[predicted]
            pixels = int(footprint.sum())
            for tolerance in tolerances:
                allowed = distances <= tolerance + 1e-8
                pairs = matching(distances, tolerance)
                matched = {pi for _, pi in pairs}
                tp = len(pairs); fn = ctx['no'] - tp; fp = n - tp
                degree_o, degree_p = allowed.sum(axis=1), allowed.sum(axis=0)
                assign20 = {pi: oi for oi, pi in matching(twenty, tolerance)}
                assign35 = {pi: oi for oi, pi in matching(thirtyfive, tolerance)}
                promoted = dict(duplicate_promotions=0, new35_promotions=0,
                    weak_promotions=0, brief_promotions=0, topology_flagged_promotions=0,
                    radar_candidate_FP_coverage_rejected=0, new20_matched_FP_without_complete_profile=0)
                for pi, original_id in enumerate(ids):
                    if pi in matched:
                        continue
                    if degree_p[pi] > 0:
                        promoted['duplicate_promotions'] += 1
                        continue
                    evidence = ctx['flags20'][assign20[pi]] if pi in assign20 else None
                    if evidence is not None and not evidence['complete']:
                        promoted['new20_matched_FP_without_complete_profile'] += 1
                    new35 = pi in assign35 or (evidence is not None and evidence['new35'])
                    weak = evidence is not None and evidence['weak']
                    brief = evidence is not None and evidence['brief']
                    if not (new35 or weak or brief):
                        continue
                    cache_key = (int(original_id), float(tolerance))
                    if cache_key not in geometry_cache:
                        geometry_cache[cache_key] = coverage_ok(ctx, predicted, int(original_id), tolerance)
                    if not geometry_cache[cache_key]:
                        promoted['radar_candidate_FP_coverage_rejected'] += 1
                        continue
                    if new35:
                        promoted['new35_promotions'] += 1
                    elif weak:
                        promoted['weak_promotions'] += 1
                    else:
                        promoted['brief_promotions'] += 1
                    if evidence is not None and evidence['topology']:
                        promoted['topology_flagged_promotions'] += 1
                output.append(dict(threshold=float(threshold), min_predicted_pixels=int(area),
                    tolerance_km=float(tolerance), tp=tp, fn=fn, fp=fp,
                    observed_objects=ctx['no'], predicted_objects=n, valid_cases=int(valid.any()),
                    negative_cases=int(valid.any() and ctx['no'] == 0),
                    negative_alarm_cases=int(valid.any() and ctx['no'] == 0 and n > 0),
                    observed_covered=int((degree_o > 0).sum()), predicted_supported=int((degree_p > 0).sum()),
                    FP_in_incomplete_radar_windows=fp if ctx['coverage'] is None else 0,
                    observed_pixels=int(ctx['sizes'].sum()), predicted_pixels=pixels,
                    valid_pixels=int(valid.sum()),
                    predicted_pixels_near_reference=int((footprint & (ctx['to_truth'] <= tolerance + 1e-8)).sum()),
                    predictions_near_multiple_references=int((degree_p > 1).sum()),
                    references_near_multiple_predictions=int((degree_o > 1).sum()),
                    exact_core_oracle_tp=int((ctx['sizes'] >= area).sum()),
                    exact_core_oracle_fn=int((ctx['sizes'] < area).sum()), **promoted))
    return output


def div(a, b):
    return float(a / b) if b else np.nan


def rates(c):
    r = dict(c)
    tp, fn, fp = r['tp'], r['fn'], r['fp']
    physical = r['new35_promotions'] + r['weak_promotions'] + r['brief_promotions']
    promoted = physical + r['duplicate_promotions']
    if promoted > fp:
        raise RuntimeError('Promotion exceeds original FP count')
    r.update(csi=div(tp, tp + fn + fp), pod=div(tp, tp + fn), far=div(fp, tp + fp),
        negative_case_fpr=div(r['negative_alarm_cases'], r['negative_cases']),
        promoted=promoted, tp_exploratory=tp + promoted, fn_original=fn, fp_remaining=fp - promoted,
        csi_exploratory=div(tp + promoted, tp + fn + fp),
        pod_exploratory=div(tp + promoted, tp + promoted + fn),
        far_exploratory=div(fp - promoted, tp + fp),
        csi_without_duplicate_promotions=div(tp + physical, tp + fn + fp),
        radar_growth_promotions=physical,
        coverage_recall=div(r['observed_covered'], r['observed_objects']),
        coverage_precision=div(r['predicted_supported'], r['predicted_objects']),
        predicted_to_observed_area_ratio=div(r['predicted_pixels'], r['observed_pixels']),
        predicted_area_fraction=div(r['predicted_pixels'], r['valid_pixels']),
        predicted_pixel_support_fraction=div(r['predicted_pixels_near_reference'], r['predicted_pixels']))
    a, b = r['coverage_recall'], r['coverage_precision']
    r['coverage_f1'] = (2 * a * b / (a + b) if a + b else 0.) if np.isfinite(a + b) else np.nan
    return r


def selftest():
    shape = (35, 45); valid = np.ones(shape, bool); core = np.zeros(shape, bool)
    core[5, 5] = core[5, 10] = True
    p = np.zeros(shape); p[4:7, 5:11] = .2; p[25:28, 30:33] = .2
    start = 1_700_000_000_000_000_000; end = start + 30 * MINUTE
    e = dict(flat=np.array([26 * shape[1] + 31]), birth_ns=start + MINUTE,
             profile=dict(profile_status='complete_observed_30min', growth_candidate=True,
                          max_native_dbz=29., lineage_topology_review=True))
    c = context(core, valid, valid, [e], [], 1, 1, start, end)
    row = rates(evaluate_probability(p, c, [.03], areas=(3,), tolerances=(2.,))[0])
    assert (row['tp'], row['fn'], row['fp']) == (1, 1, 1)
    assert row['weak_promotions'] == 1 and row['topology_flagged_promotions'] == 1
    assert row['observed_covered'] == 2 and row['csi_exploratory'] == 2/3
    # One CI and two disconnected predictions: remaining forecast promoted as duplicate.
    core[:] = False; core[5, 8] = True
    p[:] = 0; p[4:7, 5:7] = .2; p[4:7, 10:12] = .2
    c = context(core, valid, None, [], [], 1, 1, start, end)
    row = evaluate_probability(p, c, [.03], areas=(3,), tolerances=(5.,))[0]
    assert row['duplicate_promotions'] == 1 and row['FP_in_incomplete_radar_windows'] == 1
    # Window2 does not accept an episode born in window1 as new weak growth.
    assert not episode_flags(e, end, end + 30 * MINUTE)['weak']
    # Missing profiles never count as weak/brief growth.
    assert not episode_flags(dict(flat=e['flat'], birth_ns=e['birth_ns']), start, end)['weak']
    # Prior growth, static weak echo and missing coverage are not automatically accepted.
    e['birth_ns'] = start - MINUTE
    assert not episode_flags(e, start, end)['weak']
    e['birth_ns'] = start + MINUTE; e['profile']['growth_candidate'] = False
    assert not episode_flags(e, start, end)['weak']
    e['profile']['max_native_dbz'] = 40.
    assert episode_flags(e, start, end)['brief']
    # Tiny perfect reference preserved at min1, removed at min2/3.
    core[:] = False; core[5, 5] = True; p[:] = 0; p[5, 5] = .2
    c = context(core, valid, valid, [], [], 1, 3, start, end)
    rows = evaluate_probability(p, c, [.03], tolerances=(0.,))
    assert [r['tp'] for r in rows] == [1, 0, 0]
    # Physical anisotropy.
    p[:] = 0; p[5, 7] = .2
    rows = evaluate_probability(p, c, [.03], areas=(1,), tolerances=(5.,6.))
    assert [r['tp'] for r in rows] == [0,1]
    print('COMMON TWO-WINDOW EXPLORATORY SELFTEST PASSED: strict matching, tiny predictions, duplicates, growth, timing, incomplete evidence, physical distances', flush=True)
