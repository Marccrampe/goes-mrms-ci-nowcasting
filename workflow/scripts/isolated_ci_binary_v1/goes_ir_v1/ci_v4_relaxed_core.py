"""Explicit radar proxies for diagnostic use, not certified physical CI truth."""
from ci_nowcasting.paths import resolve_path as _ci_resolve, project_root as _ci_project, source_root as _ci_source
from collections import defaultdict, deque
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.optimize import linear_sum_assignment
LEVELS = (20, 25, 30, 35)

def components(binary, min_area=1):
    labels, n = label(binary, np.ones((3, 3), np.uint8))
    if n and min_area > 1:
        keep = np.bincount(labels.ravel()) >= min_area
        keep[0] = False
        labels, n = label(keep[labels], np.ones((3, 3), np.uint8))
    return (labels, n)

class EpisodeTracker:
    """Overlap/nearest-edge lineage on the GOES verification grid.

    Consecutive frames only; continuation, splits and mergers do not create a
    new birth. Ambiguous pending mergers and missing-history births are not
    eligible new-echo proxies. Activity verification still includes persistent
    preexisting echoes. Linking cannot prove individual cell identity.
    """

    def __init__(self, shape, dy, dx, scans=3, speed_km_min=2.0, max_gap_s=180):
        self.shape, self.dy, self.dx = (shape, dy, dx)
        self.scans, self.speed, self.max_gap = (scans, speed_km_min, max_gap_s)
        self.previous = np.zeros(shape, np.int32)
        self.previous_roots = {}
        self.previous_time = None
        self.events = {}
        self.next_id = 1

    def update(self, binary, valid, stamp, frame, history_valid, prior_echo=None, intensity=None):
        labels, n = components(binary & valid)
        dt = (stamp - self.previous_time) / 60000000000.0 if self.previous_time is not None else np.inf
        continuous = 0 < dt * 60 <= self.max_gap
        if not continuous:
            self.previous.fill(0)
            self.previous_roots = {}
        prev_present = self.previous > 0
        if prev_present.any():
            distance, nearest = distance_transform_edt(~prev_present, sampling=(self.dy, self.dx), return_indices=True)
            near_labels = self.previous[tuple(nearest)]
        else:
            distance = np.full(self.shape, np.inf)
            near_labels = np.zeros(self.shape, np.int32)
        groups, roots = (defaultdict(list), {})
        for component in range(1, n + 1):
            mask = labels == component
            direct = np.unique(self.previous[mask])
            direct = direct[direct > 0]
            parents = {self.previous_roots[int(p)] for p in direct}
            motion_link = False
            if not parents and continuous:
                close = mask & (distance <= self.speed * dt + 1e-08)
                ids = np.unique(near_labels[close])
                ids = ids[ids > 0]
                parents = {self.previous_roots[int(p)] for p in ids}
                motion_link = bool(parents)
            if parents:
                root = min(parents, key=lambda r: self.events[r]['birth_ns'])
                if len(parents) > 1:
                    for parent in parents:
                        if not self.events[parent]['qualified']:
                            self.events[parent]['ambiguous_before_confirmation'] = True
                if motion_link:
                    self.events[root]['motion_only_links'] += 1
            else:
                root = self.next_id
                self.next_id += 1
                self.events[root] = dict(track_id=root, birth_ns=int(stamp), last_ns=int(stamp), birth_flat=np.flatnonzero(mask).astype(np.int32), censored_history=not bool(history_valid[mask].all()), prior_echo_at_birth_location=bool(prior_echo[mask].any()) if prior_echo is not None else False, peak_dbz_on_observed_lineage=None, first_ge35_ns=None, ambiguous_before_confirmation=False, motion_only_links=0, qualified=False, qualifying_ns=None, consecutive=0, recent=deque(maxlen=self.scans), updates=0)
            roots[component] = root
            groups[root].append(np.flatnonzero(mask).astype(np.int32))
        updates = []
        for root, parts in groups.items():
            event = self.events[root]
            points = np.concatenate(parts)
            previous_stamp = event['last_ns']
            event['consecutive'] = event['consecutive'] + 1 if event['updates'] and 0 < stamp - previous_stamp <= self.max_gap * 1000000000.0 else 1
            event['last_ns'] = int(stamp)
            event['updates'] += 1
            if intensity is not None:
                peak = float(intensity.ravel()[points].max())
                old = event['peak_dbz_on_observed_lineage']
                event['peak_dbz_on_observed_lineage'] = peak if old is None else max(old, peak)
                if peak >= 35 and event['first_ge35_ns'] is None:
                    event['first_ge35_ns'] = int(stamp)
            event['recent'].append((int(frame), points))
            enough = event['consecutive'] >= self.scans and stamp - event['birth_ns'] >= (self.scans - 1) * 90000000000.0
            newly_qualified = enough and (not event['qualified'])
            if newly_qualified:
                event['qualified'] = True
                event['qualifying_ns'] = int(stamp)
                updates.extend(event['recent'])
            elif event['qualified']:
                updates.append((int(frame), points))
        self.previous, self.previous_roots = (labels, roots)
        self.previous_time = int(stamp)
        return updates

    def export(self):
        result = []
        for event in self.events.values():
            row = {k: v for k, v in event.items() if k not in ('recent', 'consecutive', 'updates')}
            row['birth_flat'] = event['birth_flat'].tolist()
            row['observed_lifetime_minutes'] = (event['last_ns'] - event['birth_ns']) / 60000000000.0
            row['eligible_new_echo_proxy'] = bool(event['qualified'] and (not event['censored_history']) and (not event['ambiguous_before_confirmation']) and (not event['prior_echo_at_birth_location']))
            result.append(row)
        return result

def object_distance_fields(objects, shape, dy, dx):
    result = []
    for obj in objects:
        mask = np.zeros(shape, bool)
        mask.ravel()[obj['flat']] = True
        result.append(distance_transform_edt(~mask, sampling=(dy, dx)).astype(np.float32))
    return result

def prediction_segments(predicted_labels):
    points = np.flatnonzero(predicted_labels).astype(np.int32)
    values = predicted_labels.ravel()[points]
    order = np.argsort(values, kind='stable')
    ordered = points[order]
    starts = np.r_[0, np.flatnonzero(np.diff(values[order])) + 1] if len(points) else np.array([], np.int32)
    return (ordered, starts)

def distance_matrix(fields, predicted_labels, count, segments=None):
    if not len(fields) or not count:
        return np.empty((len(fields), count), np.float32)
    points, starts = prediction_segments(predicted_labels) if segments is None else segments
    if len(starts) != count:
        raise RuntimeError('Nonconsecutive predicted component labels')
    return np.asarray([np.minimum.reduceat(field.ravel()[points], starts) for field in fields])

def matching(distance, tolerance):
    if not distance.size:
        return []
    allowed = distance <= tolerance + 1e-08
    penalty = (min(distance.shape) + 1) * (tolerance + 1)
    rows, cols = linear_sum_assignment(np.where(allowed, distance, penalty))
    return [(int(r), int(c)) for r, c in zip(rows, cols) if allowed[r, c]]

def neutral_counts(distance, matched, n_pred, neutral_km=30):
    assigned = {p for _, p in matched}
    neutral = duplicate = far = 0
    for p in range(n_pred):
        if p in assigned:
            continue
        nearest = float(distance[:, p].min()) if len(distance) else np.inf
        if nearest <= 5 + 1e-08:
            duplicate += 1
        elif nearest <= neutral_km + 1e-08:
            neutral += 1
        else:
            far += 1
    return (neutral, duplicate, far)

def selftest():
    shape = (24, 30)
    valid = np.ones(shape, bool)
    empty = np.zeros(shape, bool)
    tracker = EpisodeTracker(shape, 1.0, 1.0)
    for i in range(16):
        tracker.update(empty, valid, i * 120000000000, i, valid)
    for i in range(16, 19):
        field = empty.copy()
        field[8:10, 6 + (i - 16):8 + (i - 16)] = True
        tracker.update(field, valid, i * 120000000000, i, valid)
    events = tracker.export()
    assert len(events) == 1 and events[0]['eligible_new_echo_proxy']
    tracker = EpisodeTracker(shape, 1.0, 1.0)
    for i in range(3):
        field = empty.copy()
        field[8:10, 6:8] = True
        tracker.update(field, valid, i * 120000000000, i, empty)
    assert tracker.export()[0]['qualified'] and (not tracker.export()[0]['eligible_new_echo_proxy'])
    tracker.update(field, valid, 360000000000, 3, valid)
    split = empty.copy()
    split[8, 6] = True
    split[9, 9] = True
    tracker.update(split, valid, 480000000000, 4, valid)
    assert len(tracker.export()) == 1
    tracker = EpisodeTracker(shape, 1.0, 1.0)
    for i in range(3):
        tracker.update(field, valid, i * 120000000000, i, valid, field)
    assert tracker.export()[0]['qualified'] and (not tracker.export()[0]['eligible_new_echo_proxy'])
    tracker = EpisodeTracker(shape, 1.0, 1.0)
    tracker.update(field, valid, 0, 0, valid)
    tracker.update(empty, valid, 120000000000, 1, valid)
    assert not tracker.export()[0]['qualified']
    tracker.update(field, valid, 900000000000, 2, empty)
    assert len(tracker.export()) == 2 and tracker.export()[1]['censored_history']
    assert len(matching(np.array([[0.0, 1.0], [1.0, 50.0]]), 2)) == 2
    assert matching(np.empty((0, 3)), 30) == []
    d = np.array([[0.0, 0.0, 15.0, 40.0]])
    assert neutral_counts(d, [(0, 0)], 4) == (1, 1, 1)
    fields = object_distance_fields([dict(flat=[5 * 30 + 5])], shape, 1, 2)
    p = np.zeros(shape, np.int32)
    p[5, 7] = 1
    assert float(distance_matrix(fields, p, 1)[0, 0]) == 4
    print('RELAXED RADAR SELFTEST PASSED: motion, censoring, split lineage, prior echoes, persistence, one-to-one, neutral band, anisotropy', flush=True)
if __name__ == '__main__':
    selftest()
