"""Full-domain Figure 11 verification playground, independent of cluster paths."""
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pandas as pd
import object_demo_kernels as K

GREEN, RED, BLUE, ORANGE, MAGENTA = '#009E73', '#D62728', '#0072B2', '#E69F00', '#CC79A7'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def ratio(a, b):
    return a / b if b else np.nan


class CaseStudy:
    def __init__(self, root):
        self.root = Path(root)
        for row in json.loads((self.root / 'manifest.json').read_text()):
            p = (self.root / row['exported']).resolve()
            if not p.is_relative_to(self.root.resolve()):
                raise ValueError('Manifest path outside case directory')
            if sha(p) != row['sha256']:
                raise ValueError('Export checksum mismatch: ' + row['exported'])
        self.decision = json.loads((self.root / 'verification/T0_20250725T190000_decisions.json').read_text())
        self.score = self.decision['score']
        self.cfg = json.loads((self.root / 'configuration/config.json').read_text())
        self.inputs = read_npz(self.root / 'predictions/T0_20250725T190000_original_cache.npz')
        self.p = self.inputs['probability']
        self.grid = read_npz(self.root / 'grid/verification_grid.npz')
        self.lon, self.lat = self.grid['longitude'], self.grid['latitude']
        self.dx = float(np.abs(np.diff(self.grid['x_m'])).mean() / 1000)
        self.dy = float(np.abs(np.diff(self.grid['y_m'])).mean() / 1000)
        self.radar = read_npz(self.root / 'radar/mrms_projected_T0m35_T0p60.npz')
        self.t0 = pd.Timestamp(self.score['issuance_time'], tz='UTC').value
        self.end = self.t0 + 30 * K.MINUTE
        assert self.p.shape == self.lon.shape == self.lat.shape
        assert np.isfinite(self.p).all() and ((self.p >= 0) & (self.p <= 1)).all()
        self.core = np.zeros(self.p.shape, bool)
        for birth in self.decision['births']:
            self.core[np.asarray(birth['core_y'], int), np.asarray(birth['core_x'], int)] = True
        coverage = read_npz(self.root / 'growth_reference/coverage.npz')
        times = coverage['time_ns']
        anchor = int(np.searchsorted(times, self.t0, side='right')) - 1
        after = np.flatnonzero((times > self.t0) & (times <= self.end))
        if (anchor < 0 or not len(after) or self.t0 - times[anchor] > 150e9
                or self.end - times[after[-1]] > 150e9
                or np.diff(np.r_[times[anchor], times[after], self.end]).max() > 180e9):
            raise ValueError('Incomplete original radar coverage')
        self.coverage = np.unpackbits(
            np.bitwise_and.reduce(coverage['valid_bits'][np.r_[anchor, after]], axis=0),
            count=self.p.size).reshape(self.p.shape).astype(bool)
        self.events = json.loads((self.root / 'growth_reference/events.json').read_text())
        self.candidates = K.candidates(SimpleNamespace(events=self.events), 25, self.t0,
                                      self.end, self.coverage, [0, 5, 10, 15, 20], K.growth_flags)
        self.native = None  # Lazy: the full native radar cube is needed only by the object inspector.
        self.validate_original()

    @lru_cache(maxsize=16)
    def evaluate(self, threshold=None, persistence=5, radius=15., min_area=1):
        q = float(self.score['threshold'] if threshold is None else threshold)
        if not 0 < q <= 1 or persistence not in (0, 5, 10, 15, 20) or radius <= 0 or min_area < 1:
            raise ValueError('Unsupported diagnostic controls')
        events = [e for e in self.candidates if e['birth_native_pixels'] >= min_area
                  and e['persistence_flags'][persistence] is not None]
        # The original strict distance remains 5 km. Only the growth-diagnostic distance varies.
        r = K.classify(self.p, self.core, events, self.coverage, q, 5., float(radius),
                       self.dy, self.dx, K.C, K.N)
        n = r['counts']; tp, fp, fn, g = n['tp'], n['fp'], n['fn'], n['accepted_growth']
        r['metrics'] = pd.DataFrame([
            {'Verification': 'Strict CI35 · one-to-one · 5 km', 'Hits / accepted forecasts': tp,
             'FP': fp, 'FN (strict observations)': fn, 'CSI / diagnostic CSI*': ratio(tp, tp+fp+fn),
             'POD / diagnostic POD*': ratio(tp, tp+fn), 'FAR / diagnostic FAR*': ratio(fp, tp+fp)},
            {'Verification': 'Growth diagnostic · strict FN retained', 'Hits / accepted forecasts': tp+g,
             'FP': fp-g, 'FN (strict observations)': fn, 'CSI / diagnostic CSI*': ratio(tp+g, tp+fp+fn),
             'POD / diagnostic POD*': ratio(tp+g, tp+g+fn), 'FAR / diagnostic FAR*': ratio(fp-g, tp+fp)}
        ])
        r['settings'] = dict(threshold=q, persistence=persistence, radius=radius, min_area=min_area)
        r['eligible_candidates'] = events
        return r

    def validate_original(self):
        r = self.evaluate()
        for key in ('tp', 'fp', 'fn', 'accepted_growth', 'remaining_FP', 'FN_covered', 'FN_uncovered'):
            if r['counts'][key] != self.score[key]:
                raise ValueError(f'Original case does not reproduce: {key}')
        accepted = {str(k): e['id'] for k, e in r['accepted'].items()}
        saved = {str(k): e['id'] for k, e in self.decision['accepted_candidates'].items()}
        if accepted != saved:
            raise ValueError('Original growth associations differ')
        for col, key in [('CSI / diagnostic CSI*', 'csi'), ('POD / diagnostic POD*', 'pod'),
                         ('FAR / diagnostic FAR*', 'far')]:
            assert np.isclose(r['metrics'].iloc[0][col], self.score[key], atol=1e-12, rtol=0)
        ledger = pd.read_csv(self.root / 'verification/object_ledger_case.csv')
        self.ledger = ledger
        forecasts = ledger[ledger.object_kind.eq('forecast')]
        observations = ledger[ledger.object_kind.eq('observation')]
        assert set(forecasts[forecasts.strict_status.eq('TP')].object_id.astype(int)) == r['hit_pred']
        assert set(observations[observations.strict_status.eq('TP_observation')].object_id.astype(int)) == r['hit_obs']
        self.original_validated = True

    def object_table(self, r):
        rows = []
        for k in range(1, int(r['predicted'].max())+1):
            yy, xx = np.where(r['predicted'] == k)
            rows.append(dict(object_id=k, pixels=len(yy), longitude=self.lon[yy, xx].mean(),
                latitude=self.lat[yy, xx].mean(), max_probability=float(self.p[yy, xx].max()),
                strict='TP' if k in r['hit_pred'] else 'FP',
                diagnostic='strict TP' if k in r['hit_pred'] else 'accepted growth' if k in r['accepted'] else 'unmatched',
                growth_candidate=r['accepted'].get(k, {}).get('id', '')))
        return pd.DataFrame(rows, columns=['object_id', 'pixels', 'longitude', 'latitude',
                     'max_probability', 'strict', 'diagnostic', 'growth_candidate'])

    def figure(self, r, zoom='Paper overview', show_ids=False):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout='constrained')
        radar_times = self.radar['time_ns']
        for j, minute in enumerate((8, 18, 28)):
            ix = int(np.argmin(np.abs(radar_times - (self.t0 + minute*K.MINUTE))))
            assert self.t0 < radar_times[ix] <= self.end
            ax = axes[0, j]
            im = ax.pcolormesh(self.lon, self.lat,
                np.ma.masked_invalid(self.radar['reflectivity_dbz'][ix]),
                cmap='turbo', vmin=0, vmax=60, shading='nearest', rasterized=True)
            ax.set_facecolor('#241334')
            ax.set_title(pd.Timestamp(radar_times[ix], tz='UTC').strftime('%H:%M:%S UTC'))
            # Only births preceding this scan, within six minutes, get their actual uninflated outline.
            for b in self.decision['births']:
                age = (int(radar_times[ix]) - pd.Timestamp(b['birth_time'], tz='UTC').value)/K.MINUTE
                if 0 <= age <= 6:
                    mask = np.zeros(self.p.shape, bool); mask[b['core_y'], b['core_x']] = True
                    ax.contour(self.lon, self.lat, mask.astype(float), [.5], colors='white', linewidths=.9)
        fig.colorbar(im, ax=axes[0, :], shrink=.8, label='MRMS reflectivity (dBZ)')
        im = axes[1, 0].pcolormesh(self.lon, self.lat, self.p, cmap='magma', vmin=0,
                                  vmax=max(.15, float(self.p.max())), shading='nearest', rasterized=True)
        axes[1, 0].set_title('Forecast probability · 0–30 min')
        mask = (r['predicted'] > 0).astype(float)
        if mask.any() and not mask.all():
            axes[1, 0].contour(self.lon, self.lat, mask, [.5], colors='white', linewidths=.8)
        fig.colorbar(im, ax=axes[1, 0], shrink=.8, label='Raw probability')
        for ax, diagnostic in zip(axes[1, 1:], (False, True)):
            values = r['secondary'] if diagnostic else r['strict']
            colored = np.where(values > 0, np.where(values == 3, 2, 1), np.nan)
            ax.pcolormesh(self.lon, self.lat, np.ma.masked_invalid(colored),
                          cmap=ListedColormap([GREEN, RED]), vmin=1, vmax=2, shading='nearest')
            for k in range(1, int(r['observed'].max())+1):
                yy, xx = np.where(r['observed'] == k)
                hit = k in r['hit_obs']
                ax.scatter(self.lon[yy, xx].mean(), self.lat[yy, xx].mean(), marker='o' if hit else 'x',
                           s=34 if hit else 45, c=BLUE, linewidths=1.5,
                           **({'edgecolors': 'white'} if hit else {}), zorder=5)
            if diagnostic:
                for e in r['accepted'].values():
                    yy, xx = np.unravel_index(np.asarray(e['flat'], int), self.p.shape)
                    ax.scatter(self.lon[yy, xx].mean(), self.lat[yy, xx].mean(), c=ORANGE,
                               marker='o', s=30, edgecolors='white', linewidths=.6, zorder=6)
            metric = r['metrics'].iloc[int(diagnostic)]
            vals = [metric[k] for k in ('CSI / diagnostic CSI*', 'POD / diagnostic POD*', 'FAR / diagnostic FAR*')]
            ax.text(.98, .98, '\n'.join(f'{k}{"*" if diagnostic else ""} {v:.3f}'
                    for k, v in zip(('CSI', 'POD', 'FAR'), vals)), transform=ax.transAxes,
                    ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', edgecolor='.8', alpha=.9))
            ax.set_title('Growth diagnostic' if diagnostic else 'Strict CI35 · one-to-one')
            if show_ids:
                for row in self.object_table(r).itertuples():
                    ax.text(row.longitude, row.latitude, str(row.object_id), fontsize=8,
                            ha='center', bbox=dict(fc='white', ec='none', alpha=.8), zorder=7, clip_on=True)
        if zoom == 'Paper overview':
            extent = (-76.55, -72.35, 40.05, 43.25)
        elif zoom == 'Source-config zoom':
            extent = self.cfg['zoom']
        else:
            extent = (np.nanmin(self.lon), np.nanmax(self.lon), np.nanmin(self.lat), np.nanmax(self.lat))
        for j, ax in enumerate(axes.ravel()):
            ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.set_aspect(1/np.cos(np.deg2rad(42)))
            ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
            ax.text(-.09, 1.04, 'abcdef'[j], transform=ax.transAxes, fontweight='bold', fontsize=13)
            ax.grid(alpha=.12)
        handles = [Patch(color=GREEN, label='Accepted forecast'), Patch(color=RED, label='Unmatched forecast'),
            Line2D([], [], marker='o', color=BLUE, ls='', label='Matched CI35'),
            Line2D([], [], marker='x', color=BLUE, ls='', label='Strict CI35 FN'),
            Line2D([], [], marker='o', color=ORANGE, ls='', label='Accepted ≥25-dBZ growth')]
        axes[1, 1].legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.22),
                          ncol=3, frameon=False, fontsize=9)
        return fig

    def inspect(self, r, object_id):
        import matplotlib.pyplot as plt
        k = int(object_id); mask = r['predicted'] == k
        if not mask.any():
            raise ValueError('No such forecast object at this probability threshold')
        yy, xx = np.where(mask)
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), layout='constrained')
        # Inspect probability locally without restricting any verification calculation.
        extent = (max(0, xx.min()-12), min(self.p.shape[1], xx.max()+13),
                  max(0, yy.min()-12), min(self.p.shape[0], yy.max()+13))
        axes[0].imshow(self.p, cmap='magma', origin='upper', vmin=0, vmax=max(.15, self.p.max()))
        axes[0].contour(mask, [.5], colors='white', linewidths=1)
        axes[0].set_xlim(extent[:2]); axes[0].set_ylim(extent[3], extent[2])
        axes[0].set_title(f'Forecast object {k} · actual grid pixels')
        axes[0].set_xlabel('Grid column'); axes[0].set_ylabel('Grid row')
        field = self.radar['reflectivity_dbz'][:, mask]
        valid = self.radar['valid_mask'][:, mask]
        peak = np.max(np.where(valid & np.isfinite(field), field, -np.inf), axis=1)
        peak[~np.isfinite(peak)] = np.nan
        minutes = (self.radar['time_ns']-self.t0)/K.MINUTE
        axes[1].plot(minutes, peak, color=BLUE, label='Max projected dBZ inside fixed forecast footprint')
        axes[1].axvspan(0, 30, color=BLUE, alpha=.07)
        for level in (25, 35):
            axes[1].axhline(level, ls=':', color='.5', lw=.8)
        event = r['accepted'].get(k)
        if event:
            track = next(e for e in self.events if f"Z25_{e['track_id']}" == event['id'])
            profile = track['profile']
            axes[1].plot([(v['time_ns']-self.t0)/K.MINUTE for v in profile],
                         [v['native_max_dbz'] for v in profile], color=ORANGE, marker='.',
                         label='Associated native-radar lineage maximum')
        axes[1].set_xlabel('Minutes from initialization'); axes[1].set_ylabel('Reflectivity (dBZ)')
        axes[1].set_title('Radar evolution · context, not a new CI35 definition')
        axes[1].legend(fontsize=8, loc='upper left'); axes[1].grid(alpha=.2)
        return fig


def launch(case):
    import ipywidgets as w
    import matplotlib.pyplot as plt
    from IPython.display import display, HTML, clear_output
    style = {'description_width': 'initial'}
    q = w.FloatSlider(value=float(case.score['threshold']), min=.005, max=.25, step=.005,
                      description='Raw probability threshold', readout_format='.3f', continuous_update=False, style=style)
    persistence = w.Dropdown(options=[('Original figure · 5 min', 5), ('10 min', 10), ('15 min', 15),
        ('20 min', 20), ('Base growth rule only', 0)], value=5, description='≥25-dBZ persistence', style=style)
    radius = w.Dropdown(options=[5., 10., 15., 20., 30.], value=15., description='Growth radius (km)', style=style)
    area = w.Dropdown(options=[1, 2, 3, 5], value=1, description='Min native birth pixels', style=style)
    zoom = w.Dropdown(options=['Paper overview', 'Full domain', 'Source-config zoom'], description='Display extent', style=style)
    ids = w.Checkbox(value=False, description='Show forecast IDs')
    reset = w.Button(description='Restore original figure settings', icon='undo', layout=w.Layout(width='250px'))
    output, feedback = w.Output(), w.Output()
    export = w.Button(description='Export current tables + settings', icon='download', layout=w.Layout(width='250px'))
    state = {}
    def update(change=None):
        with output:
            clear_output(wait=True)
            r = case.evaluate(float(q.value), int(persistence.value), float(radius.value), int(area.value)); state['r'] = r
            display(HTML('<b>25 July 2025 · 19:00 UTC · 0–30 min</b><br>'
                'All counts use the full 212 × 230 domain before display zoom. Strict CI35 stays at 5 km; '
                'growth diagnostics preserve strict FN and do not recover matching-competition FN.'))
            display(r['metrics'].round(6))
            n = r['counts']
            display(HTML(f"<b>{n['predicted_objects']} forecasts · {n['observed_objects']} CI35 reference objects · "
                         f"{n['accepted_growth']} growth promotions · {n['FN_covered']} spatially covered strict FN.</b>"))
            fig = case.figure(r, zoom.value, ids.value); plt.show(); plt.close(fig)
    def restore(_):
        # Suspend listeners to avoid evaluating five intermediate states.
        for control in controls: control.unobserve(update, names='value')
        q.value = float(case.score['threshold']); persistence.value = 5; radius.value = 15.; area.value = 1
        zoom.value = 'Paper overview'; ids.value = False
        for control in controls: control.observe(update, names='value')
        update()
    def save(_):
        from datetime import datetime, timezone
        out = Path('outputs') / ('case_exploration_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        out.mkdir(parents=True)
        r = state['r']; r['metrics'].to_csv(out/'metrics.csv', index=False)
        case.object_table(r).to_csv(out/'forecast_objects.csv', index=False)
        (out/'settings.json').write_text(json.dumps(dict(**r['settings'], split=case.score['split'],
            window='(0,30]', strict_km=5, min_forecast_pixels=2, matching_domain='full',
            note='Exploratory Fig11 growth diagnostic; original strict FN retained; not S12 or FN recovery'), indent=2))
        with feedback:
            clear_output(wait=True)
            print('Saved:', out.resolve())
    controls = [q, persistence, radius, area, zoom, ids]
    for control in controls: control.observe(update, names='value')
    reset.on_click(restore); export.on_click(save)
    display(w.VBox([w.HTML('<h3>Explore the object-based verification</h3>'),
        w.HTML('The ≥25-dBZ growth rule also requires temporal novelty, ≥3 contiguous scans, '
               'growth in intensity or native area, and topology/coverage checks. The controls do not replace those rules.'),
        q, w.HBox([persistence, radius]), w.HBox([area, zoom]), w.HBox([ids, reset]), output,
        export, feedback]))
    update()
    return state


def radar_player(case):
    """Browse all exported radar scans; no verification data are changed."""
    import ipywidgets as w
    import matplotlib.pyplot as plt
    from IPython.display import display, clear_output
    stamps = case.radar['time_ns']
    indices = np.flatnonzero((stamps > case.t0) & (stamps <= case.end))
    scan = w.IntSlider(min=0, max=len(indices)-1, value=0, description='Future scan', continuous_update=False)
    play = w.Play(min=0, max=len(indices)-1, interval=700)
    w.jslink((play, 'value'), (scan, 'value'))
    contours = w.Checkbox(value=True, description='Show frozen forecast outlines (p = 0.05)')
    output = w.Output()
    def draw(change=None):
        with output:
            clear_output(wait=True)
            i = indices[scan.value]
            fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
            im = ax.pcolormesh(case.lon, case.lat, np.ma.masked_invalid(case.radar['reflectivity_dbz'][i]),
                shading='nearest', cmap='turbo', vmin=0, vmax=60, rasterized=True)
            ax.set_facecolor('#241334')
            if contours.value:
                ax.contour(case.lon, case.lat, (case.evaluate()['predicted'] > 0).astype(float),
                           [.5], colors='white', linewidths=.7)
            ax.set_xlim(-76.55, -72.35); ax.set_ylim(40.05, 43.25)
            ax.set_aspect(1/np.cos(np.deg2rad(42)))
            ax.set_title(pd.Timestamp(stamps[i], tz='UTC').strftime('%d July %Y · %H:%M:%S UTC'))
            ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
            fig.colorbar(im, ax=ax, label='MRMS reflectivity (dBZ)')
            plt.show(); plt.close(fig)
    scan.observe(draw, names='value'); contours.observe(draw, names='value')
    display(w.VBox([w.HBox([play, scan]), contours, output])); draw()
