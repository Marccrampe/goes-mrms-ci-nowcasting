from pathlib import Path
import os
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def padded_extent(lon_min, lon_max, lat_min, lat_max,
                  pad_frac=0.22, min_lon_pad=0.30, min_lat_pad=0.20):
    dx = max(lon_max - lon_min, 1e-6)
    dy = max(lat_max - lat_min, 1e-6)
    pad_x = max(min_lon_pad, dx * pad_frac)
    pad_y = max(min_lat_pad, dy * pad_frac)
    return lon_min - pad_x, lon_max + pad_x, lat_min - pad_y, lat_max + pad_y

from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap, BoundaryNorm, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


# ============================================================
# PAPER COLORS
# ============================================================

GREEN = "#009E73"   # TP / accepted forecast
RED   = "#D62728"   # FP / unmatched forecast
BLUE  = "#0072B2"   # FN / missed observed CI
ORANGE = "#D55E00"  # 25-dBZ radar contour
DARK = "#222222"

# Wide case-study domain.
DISPLAY_LON_WIDTH = 5.4
DISPLAY_LAT_HEIGHT = 4.2


def safe_score(num, den):
    return num / den if den else np.nan


def page(folder, name, d):

    r = d["result"]
    g = d["settings"]

    lon = np.asarray(d["lon"])
    lat = np.asarray(d["lat"])

    t = d["t"]
    counts = r["counts"]

    # ========================================================
    # LARGER DISPLAY DOMAIN
    # ========================================================

    # --------------------------------------------------------
    # Object-focused case-study domain.
    #
    # Centre is determined objectively from the median of the
    # observed/predicted object centroids. This removes remote
    # convection while keeping a broad regional context.
    # --------------------------------------------------------

    observed_for_zoom = np.asarray(r["observed"])
    predicted_for_zoom = np.asarray(r["predicted"])

    object_centers = []

    for labels in [observed_for_zoom, predicted_for_zoom]:

        ids = np.unique(labels)
        ids = ids[ids > 0]

        for object_id in ids:

            mask = labels == object_id

            if not np.any(mask):
                continue

            object_centers.append(
                (
                    float(np.nanmean(lon[mask])),
                    float(np.nanmean(lat[mask])),
                )
            )

    finite_geo = np.isfinite(lon) & np.isfinite(lat)

    domain_lon_min = float(np.nanmin(lon[finite_geo]))
    domain_lon_max = float(np.nanmax(lon[finite_geo]))
    domain_lat_min = float(np.nanmin(lat[finite_geo]))
    domain_lat_max = float(np.nanmax(lat[finite_geo]))

    if object_centers:

        object_centers = np.asarray(object_centers)

        cx = float(np.median(object_centers[:, 0]))
        cy = float(np.median(object_centers[:, 1]))

    else:

        cx = 0.5 * (
            domain_lon_min + domain_lon_max
        )

        cy = 0.5 * (
            domain_lat_min + domain_lat_max
        )

    # Intermediate zoom:
    # large enough for regional context, small enough to read
    # isolated CI objects clearly.
    DISPLAY_LON_WIDTH = 4.2
    DISPLAY_LAT_HEIGHT = 3.2

    xmin = max(
        domain_lon_min,
        cx - DISPLAY_LON_WIDTH / 2,
    )

    xmax = min(
        domain_lon_max,
        cx + DISPLAY_LON_WIDTH / 2,
    )

    ymin = max(
        domain_lat_min,
        cy - DISPLAY_LAT_HEIGHT / 2,
    )

    ymax = min(
        domain_lat_max,
        cy + DISPLAY_LAT_HEIGHT / 2,
    )

    plot_roi = (
        np.isfinite(lon) &
        np.isfinite(lat) &
        (lon >= xmin) & (lon <= xmax) &
        (lat >= ymin) & (lat <= ymax)
    )

    yy, xx = np.where(plot_roi)

    if len(yy) == 0:
        raise RuntimeError("Expanded plotting domain contains no grid cells")

    ys = slice(yy.min(), yy.max() + 1)
    xs = slice(xx.min(), xx.max() + 1)

    X = lon[ys, xs]
    Y = lat[ys, xs]

    # ========================================================
    # FONT / STYLE
    # ========================================================

    font_path = os.environ.get("CI_PAPER_FONT", "")

    if font_path and Path(font_path).is_file():
        fm.fontManager.addfont(font_path)
        family = fm.FontProperties(fname=font_path).get_name()
        for sibling in Path(font_path).parent.glob('*.ttf'):
            fm.fontManager.addfont(str(sibling))
    else:
        family = "DejaVu Sans"

    plt.rcParams.update({
        "font.family": family,
        "font.size": 6.8,
        "axes.titlesize": 7.4,
        "axes.labelsize": 6.8,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    # ========================================================
    # LAYOUT
    #
    # 12-column grid:
    # top:    4 equal panels (3 cols each)
    # bottom: 3 equal panels (4 cols each)
    # ========================================================

    # Publication layout:
    # 183 mm wide = full double-column figure.
    # Six large panels only.
    # Full-width Scientific Reports / Nature-style figure.
    # 183 mm wide, comfortably below maximum page height.
    # Full-width publication figure: 183 mm.
    # Tight 2x3 grid so the maps use as much area as possible.
    fig = plt.figure(
        figsize=(183 / 25.4, 132 / 25.4),
        facecolor="white",
    )

    gs = fig.add_gridspec(
        2,
        6,
        left=0.075,
        right=0.895,
        bottom=0.13,
        top=0.90,
        wspace=0.18,
        hspace=0.26,
    )

    top_axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
    ]

    bottom_axes = [
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
    ]

    # ========================================================
    # HELPERS
    # ========================================================

    def base(ax, title, show_ylabel=True):
        match=re.match(r'^\(([a-f])\)\s*(.*)$',title)
        if match:
            ax.text(0,1.035,match.group(1),transform=ax.transAxes,
                fontsize=8,fontweight='bold',ha='left',va='bottom',clip_on=False)
            ax.annotate(match.group(2),xy=(0,1.035),xycoords='axes fraction',xytext=(11,0),
                textcoords='offset points',fontsize=7,ha='left',va='bottom',annotation_clip=False)
        else:ax.set_title(title,loc='left',fontsize=7)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        mean_lat = np.nanmean(Y)
        ax.set_aspect(
            1 / np.cos(np.deg2rad(mean_lat))
        )

        ax.grid(
            alpha=0.08,
            lw=0.45,
        )

        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))

        ax.xaxis.set_major_formatter(
            FormatStrFormatter("%.1f")
        )
        ax.yaxis.set_major_formatter(
            FormatStrFormatter("%.1f")
        )

        ax.set_xlabel("Longitude")

        if show_ylabel:
            ax.set_ylabel("Latitude")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    def field_plot(
        ax,
        field,
        cmap,
        vmin=None,
        vmax=None,
        norm=None,
    ):

        a = np.asarray(field)[ys, xs]

        return ax.pcolormesh(
            X,
            Y,
            np.ma.masked_invalid(a),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            shading="nearest",
            rasterized=True,
        )

    def contour(
        ax,
        mask,
        color,
        lw=0.8,
        ls="-",
        alpha=1.0,
    ):

        a = np.asarray(mask)[ys, xs]

        if not np.any(a):
            return

        if np.all(a):
            return

        ax.contour(
            X,
            Y,
            a.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=lw,
            linestyles=ls,
            alpha=alpha,
        )

    # ========================================================
    # TOP ROW — MRMS AROUND FORECAST INITIALIZATION
    #
    # White contours are NOT future targets projected backward
    # onto T0.  Each CI35 core is shown only on the first displayed
    # radar scan occurring after its actual catalogue birth time.
    # The core is dilated slightly for visual readability.
    # ========================================================

    from scipy.ndimage import binary_dilation

    radar_fields = d['display_radar_fields']
    radar_times = list(map(int,d['display_radar_times']))
    offsets = [(stamp-int(t.value))/60e9 for stamp in radar_times]

    # One contour mask per displayed radar frame.
    birth_masks = [
        np.zeros_like(
            np.asarray(r["observed"]),
            dtype=bool
        )
        for _ in radar_times
    ]

    # Assign each birth to the FIRST displayed radar scan
    # occurring after the actual birth.
    #
    # Max 6-min delay avoids drawing a birth long after it
    # appeared and prevents future births from appearing at T0.
    MAX_DISPLAY_LAG_NS = int(
        6 * 60 * 1e9
    )

    for event in d.get("births",[]):

        try:
            bt = pd.Timestamp(
                event["birth_time"]
            )

            if bt.tzinfo is None:
                bt = bt.tz_localize("UTC")
            else:
                bt = bt.tz_convert("UTC")

            birth_ns = int(bt.value)

            after = [
                j
                for j,scan_ns
                in enumerate(radar_times)
                if scan_ns >= birth_ns
            ]

            if not after:
                continue

            j = after[0]

            if radar_times[j] - birth_ns > MAX_DISPLAY_LAG_NS:
                continue

            yy=np.asarray(
                event.get("core_y",[]),
                dtype=int
            )

            xx=np.asarray(
                event.get("core_x",[]),
                dtype=int
            )

            if len(yy)==0:
                continue

            m=np.zeros_like(
                birth_masks[j]
            )

            good=(
                (yy>=0) &
                (yy<m.shape[0]) &
                (xx>=0) &
                (xx<m.shape[1])
            )

            m[
                yy[good],
                xx[good]
            ]=True

            # Larger outline only for display.
            m=binary_dilation(
                m,
                iterations=2
            )

            birth_masks[j] |= m

        except Exception:
            continue

    radar_im=None

    for j,(ax,field,stamp_ns,offset) in enumerate(
        zip(
            top_axes,
            radar_fields,
            radar_times,
            offsets,
        )
    ):

        radar_im=field_plot(
            ax,
            field,
            "turbo",
            vmin=0,
            vmax=60,
        )

        # Halo + white contour.
        #
        # IMPORTANT:
        # this is drawn only when the birth has actually occurred
        # by this displayed radar scan.
        if np.any(birth_masks[j]):

            contour(
                ax,
                birth_masks[j],
                "black",
                lw=2.6,
                alpha=0.85,
            )

            contour(
                ax,
                birth_masks[j],
                "white",
                lw=1.45,
                alpha=1.0,
            )

        stamp=pd.Timestamp(
            stamp_ns,
            tz="UTC"
        )

        if offset<0:
            label=f"T{offset} min"
        elif offset==0:
            label="T0"
        else:
            label=f"T+{offset} min"

        base(
            ax,
            f"({chr(97+j)}) {stamp.strftime('%H:%M:%S UTC')}",
            show_ylabel=(j==0),
        )

    # Top row does not need repeated x-axis labels.
    # Removing them creates clear separation from the lower titles.
    for ax in top_axes:
        ax.set_xlabel("")

    # ========================================================
    # BOTTOM LEFT — RAW MODEL PROBABILITY
    # ========================================================

    ax = bottom_axes[0]

    prob = np.asarray(r["probability"])

    prob_im = field_plot(
        ax,
        prob,
        "magma",
        norm=PowerNorm(
            gamma=0.35,
            vmin=0,
            vmax=1,
        ),
    )

    # Only one forecast contour here:
    # the operational/frozen threshold.
    threshold_mask = prob >= float(r["threshold"])

    contour(
        ax,
        threshold_mask,
        "white",
        lw=1.05,
    )

    base(
        ax,
        "(d) Probability",
        show_ylabel=True,
    )

    # ========================================================
    # STRICT / RELAXED BINARY FORECAST VISUALIZATION
    #
    # Forecast objects:
    #   GREEN = accepted forecast
    #   RED   = unmatched / false-positive forecast
    #
    # Targets are POINTS, not filled masks:
    #   BLUE dot = observed strict CI35 birth
    #   BLUE X   = missed strict CI35 birth
    #   ORANGE dot = eligible diagnostic CI25 onset associated with a
    #              strict FP (promotion candidate)
    #
    # Thus a red object in strict can turn green in relaxed.
    # ========================================================

    observed = np.asarray(r["observed"])
    predicted = np.asarray(r["predicted"])

    def mask_center(mask):
        if not np.any(mask):
            return None
        return (
            float(np.nanmean(lon[mask])),
            float(np.nanmean(lat[mask])),
        )

    def candidate_center(event):
        flat = np.asarray(event["flat"], dtype=int)
        if flat.size == 0:
            return None

        cy, cx = np.unravel_index(
            flat,
            lon.shape,
        )

        return (
            float(np.nanmean(lon[cy, cx])),
            float(np.nanmean(lat[cy, cx])),
        )

    def add_target_points(ax):

        covered_obs=set(
            int(x)
            for x in r.get(
                "covered_obs",
                set()
            )
        )

        hit_obs=set(
            int(x)
            for x in r.get(
                "hit_obs",
                set()
            )
        )

        # Strict canonical CI35 births.
        for obs_id in range(
            1,
            int(observed.max())+1
        ):

            pos=mask_center(
                observed==obs_id
            )

            if pos is None:
                continue

            x,y=pos

            if obs_id in covered_obs:

                # Either a formal one-to-one TP or an observation
                # already spatially covered by a forecast object.
                #
                # Do NOT show a blue miss X.
                ax.scatter(
                    x,
                    y,
                    marker="o",
                    s=24,
                    c=BLUE,
                    edgecolors="white",
                    linewidths=0.55,
                    zorder=15,
                )

            else:

                # Genuine uncovered FN.
                ax.scatter(
                    x,
                    y,
                    marker="x",
                    s=24,
                    c=BLUE,
                    linewidths=1.45,
                    zorder=16,
                )

        # Persistent CI25 growth candidates associated with
        # strict false-positive forecast objects.
        for pred_id,event in r["accepted"].items():

            pos=candidate_center(
                event
            )

            if pos is None:
                continue

            x,y=pos

            ax.scatter(
                x,
                y,
                marker="o",
                s=24,
                c=ORANGE,
                edgecolors="white",
                linewidths=0.55,
                zorder=17,
            )

    # --------------------------------------------------------
    # STRICT CI35 PANEL
    # --------------------------------------------------------

    ax = bottom_axes[1]

    strict = np.asarray(r["strict"])

    strict_forecast = np.zeros(
        strict.shape,
        dtype=np.uint8,
    )

    # strict code:
    # 1 = strict TP
    # 3 = strict FP
    strict_forecast[strict == 1] = 1
    strict_forecast[strict >= 2] = 2

    strict_display = np.where(
        strict_forecast > 0,
        strict_forecast,
        np.nan,
    )

    forecast_cmap = ListedColormap([
        GREEN,
        RED,
    ])

    forecast_norm = BoundaryNorm(
        [0.5, 1.5, 2.5],
        2,
    )

    field_plot(
        ax,
        strict_display,
        forecast_cmap,
        norm=forecast_norm,
    )

    add_target_points(ax)

    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])

    fn_covered = int(
        counts.get(
            "FN_covered",
            len(
                set(r.get("covered_obs",set()))
                -
                set(r.get("hit_obs",set()))
            )
        )
    )

    fn_uncovered = int(
        counts.get(
            "FN_uncovered",
            max(fn-fn_covered,0)
        )
    )

    strict_csi = safe_score(
        tp,
        tp + fp + fn,
    )

    strict_pod = safe_score(
        tp,
        tp + fn,
    )

    strict_far = safe_score(
        fp,
        tp + fp,
    )

    base(
        ax,
        "(e) Strict",
        show_ylabel=False,
    )

    ax.text(
        0.975,
        0.975,
        f"CSI {strict_csi:.3f}\n"
        f"POD {strict_pod:.3f}\n"
        f"FAR {strict_far:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        bbox=dict(
            facecolor="white",
            edgecolor="0.75",
            linewidth=0.55,
            alpha=0.87,
            pad=1.2,
        ),
        zorder=30,
    )

    # --------------------------------------------------------
    # RELAXED CI25 PANEL
    # --------------------------------------------------------

    ax = bottom_axes[2]

    secondary = np.asarray(
        r["secondary"]
    )

    relaxed_forecast = np.zeros(
        secondary.shape,
        dtype=np.uint8,
    )

    # 1 = original strict TP
    # 2 = strict FP promoted by persistent CI25 growth
    # both become GREEN here.
    relaxed_forecast[
        (secondary == 1) |
        (secondary == 2)
    ] = 1

    # Remaining FP stays red.
    relaxed_forecast[
        secondary >= 3
    ] = 2

    relaxed_display = np.where(
        relaxed_forecast > 0,
        relaxed_forecast,
        np.nan,
    )

    field_plot(
        ax,
        relaxed_display,
        forecast_cmap,
        norm=forecast_norm,
    )

    add_target_points(ax)

    promoted = int(
        counts["accepted_growth"]
    )

    tp_star = tp + promoted
    fp_star = int(
        counts["remaining_FP"]
    )
    fn_star = fn

    csi_star = safe_score(
        tp_star,
        tp_star + fp_star + fn_star,
    )

    pod_star = safe_score(
        tp_star,
        tp_star + fn_star,
    )

    far_star = safe_score(
        fp_star,
        tp_star + fp_star,
    )

    base(
        ax,
        "(f) Relaxed",
        show_ylabel=False,
    )

    ax.text(
        0.975,
        0.975,
        f"CSI* {csi_star:.3f}\n"
        f"POD* {pod_star:.3f}\n"
        f"FAR* {far_star:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        bbox=dict(
            facecolor="white",
            edgecolor="0.75",
            linewidth=0.55,
            alpha=0.87,
            pad=1.2,
        ),
        zorder=30,
    )

    # ========================================================
    # COLORBARS
    # ========================================================

    cax_radar = fig.add_axes([
        0.912,
        0.590,
        0.010,
        0.245,
    ])

    cb = fig.colorbar(
        radar_im,
        cax=cax_radar,
    )

    cb.set_label(
        "MRMS reflectivity (dBZ)",
        fontsize=6.2,
    )

    cb.ax.tick_params(
        labelsize=6.0
    )

    cax_prob = fig.add_axes([
        0.912,
        0.175,
        0.010,
        0.225,
    ])

    cb = fig.colorbar(
        prob_im,
        cax=cax_prob,
        ticks=[
            0,
            0.01,
            0.05,
            0.10,
            0.50,
            1.00,
        ],
    )

    cb.set_label(
        "Forecast probability",
        fontsize=6.2,
    )

    cb.ax.tick_params(
        labelsize=6.0
    )

    # ========================================================
    # HEADER
    # ========================================================

    split = str(
        d["split"]
    ).upper()



    # ========================================================
    # COMPACT COMMON LEGEND
    # ========================================================

    handles = [
        Line2D(
            [], [],
            color=GREEN,
            lw=4.0,
            label="Accepted",
        ),
        Line2D(
            [], [],
            color=RED,
            lw=4.0,
            label="Unmatched",
        ),
        Line2D(
            [], [],
            marker="o",
            markerfacecolor=BLUE,
            markeredgecolor="white",
            color="none",
            markersize=4.5,
            label="CI35",
        ),
        Line2D(
            [], [],
            marker="x",
            color=BLUE,
            ls="",
            markersize=5.5,
            label="Uncovered miss",
        ),
        Line2D(
            [], [],
            marker="o",
            color="none",
            markerfacecolor=ORANGE,
            markeredgecolor="white",
            ls="",
            markersize=4.5,
            label="CI25",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.49, 0.035),
        ncol=5,
        frameon=False,
        fontsize=6.5,
        columnspacing=1.05,
        handlelength=1.55,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    # ========================================================
    # EXPORT
    # ========================================================

    stem = folder / (
        name + "_Fig11_future_radar"
    )

    for ext in [
        "png",
        "pdf",
        "svg",
    ]:

        fig.savefig(
            stem.with_suffix("." + ext),
            dpi=600 if ext == "png" else None,
            facecolor="white",
        )

    plt.close(fig)

    return (
        folder.name
        + "/"
        + stem.with_suffix(".png").name
    )
