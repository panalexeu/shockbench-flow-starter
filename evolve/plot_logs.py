"""Plot the average reward per iteration of every search run on one chart.

    uv run python -m evolve.plot_logs
    uv run python -m evolve.plot_logs --policies_dir=evolve/policies --out=evolve/outputs/plot_logs/mine.png

One line per run folder, named after it, from its oldest log (logs of a continued run are ignored). The reward of
an iteration is the mean reward of its proposed policies, failed ones (-999) excluded; an iteration where every
policy failed leaves a gap.
"""

import datetime
import json
import os
import time
from pathlib import Path

import fire
import matplotlib
import numpy as np


matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


FAIL = -999
# categorical slots of the dataviz default palette plus three more hues; the ten pass the palette validator
# (lightness, chroma, colorblind and normal-vision separation)
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#7a5c00', '#0091a0', '#a347d4']
STYLES = ['-', '--', ':']  # past 10 lines the colors repeat with another dash, so every line stays distinct
ALPHA = 0.6  # lines overlap: see-through lines keep the ones underneath visible
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'


def _log_time(path: Path) -> float:
    # log_<dd-mm-YYYY_HH-MM-SS>.json; anything else (an older log.json) by its modification time
    try:
        return datetime.datetime.strptime(path.stem.removeprefix('log_'), '%d-%m-%Y_%H-%M-%S').timestamp()
    except ValueError:
        return path.stat().st_mtime


def load_runs(policies_dir: str) -> list[dict]:
    """One entry per run folder with a log: its label, x (iteration) and y (mean reward, NaN if all failed)."""
    runs = []
    for folder in sorted(p for p in Path(policies_dir).iterdir() if p.is_dir()):
        logs = sorted(folder.glob('log*.json'), key=_log_time)
        if not logs:
            continue
        iterations = json.loads(logs[0].read_text())['iterations']  # the oldest log only: continuations ignored
        y = []
        for it in iterations:
            ok = [p['reward'] for p in it['proposed'] if p['reward'] > FAIL]
            y.append(np.mean(ok) if ok else np.nan)
        runs.append({'label': folder.name, 'x': np.arange(len(y)), 'y': np.array(y)})
    return runs


def plot(runs: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor='white')
    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate('naive rule (0)', xy=(1, 0), xycoords=('axes fraction', 'data'), xytext=(-4, 4),
                textcoords='offset points', ha='right', va='bottom', color=MUTED, fontsize=9)
    for i, r in enumerate(runs):
        color, style = COLORS[i % len(COLORS)], STYLES[(i // len(COLORS)) % len(STYLES)]
        ax.plot(r['x'], r['y'], color=color, ls=style, lw=2, marker='o', ms=4, alpha=ALPHA, label=r['label'],
                zorder=3)

    ax.set_xlabel('iteration', color=INK)
    ax.set_ylabel('mean reward of proposed policies (failures excluded)', color=INK)
    ax.set_title('Search progress per run', color=INK, loc='left', fontsize=13)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)

    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9, labelcolor=INK)
    for handle in legend.legend_handles:
        handle.set_alpha(1)  # full color in the legend, so each name's color is easy to match
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(policies_dir: str = 'evolve/policies', out: str | None = None) -> None:
    runs = load_runs(policies_dir)
    if not runs:
        raise SystemExit(f'no log*.json files under {policies_dir}')
    out = Path(out or f"evolve/outputs/plot_logs/{time.strftime('%Y-%m-%d_%H-%M-%S')}/rewards.png")
    plot(runs, out)
    # the same numbers as a table, for reading exact values
    width = max(len(r['label']) for r in runs)
    print(f"{'run':<{width}}  iters  best mean  last mean")
    for r in runs:
        valid = r['y'][~np.isnan(r['y'])]
        best, last = (valid.max(), valid[-1]) if len(valid) else (float('nan'), float('nan'))
        print(f"{r['label']:<{width}}  {len(r['x']):>5}  {best:>9.3f}  {last:>9.3f}")
    print(f'written {os.path.relpath(out)}')


if __name__ == '__main__':
    fire.Fire(main)
