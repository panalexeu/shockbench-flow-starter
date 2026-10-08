"""Plot every search run in one image: mean reward per iteration (left) and model cost (right).

    uv run python -m evolve.plot_logs
    uv run python -m evolve.plot_logs --policies_dir=evolve/policies --out=evolve/outputs/plot_logs/mine.png

One line per run folder, named after it, from its oldest log (logs of a continued run are ignored). The reward of
an iteration is the mean reward of its proposed policies, failed ones (-999) excluded; an iteration where every
policy failed leaves a gap. The cost of a run is summed over all its logs; a run has the same color in both
panels.
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
HATCHES = ['', '//', '..']  # the bars' counterpart of STYLES
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


CLAUDE_NAMES = ('claude', 'haiku', 'sonnet', 'opus', 'fable')
CACHE_NOTE = ('Note: Claude costs are slightly inflated. A misplaced cache marker wrote each iteration\'s sampled '
              'policies to the prompt cache\n(billed at 1.25x input) without ever reading them back; fixed since.')


def load_costs(policies_dir: str) -> list[dict]:
    """Total cost of every run folder, summed over all its logs (continuations included: they were paid for)."""
    costs = []
    for folder in sorted(p for p in Path(policies_dir).iterdir() if p.is_dir()):
        logs = sorted(folder.glob('log*.json'), key=_log_time)
        if not logs:
            continue
        iterations = [it for path in logs for it in json.loads(path.read_text())['iterations']]
        total = sum(it['usage']['cost'] for it in iterations if 'usage' in it)
        provider = 'Claude' if any(n in folder.name.lower() for n in CLAUDE_NAMES) else 'OpenAI'
        costs.append({'label': folder.name, 'cost': total, 'iters': len(iterations), 'provider': provider,
                      'missing': sum('usage' not in it for it in iterations)})
    return costs


def _style(ax) -> None:
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_axisbelow(True)


def plot(runs: list[dict], costs: list[dict], out: Path) -> None:
    """Rewards per iteration (left) and cost per run (right) in one image; a run has the same color in both."""
    look = {r['label']: (COLORS[i % len(COLORS)], STYLES[(i // len(COLORS)) % len(STYLES)], HATCHES[(i // len(COLORS)) % len(HATCHES)])
            for i, r in enumerate(runs)}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(19, 7.5), facecolor='white', gridspec_kw={'width_ratios': [1.55, 1]})

    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate('naive rule (0)', xy=(1, 0), xycoords=('axes fraction', 'data'), xytext=(-4, 4),
                textcoords='offset points', ha='right', va='bottom', color=MUTED, fontsize=9)
    for r in runs:
        color, style, _ = look[r['label']]
        ax.plot(r['x'], r['y'], color=color, ls=style, lw=2, marker='o', ms=4, alpha=ALPHA, zorder=3)
    ax.set_xlabel('iteration', color=INK)
    ax.set_ylabel('mean reward of proposed policies (failures excluded)', color=INK)
    ax.set_title('Search progress per run (run names and colors: right)', color=INK, loc='left', fontsize=13)
    ax.grid(True, color=GRID, lw=0.8)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    _style(ax)

    costs = sorted(costs, key=lambda c: c['cost'])  # largest at the top
    y = np.arange(len(costs))
    for yi, c in zip(y, costs):
        color, _, hatch = look.get(c['label'], (MUTED, '-', ''))
        bx.barh(yi, c['cost'], height=0.6, color=color, alpha=ALPHA, hatch=hatch, edgecolor='white')
        bx.annotate(f"${c['cost']:.2f}  ({c['iters']} iters)", xy=(c['cost'], yi), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=9, color=INK)
    bx.set_yticks(y, [c['label'] for c in costs], fontsize=9, color=INK)
    bx.set_xlabel('total cost of the run (USD)', color=INK)
    bx.set_title('Model cost per run', color=INK, loc='left', fontsize=13)
    bx.xaxis.grid(True, color=GRID, lw=0.8)
    bx.set_xlim(0, max(c['cost'] for c in costs) * 1.3)  # room for the value labels
    _style(bx)

    fig.text(0.01, 0.01, CACHE_NOTE, fontsize=8.5, color=MUTED, ha='left', va='bottom')
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(policies_dir: str = 'evolve/policies', out: str | None = None) -> None:
    runs, costs = load_runs(policies_dir), load_costs(policies_dir)
    if not runs:
        raise SystemExit(f'no log*.json files under {policies_dir}')
    out = Path(out or f"evolve/outputs/plot_logs/{time.strftime('%Y-%m-%d_%H-%M-%S')}/search.png")
    plot(runs, costs, out)
    # the same numbers as a table, for reading exact values
    cost_of = {c['label']: c for c in costs}
    width = max(len(r['label']) for r in runs)
    print(f"{'run':<{width}}  iters  best mean  last mean  provider   cost USD")
    for r in runs:
        valid = r['y'][~np.isnan(r['y'])]
        best, last = (valid.max(), valid[-1]) if len(valid) else (float('nan'), float('nan'))
        c = cost_of[r['label']]
        missing = f"  ({c['missing']} iterations without usage)" if c['missing'] else ''
        print(f"{r['label']:<{width}}  {len(r['x']):>5}  {best:>9.3f}  {last:>9.3f}  {c['provider']:<8}  {c['cost']:>9.2f}{missing}")
    print(f"{'total':<{width}}  {'':>5}  {'':>9}  {'':>9}  {'':<8}  {sum(c['cost'] for c in costs):>9.2f}")
    print(f'written {os.path.relpath(out)}')


if __name__ == '__main__':
    fire.Fire(main)
