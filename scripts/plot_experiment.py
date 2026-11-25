#!/usr/bin/env python3
"""Create a paper figure and LaTeX table from completed measured exports.

Run on an analysis workstation with matplotlib installed; it is not required by
the Jetson image. Raw measurements are read without modification.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics


def number(value):
    return float(value) if value not in ('', None) else None


def plot(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with (folder / 'comparison.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('No completed conditions to plot.')
    output = folder / 'figures'
    output.mkdir(exist_ok=True)
    labels = [r['condition'].replace('-width', '\nwidth ') for r in rows]
    x = list(range(len(rows)))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(3, 2, figsize=(9, 8.4), constrained_layout=True)
    colors = ['#276c8e', '#4a8c70', '#bb7342']
    fps = [number(r['throughput_mean_fps']) for r in rows]
    error = [number(r['throughput_stdev_fps']) or 0 for r in rows]
    axes[0, 0].bar(x, fps, yerr=error, capsize=4, color=colors[:len(rows)])
    axes[0, 0].set(ylabel='Processed frames / s', title='(a) Wall throughput; error bars = repeat SD')
    for i, value in enumerate(fps):
        axes[0, 0].text(i, value + error[i] + max(fps) * 0.025, f'{value:.2f}', ha='center', fontsize=8)
    axes[0, 0].set_ylim(0, max(v + e for v, e in zip(fps, error)) * 1.2)
    bottom = [0.0] * len(rows)
    stages = [('decode_resize_ms', 'Decode/resize', '#b2c4d6'), ('detect_ms', 'Detection', '#276c8e'),
              ('order_ms', 'Order', '#7e7598'), ('recognize_ms', 'Recognition', '#4a8c70'),
              ('annotate_ms', 'Annotation', '#bb7342'), ('encode_ms', 'JPEG encode', '#d5b85a')]
    summaries = [json.loads((folder / r['condition'] / 'summary.json').read_text()) for r in rows]
    for key, label, color in stages:
        values = [statistics.fmean(repeat['stages_ms'][key]['mean'] for repeat in summary['repeats']) for summary in summaries]
        axes[0, 1].bar(x, values, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[0, 1].set(ylabel='Mean processing time (ms)', title='(b) Arithmetic mean of repeat stage means')
    axes[0, 1].legend(fontsize=7, ncol=2, frameon=False)
    rss = [number(r['process_peak_rss_mib']) for r in rows]
    axes[1, 0].bar(x, [v or 0 for v in rss], color=colors[:len(rows)])
    axes[1, 0].set(ylabel='Process peak RSS (MiB)', title='(c) OCR runtime + benchmark process')
    for i, value in enumerate(rss):
        axes[1, 0].text(i, (value or 0) + max(v or 1 for v in rss) * 0.02,
                        f'{value:.0f}' if value is not None else 'Unavailable', ha='center', fontsize=8)
    power = [number(r['module_power_w_sample_mean']) for r in rows]
    axes[1, 1].bar(x, [v or 0 for v in power], color=colors[:len(rows)])
    axes[1, 1].set(ylabel='Sampled mean VDD_IN (W)', title='(d) Module power during measured windows')
    for i, value in enumerate(power):
        axes[1, 1].text(i, (value or 0) + max(v or 1 for v in power) * 0.02,
                        f'{value:.2f}' if value is not None else 'Unavailable', ha='center', fontsize=8)
    for axis, metric, title in [(axes[2, 0], 'cpu_percent_sample_mean', '(e) Mean CPU utilization across reported cores'),
                                (axes[2, 1], 'gpu_percent_sample_mean', '(f) Mean GPU utilization')]:
        values = [number(r[metric]) for r in rows]
        axis.bar(x, [v or 0 for v in values], color=colors[:len(rows)])
        axis.set(ylabel='Sampled utilization (%)', title=title, ylim=(0, 100))
        for i, value in enumerate(values):
            axis.text(i, (value or 0) + 2, f'{value:.1f}' if value is not None else 'Unavailable', ha='center', fontsize=8)
    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(axis='y', alpha=0.18)
        axis.set_axisbelow(True)
    figure.suptitle('Jetson Orin Nano: four synthetic Assamese scenes\nOffline OCR + annotation + JPEG; translation disabled', fontsize=11)
    figure.savefig(output / 'resource_comparison.png', dpi=300)
    figure.savefig(output / 'resource_comparison.svg')
    plt.close(figure)

    def fmt(value, precision=2):
        result = number(value)
        return f'{result:.{precision}f}' if result is not None else '--'
    lines = [r'% Generated from comparison.csv; preserve the accompanying raw measurements.',
             r'\begin{table}[t]', r'\centering\small',
             r'\caption{Synthetic resource pilot: four generated scenes, three timed repeats. Processing p95 is the mean of repeat-level p95 values; RSS includes the benchmark runtime.}',
             r'\label{tab:synthetic-pilot}', r'\begin{tabular}{lrrrr}', r'\hline',
             r'Condition & FPS $\pm$ SD & p95 (ms) & RSS (MiB) & Power (W)\\', r'\hline']
    for row in rows:
        name = row['condition'].replace('-width', ' ')
        lines.append(name + ' & ' + fmt(row['throughput_mean_fps']) + r' $\pm$ ' + fmt(row['throughput_stdev_fps']) +
                     ' & ' + fmt(row['mean_repeat_processing_p95_ms']) + ' & ' + fmt(row['process_peak_rss_mib'], 0) +
                     ' & ' + fmt(row['module_power_w_sample_mean']) + r'\\')
    lines += [r'\hline', r'\end{tabular}', r'\end{table}', '']
    (output / 'paper_table.tex').write_text('\n'.join(lines), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_directory', type=Path)
    args = parser.parse_args()
    plot(args.result_directory.resolve())
