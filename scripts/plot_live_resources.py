"""Create a vector paper figure from a completed, translation-enabled live run."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    summary = json.loads((args.folder/'summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'complete':
        raise ValueError('Completed live resource record required')
    rows = [json.loads(line) for line in (args.folder/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    if not all(r['translation_enabled'] and r['state']=='live' for r in rows):
        raise ValueError('The figure requires live OCR with translation enabled')
    start, end = rows[0]['monotonic'],rows[-1]['monotonic']
    telemetry = [json.loads(line) for line in (args.folder/'tegrastats.jsonl').read_text().splitlines()]
    telemetry = [r for r in telemetry if start<=r['monotonic']<=end]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    figure,axes = plt.subplots(3,1,figsize=(3.5,2.85),sharex=True,constrained_layout=True)
    for key,label,color in [('cpu_percent','CPU','#276c8e'),('gpu_percent','GPU','#bb7342')]:
        records = [r for r in telemetry if key in r]
        axes[0].plot([r['monotonic']-start for r in records],[r[key] for r in records],label=label,color=color,lw=.8)
    axes[0].set(ylabel='Utilization (%)',ylim=(0,100))
    axes[0].legend(loc='lower right',bbox_to_anchor=(1,1.04),
                   ncol=2,frameon=False,fontsize=6,borderaxespad=0)
    axes[1].plot([r['monotonic']-start for r in rows],[r['VmRSS_mib'] for r in rows],color='#4a8c70',lw=1)
    axes[1].set(ylabel='Worker RSS\n(MiB)',ylim=(0,summary['sampled_peak_worker_rss_mib']*1.15))
    records = [r for r in telemetry if 'module_power_w' in r]
    if not records:
        raise ValueError('Module power sensor unavailable; do not plot as zero')
    axes[2].plot([r['monotonic']-start for r in records],[r['module_power_w'] for r in records],color='#7e7598',lw=.8)
    axes[2].set(ylabel='VDD_IN (W)',xlabel='Elapsed live observation (s)',ylim=(0,max(r['module_power_w'] for r in records)*1.15))
    for axis in axes:
        axis.grid(alpha=.18)
        axis.set_xlim(0,end-start)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ('pdf','svg','png'):
        figure.savefig(args.output.with_suffix('.'+extension),dpi=300)
    plt.close(figure)
    provenance = dict(source_directory=str(args.folder),summary=summary,
                      plotted_window='First through last live API observation; telemetry outside this window excluded.',
                      sha256={name:hashlib.sha256((args.folder/name).read_bytes()).hexdigest()
                              for name in ('metadata.json','samples.jsonl','tegrastats.jsonl','summary.json')})
    args.output.with_suffix('.metadata.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
