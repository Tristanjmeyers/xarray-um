#!/usr/bin/env python3
"""Create public, path-free compatibility and performance evidence."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from comparison_notebook import VIEWER


def build(comparison,benchmark,destination):
    report=json.loads((comparison/'results.json').read_text())
    performance=json.loads((benchmark/'results.json').read_text())
    if not report.get('success') or performance['status']!='passed':
        raise RuntimeError('Correctness and performance runs must pass before public evidence is generated')
    destination.mkdir(parents=True,exist_ok=False)
    rows=[]
    for field in report['fields']:
        rows.append({k:field.get(k) for k in ('stream','field','candidate_field','status','finite_pairs',
            'max_abs','rmse','bias','outside_tolerance','nan_mask_mismatches','positive_inf_mismatches','negative_inf_mismatches')} | {'units':field.get('reference',{}).get('attrs',{}).get('units')})
    fields=pd.DataFrame(rows);fields.to_csv(destination/'field_comparison.csv',index=False)
    counts={s:dict(Counter(f['status'] for f in report['fields'] if f['stream']==s)) for s in report['streams']}
    raw=pd.DataFrame(performance['trials'])
    summary=raw.groupby(['mode','stream','implementation','workers'],as_index=False).agg(
        median_seconds=('wall_seconds','median'),min_seconds=('wall_seconds','min'),max_seconds=('wall_seconds','max'),
        peak_aggregate_RSS_bytes=('peak_aggregate_rss_bytes','max'),logical_bytes=('logical_bytes','first'),
        median_output_bytes=('output_bytes','median'),repeats=('repeat','count'))
    summary['logical_GB_hour']=summary.logical_bytes/1e9/summary.median_seconds*3600
    summary['written_GB_hour']=summary.median_output_bytes/1e9/summary.median_seconds*3600
    def baseline(row):
        c=summary[(summary['mode']==row['mode']) & (summary.stream==row.stream) & (summary.implementation=='C')]
        if row['mode']=='batch':c=c[c.workers==row.workers]
        return c.iloc[0]
    summary['speedup_vs_C']=summary.apply(lambda r:baseline(r).median_seconds/r.median_seconds,axis=1)
    summary.to_csv(destination/'scaling.csv',index=False)
    # A repeated improvement must survive both measured repeat ranges.
    gains=[]
    for _,row in summary[summary.implementation=='Rust'].iterrows():
        c=baseline(row)
        if row.max_seconds<c.min_seconds:
            gains.append({'mode':row['mode'],'stream':row.stream,'workers':int(row.workers),
                          'median_speedup':row.speedup_vs_C})
    if not gains:raise RuntimeError('No repeatable end-to-end improvement; release gate remains closed')
    public={'package_version':performance['versions']['xarray-um'],'reference':performance.get('converter_version','C converter recorded by SHA256'),
        'workload':[{k:m[k] for k in ('stream','codes','fields','logical_bytes')} for m in json.loads((benchmark/'workload.json').read_text())],
        'total_finite_pairs':sum(f.get('finite_pairs',0) for f in report['fields']),
        'outside_tolerance':sum(f.get('outside_tolerance',0) for f in report['fields']),
        'NaN_mask_mismatches':sum(f.get('nan_mask_mismatches',0) for f in report['fields']),
        'counts':counts,'rtol':report['rtol'],'atol':report['atol'],
        'inputs':{s:Path(v['input']).name for s,v in report['streams'].items()},
        'cpu_model':performance['cpu_model'],'cpu_counts':'Trials run in one PBS allocation and are restricted by CPU affinity; physical cores preferred before SMT siblings',
        'cache_policy':performance['cache_policy'],'memory_note':performance['memory_note'],
        'repeated_improvements':gains,'versions':performance['versions'],'library_versions':({**performance['library_versions'],'c_dependency_names':[Path(n).name for n in performance['library_versions']['c_dependency_names']]} if performance.get('library_versions') else None),
        'library_note':performance.get('library_note'),
        'xml_sha256':performance['xml_sha256'],'converter_sha256':performance['converter_sha256'],
        'auxiliary_note':'C geographic auxiliaries disagree with their own CF projection. xarray-um geographic coordinates are independently checked against CF; raw discrepancies remain reported.',
        'coordinate_policy':'C float32 elapsed-time rounding is bounded by encoding ULP and canonicalised to the whole seconds stored in UM lookup records. C zero-based tile ordinals are matched explicitly to original one-based pseudo-level IDs.'}
    (destination/'summary.json').write_text(json.dumps(public,indent=2)+'\n')
    # Native-grid maps make field comparisons independent of geographic auxiliary errors.
    namespace={'report':report};exec(VIEWER,namespace)
    for stream,name in [('mlev','turb_kinetic_energy'),('sfc','sfc_soil_moisture'),('plev','std_lev_temp')]:
        row=next(r for r in report['fields'] if r['stream']==stream and r['field']==name)
        import xarray as xr
        with xr.open_dataset(report['streams'][stream]['reference'],decode_coords='all') as ds:
            a=ds[name];role=namespace['axis_role']
            x=next(d for d in a.dims if role(a[d])=='x');y=next(d for d in a.dims if role(a[d])=='y')
            extra={d:min(9 if stream=='mlev' else a.sizes[d]//2,a.sizes[d]-1) for d in a.dims if role(a[d]) not in ('time','x','y')}
            px=float(a[x][a.sizes[x]//2]);py=float(a[y][a.sizes[y]//2])
        data=namespace['load_plane'](row,0,extra,px,py)
        namespace['map_figure'](data).savefig(destination/f'{stream}_comparison.png',dpi=120)
        if stream=='mlev':namespace['series_figure'](data).savefig(destination/'point_series.png',dpi=120)
        plt.close('all')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for stream in ('mlev','sfc','plev'):
        subset=summary[(summary['mode']=='single')&(summary.stream==stream)&(summary.implementation=='Rust')]
        axes[0].errorbar(subset.workers,subset.speedup_vs_C,
            yerr=[subset.speedup_vs_C-subset.apply(lambda r:baseline(r).median_seconds/r.max_seconds,axis=1),
                  subset.apply(lambda r:baseline(r).median_seconds/r.min_seconds,axis=1)-subset.speedup_vs_C],
            marker='o',capsize=3,label=stream.upper())
    axes[0].axhline(1,color='black',ls='--',label='Serial C baseline')
    axes[0].set(title='NZRA single-file conversion',ylabel='Speedup relative to serial C')
    batch=summary[summary['mode']=='batch']
    for tool in ('C','Rust'):
        subset=batch[batch.implementation==tool]
        axes[1].errorbar(subset.workers,subset.logical_GB_hour,
            yerr=[subset.logical_GB_hour-subset.logical_bytes/1e9/subset.max_seconds*3600,
                  subset.logical_bytes/1e9/subset.min_seconds*3600-subset.logical_GB_hour],marker='o',capsize=3,label=tool)
    axes[1].set(title='NZRA batch: same 32 STASH-group tasks',ylabel='Decoded float32 GB/hour')
    for ax in axes:
        ax.set(xlabel='CPU affinity limit / workers (logical CPUs)',xticks=[1,2,4,8,16]);ax.grid(alpha=.25);ax.legend()
    fig.suptitle('xarray-um versus C um2netcdf | matched fields, interpolation and storage encoding\nMedians and observed ranges from two repeats; not confidence intervals')
    fig.savefig(destination/'scaling.png',dpi=140);plt.close(fig)
    # Markdown generated from measured summaries, never hand-entered speed claims.
    lines=['| Workload | Tool | Workers | Seconds (median; range) | Logical GB/h | Written GB/h | Speedup vs C |',
           '|---|---|---:|---:|---:|---:|---:|']
    for _,r in summary.iterrows():
        lines.append(f'| {r["mode"]}/{r.stream} | {r.implementation} | {int(r.workers)} | {r.median_seconds:.1f} ({r.min_seconds:.1f}–{r.max_seconds:.1f}) | {r.logical_GB_hour:.1f} | {r.written_GB_hour:.1f} | {r.speedup_vs_C:.2f}× |')
    (destination/'scaling.md').write_text('\n'.join(lines)+'\n')
    from comparison_notebook import create_notebook
    create_notebook(comparison,'comparison_v020.ipynb')
    print(json.dumps(public,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',required=True,type=Path);p.add_argument('--benchmark',required=True,type=Path)
    p.add_argument('--destination',required=True,type=Path);a=p.parse_args();build(a.comparison,a.benchmark,a.destination)

if __name__=='__main__':main()
