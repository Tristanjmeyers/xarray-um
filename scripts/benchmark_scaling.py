#!/usr/bin/env python3
"""Reproducible single-file latency and equal-budget batch export benchmarks.

Real conversion requires a PBS allocation. Paths/project choices are supplied by
arguments and the batch environment; no site-specific defaults are embedded.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

LOG=logging.getLogger('benchmark')


def dump(path,value):
    from compare_converters import write_json
    write_json(path,value)


def identity(path):
    path=Path(path);st=path.stat()
    return {'name':path.name,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_order(allowed):
    first,siblings,seen=[],[],set()
    for cpu in allowed:
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=((p/'physical_package_id').read_text().strip(),(p/'core_id').read_text().strip())
        (siblings if key in seen else first).append(cpu);seen.add(key)
    return first+siblings


def encoding(path):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        out={}
        for name,v in ds.variables.items():
            if 'stash_item' not in v.ncattrs():continue
            f=v.filters();c=v.chunking()
            out[name]={'dtype':str(v.dtype),'zlib':bool(f.get('zlib')),
                'complevel':int(f.get('complevel',0)),'shuffle':bool(f.get('shuffle')),
                'fletcher32':bool(f.get('fletcher32')),
                '_FillValue':float(v.getncattr('_FillValue')) if '_FillValue' in v.ncattrs() else None}
            out[name].update({'contiguous':True} if c=='contiguous' else {'chunksizes':c})
        return out


def make_manifest(fixtures,references,xml):
    import xarray as xr
    from compare_converters import data_fields,methods
    manifest=[]
    for stream,count in [('mlev',8),('plev',4),('sfc',20)]:
        source=Path(fixtures[stream]['path']);reference=references/stream/'c.nc'
        with xr.open_dataset(reference,decode_coords='all') as c,xr.open_dataset(source,engine='um',stash_xml=xml) as rust:
            code=lambda v:int(v.attrs['stash_section'])*1000+int(v.attrs['stash_item'])
            common=sorted({code(v) for v in data_fields(c).values()} & {code(v) for v in rust.data_vars.values()})
            if len(common)<count:raise ValueError(f'Need {count} common STASH groups for {stream}; found {len(common)}')
            codes=common[:count];templates=encoding(reference);names=[];translated={}
            for name,v in rust.data_vars.items():
                if code(v) not in codes:continue
                matches=[n for n,f in data_fields(c).items() if code(f)==code(v) and methods(f)==methods(v)]
                if len(matches)!=1:raise ValueError(f'Ambiguous encoding template for {stream}/{name}')
                translated[name]=templates[matches[0]];names.append(name)
            manifest.append({'stream':stream,'source':str(source),'input':identity(source),
                'reference':str(reference),'codes':codes,'fields':names,'encoding':translated,
                'logical_bytes':sum(v.size*4 for v in rust.data_vars.values() if code(v) in codes)})
    return manifest


def export_task(task):
    logging.basicConfig(level=logging.WARNING)
    if task['implementation']=='C':
        cmd=[task['converter'],task['source'],'-q','-c',task['xml'],'-i','-p','32','-o',task['output'],'-s',*map(str,task['codes'])]
        subprocess.run(cmd,check=True)
    else:
        from xarray_um.cli import write_single
        write_single(task['source'],task['output'],stash_xml=task['xml'],stash_codes=task['codes'],
            interpolate=True,workers=task['workers'],encoding_template=task['encoding'])


def worker(path):
    plan=json.loads(Path(path).read_text())
    if plan['mode']=='single':export_task(plan['tasks'][0]);return
    if plan['implementation']=='C':
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:list(pool.map(export_task,plan['tasks']))
    else:
        with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            list(pool.map(export_task,plan['tasks']))


def measured(command,cpus,root,label):
    import psutil
    metrics=root/(label+'.time.json');log=root/(label+'.log')
    fmt='{"user_seconds":%U,"system_seconds":%S,"peak_process_rss_kib":%M,"filesystem_input_blocks":%I,"filesystem_output_blocks":%O,"exit_status":%x}'
    invocation=['/usr/bin/time','--quiet','-f',fmt,'-o',str(metrics),'/usr/bin/taskset','--cpu-list',','.join(map(str,cpus)),*command]
    start=time.perf_counter();rss=0
    with log.open('x') as output:
        process=subprocess.Popen(invocation,stdout=output,stderr=subprocess.STDOUT)
        probe=psutil.Process(process.pid)
        while process.poll() is None:
            try:
                children=probe.children(recursive=True)
                total=0
                for child in children:
                    try:total+=child.memory_info().rss
                    except psutil.Error:pass
                rss=max(rss,total)
            except psutil.Error:pass
            time.sleep(.2)
        returncode=process.wait()
    result=json.loads(metrics.read_text())
    result.update(wall_seconds=time.perf_counter()-start,peak_aggregate_rss_bytes=rss,
        command=command,cpu_ids=cpus,returncode=returncode,log=log.name)
    if returncode:raise RuntimeError(f'{label} failed; see {log}')
    return result


def validate(plan,manifest):
    from compare_converters import compare_files
    results=[]
    for task in plan['tasks']:
        stream=task['stream'];reference=next(m['reference'] for m in manifest if m['stream']==stream)
        rows=compare_files(reference,task['output'],stream,require_candidate_cf=task['implementation']!='C')
        # The reference contains the entire file; selected exports must account
        # for every candidate and every requested source STASH group only.
        requested=set(task['codes'])
        rows=[r for r in rows if r['status']=='missing_reference' or
            int(r.get('reference',r.get('candidate'))['attrs']['stash_section'])*1000+
            int(r.get('reference',r.get('candidate'))['attrs']['stash_item']) in requested]
        failures=[r for r in rows if r['status'] not in ('exact','within_tolerance','matching_masks_no_finite_pairs')]
        results.extend(rows)
        if failures:
            dump(Path(task['output']).with_suffix('.validation-failure.json'),failures)
            raise RuntimeError(f'Validation failed for {stream}: {[(r["field"],r["status"]) for r in failures]}')
    return results


def warm(paths):
    start=time.perf_counter();buffer=bytearray(16*1024**2)
    for path in paths:
        with Path(path).open('rb',buffering=0) as f:
            while f.readinto(buffer):pass
    return time.perf_counter()-start


def run(args):
    import importlib.metadata
    if not os.environ.get('PBS_JOBID'):raise RuntimeError('Real-data benchmarks require PBS')
    if args.repeats!=2 or args.workers!=[1,2,4,8,16]:raise ValueError('Release matrix is fixed: 1,2,4,8,16 workers and two repeats')
    if max(args.workers)>int(os.environ.get('NCPUS','1')):raise ValueError('Workers exceed allocation')
    allowed=core_order(sorted(os.sched_getaffinity(0)))
    if len(allowed)<max(args.workers):raise ValueError('Insufficient CPU affinity')
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    fixtures=json.loads(args.fixtures.read_text())
    manifest=make_manifest(fixtures,args.references,args.xml)
    dump(root/'workload.json',manifest)
    report={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'job_id':os.environ['PBS_JOBID'],'cpu_order':allowed[:16],
        'cpu_model':next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown'),
        'versions':{p:importlib.metadata.version(p) for p in ('xarray-um','xarray','numpy','netCDF4')},
        'native_sha256':sha(__import__('xarray_um._native',fromlist=['_native']).__file__),'converter_version':subprocess.run([str(args.converter),'--version'],capture_output=True,text=True,check=True).stdout.strip(),'converter_sha256':sha(args.converter),'xml_sha256':sha(args.xml),'runner_sha256':sha(__file__),
        'library_versions':{'python_netcdf_c':__import__('netCDF4').getlibversion(),
            'python_hdf5':__import__('netCDF4').__hdf5libversion__,
            'python_zlib':__import__('zlib').ZLIB_RUNTIME_VERSION,
            'c_dependency_names':[line.strip().split()[0] for line in subprocess.run(['ldd',str(args.converter)],capture_output=True,text=True,check=True).stdout.splitlines() if line.strip()]},
        'library_note':'These are the installed C and Rust/Python software stacks. Data selection, precision, chunks and compression settings match; linked library builds can differ.',
        'cache_policy':'One best-effort sequential input read before each stream/repeat and batch/repeat; no shared cache flushing. Storage/server/output cache effects remain uncontrolled.',
        'memory_note':'Peak aggregate RSS sampled every 0.2 seconds across the process tree; shared library pages can be counted more than once.',
        'workers':args.workers,'repeats':args.repeats,'trials':[]}
    dump(root/'results.json',report)
    cases=[]
    for item in manifest:
        for repeat in (1,2):
            cases.append(('warm',[item['source']]))
            trials=[('C',1)]+[('Rust',w) for w in args.workers]
            if repeat==2:trials.reverse()
            cases.extend(('single',item,tool,w,repeat) for tool,w in trials)
    for repeat in (1,2):
        cases.append(('warm',[m['source'] for m in manifest]))
        trials=[(tool,w) for w in args.workers for tool in ('C','Rust')]
        if repeat==2:trials.reverse()
        cases.extend(('batch',None,tool,w,repeat) for tool,w in trials)
    try:
        for case in cases:
            if case[0]=='warm':
                LOG.info('Best-effort input cache pass %.1fs',warm(case[1]));continue
            mode,item,tool,workers,repeat=case
            label=f'{mode}-{item["stream"] if item else "all"}-{tool}-{workers}cpu-r{repeat}'
            directory=root/label;directory.mkdir();tasks=[]
            for selected in ([item] if item else manifest):
                for codes in ([selected['codes']] if mode=='single' else [[c] for c in selected['codes']]):
                    output=directory/(f'{Path(selected["source"]).stem}_{codes[0]}.nc' if mode=='batch' else 'dataset.nc')
                    tasks.append({'stream':selected['stream'],'source':selected['source'],'output':str(output),
                        'codes':codes,'encoding':selected['encoding'],'workers':workers if mode=='single' else 1,
                        'xml':str(args.xml),'converter':str(args.converter),'implementation':tool})
            plan={'mode':mode,'implementation':tool,'workers':workers,'tasks':tasks}
            taskfile=root/(label+'.task.json');dump(taskfile,plan)
            LOG.info('Starting %s',label)
            row=measured([sys.executable,str(Path(__file__).resolve()),'--worker-task',str(taskfile)],allowed[:workers],root,label)
            row.update(mode=mode,stream=item['stream'] if item else 'all',implementation=tool,workers=workers,repeat=repeat,
                logical_bytes=item['logical_bytes'] if item else sum(m['logical_bytes'] for m in manifest),
                output_bytes=sum(Path(t['output']).stat().st_size for t in tasks))
            start=time.perf_counter();verification=validate(plan,manifest)
            dump(root/(label+'.validation.json'),verification)
            row.update(validation_seconds=time.perf_counter()-start,fields_verified=len(verification))
            report['trials'].append(row);dump(root/'results.json',report)
            LOG.info('PASS %s %.1fs validation %.1fs',label,row['wall_seconds'],row['validation_seconds'])
            # Retain one representative result for each endpoint and implementation.
            retain=repeat==1 and workers in (1,16)
            if not retain:
                for task in tasks:Path(task['output']).unlink()
        report['status']='passed'
    except Exception as error:
        report['status']='failed';report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat();dump(root/'results.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-task',type=Path)
    p.add_argument('--fixtures',type=Path);p.add_argument('--references',type=Path)
    p.add_argument('--xml',type=Path);p.add_argument('--converter',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--workers',nargs='+',type=int,default=[1,2,4,8,16])
    p.add_argument('--repeats',type=int,default=2)
    args=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.worker_task:worker(args.worker_task)
    else:run(args)

if __name__=='__main__':main()
