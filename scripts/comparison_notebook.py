"""Generate a self-contained interactive comparison notebook and sample figures."""
from pathlib import Path
import nbformat

VIEWER = r'''
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output


def axis_role(array):
    attrs=array.attrs
    if array.dtype.kind=='M' or attrs.get('axis')=='T':return 'time'
    if attrs.get('standard_name')=='grid_latitude' or attrs.get('axis')=='Y':return 'y'
    if attrs.get('standard_name')=='grid_longitude' or attrs.get('axis')=='X':return 'x'
    if attrs.get('axis')=='Z' or attrs.get('positive'):return 'z'
    return 'other'


def load_plane(row,time_index,extra,point_x,point_y,begin=None,end=None):
    paths=report['streams'][row['stream']]
    with xr.open_dataset(paths['reference'],decode_coords='all') as c, xr.open_dataset(paths['candidate'],decode_coords='all') as r:
        a=c[row['field']];b=r[row['candidate_field']];mapping=row['dimension_mapping']
        x=next(d for d in a.dims if axis_role(a[d])=='x')
        y=next(d for d in a.dims if axis_role(a[d])=='y')
        t=next(d for d in a.dims if axis_role(a[d])=='time')
        coords={d:a[d].to_numpy() for d in a.dims}
        selection=dict(extra)|{t:time_index}
        left=a.isel(selection).transpose(y,x).to_numpy().astype(float)
        right=b.isel({mapping[d]:v for d,v in selection.items()}).transpose(mapping[y],mapping[x]).to_numpy().astype(float)
        xi=int(np.argmin(abs(coords[x]-point_x)));yi=int(np.argmin(abs(coords[y]-point_y)))
        times=coords[t]
        if np.dtype(a[t].encoding.get('dtype','float64'))==np.dtype('float32'):
            times=pd.to_datetime(times).round('s').to_numpy()
        aggregation=a.attrs.get('cell_methods','instantaneous')
        bound_name=a[t].attrs.get('bounds',a[t].encoding.get('bounds'))
        if bound_name in c:
            window=c[bound_name].isel({t:time_index}).to_numpy()
            aggregation+=f' ({pd.Timestamp(window[0]).round("s")} to {pd.Timestamp(window[1]).round("s")} UTC)'
        def parse(value):
            stamp=pd.Timestamp(value)
            if stamp.tzinfo:stamp=stamp.tz_convert('UTC').tz_localize(None)
            return stamp.to_datetime64()
        start=parse(begin) if begin else times[0];finish=parse(end) if end else times[-1]
        if start>finish:raise ValueError('Time-series end must follow start')
        indices=np.flatnonzero((times>=start)&(times<=finish))
        if not len(indices):raise ValueError('No records within selected time range')
        point=dict(extra)|{x:xi,y:yi,t:indices}
        ca=a.isel(point).transpose(t).to_numpy().astype(float)
        rb=b.isel({mapping[d]:v for d,v in point.items()}).transpose(mapping[t]).to_numpy().astype(float)
        levels=', '.join(f'{d}={coords[d][i]} {a[d].attrs.get("units", "")}' for d,i in extra.items()) or 'surface'
        return dict(left=left,right=right,x=coords[x],y=coords[y],time=pd.Timestamp(times[time_index]),aggregation=aggregation,
            times=times[indices],c_series=ca,r_series=rb,rlon=float(coords[x][xi]),rlat=float(coords[y][yi]),
            levels=levels,units=a.attrs.get('units','unspecified'),field=row['field'],stream=row['stream'],begin=str(start),end=str(finish))


def map_figure(data,mask=False):
    a,b=data['left'],data['right'];finite=np.concatenate([a[np.isfinite(a)],b[np.isfinite(b)]])
    low=float(finite.min()) if finite.size else 0.;high=float(finite.max()) if finite.size else 1.
    if high==low:low-=max(abs(low)*1e-6,1e-6);high+=max(abs(high)*1e-6,1e-6)
    with np.errstate(invalid='ignore'):delta=b-a
    good=delta[np.isfinite(delta)];limit=float(np.max(abs(good))) if good.size else 0.
    limit=max(limit,np.finfo(float).eps)
    diff=(np.isnan(a)!=np.isnan(b)).astype(float) if mask else delta
    fig,axes=plt.subplots(1,3,figsize=(16,4.6),constrained_layout=True)
    exact=bool(good.size and not np.any(good))
    titles=('C converter','Rust converter','Missing-mask mismatch' if mask else ('Rust − C (exact match)' if exact else 'Rust − C'))
    for i,(ax,values,label) in enumerate(zip(axes,(a,b,diff),titles)):
        cmap='viridis' if i<2 else ('binary' if mask else 'RdBu_r')
        limits=(low,high) if i<2 else ((0,1) if mask else (-limit,limit))
        image=ax.pcolormesh(data['x'],data['y'],values,shading='auto',cmap=cmap,vmin=limits[0],vmax=limits[1])
        ax.set(title=label,xlabel='Native rotated rlon [degrees]',ylabel='Native rotated rlat [degrees]')
        colourbar=fig.colorbar(image,ax=ax,label='Mismatch (0/1)' if i==2 and mask else data['units'])
        if i==2 and exact and not mask:colourbar.set_ticks([0])
    fig.suptitle(f"NZRA {data['stream'].upper()} {data['field']} [{data['units']}] | {data['time']} UTC | {data['levels']} | {data['aggregation']}")
    return fig


def series_figure(data):
    a,b=data['c_series'],data['r_series']
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True,constrained_layout=True)
    axes[0].plot(data['times'],a,color='black',label='C');axes[0].plot(data['times'],b,color='orange',ls='--',label='Rust')
    axes[0].legend();axes[0].set_ylabel(f"{data['field']} [{data['units']}]")
    with np.errstate(invalid='ignore'):delta=b-a
    axes[1].plot(data['times'],delta);axes[1].axhline(0,color='grey',lw=.6)
    axes[1].set(ylabel=f"Rust − C [{data['units']}]",xlabel='Valid time (UTC)')
    fig.suptitle(f"NZRA {data['field']} | rlat={data['rlat']:g}, rlon={data['rlon']:g} | {data['levels']}\n{data['begin']} to {data['end']} UTC")
    return fig


def comparison_widget():
    stream=widgets.Dropdown(options=list(report['streams']),description='Stream')
    field=widgets.Dropdown(description='Field',layout=widgets.Layout(width='75%'))
    time_index=widgets.IntSlider(description='Map time',min=0,max=1,continuous_update=False)
    time_label=widgets.HTML()
    levels=widgets.VBox();extra={}
    x=widgets.FloatText(description='Point rlon');y=widgets.FloatText(description='Point rlat')
    begin=widgets.Text(description='Start UTC');end=widgets.Text(description='End UTC')
    mode=widgets.ToggleButtons(options=['Value difference','Missing masks'],description='Third map')
    button=widgets.Button(description='Plot comparison',button_style='primary')
    status=widgets.HTML();output=widgets.Output();state={}
    def reset_field(event=None):
        extra.clear();output.clear_output();state.clear()
        if not field.value:button.disabled=True;status.value='No comparable fields. See failures/logs below.';return
        row=next(r for r in report['fields'] if r['stream']==stream.value and r['field']==field.value)
        state['row']=row
        with xr.open_dataset(report['streams'][stream.value]['reference'],decode_coords='all') as ds:
            a=ds[row['field']];coords={d:a[d].to_numpy() for d in a.dims}
            axes={axis_role(a[d]):d for d in a.dims}
            if not {'x','y','time'}.issubset(axes):button.disabled=True;status.value='Field has no supported spatial/time axes';return
            t=axes['time'];time_index.max=max(0,a.sizes[t]-1);time_index.value=0;state['times']=coords[t]
            time_label.value=f"Map time: {pd.Timestamp(coords[t][0])} UTC"
            begin.value=str(pd.Timestamp(coords[t][0]));end.value=str(pd.Timestamp(coords[t][-1]))
            x.value=float(coords[axes['x']][a.sizes[axes['x']]//2]);y.value=float(coords[axes['y']][a.sizes[axes['y']]//2])
            for d in a.dims:
                if d not in (axes['x'],axes['y'],t):
                    extra[d]=widgets.Dropdown(description=d,options=[(f'{v} {a[d].attrs.get("units", "")}',i) for i,v in enumerate(coords[d])])
        levels.children=tuple(extra.values());button.disabled=False
        status.value=f"Full-field status: <b>{row['status']}</b>; max absolute error={row.get('max_abs')}; RMSE={row.get('rmse')}. These are full-field statistics."
    def reset_stream(event=None):
        choices=[r['field'] for r in report['fields'] if r['stream']==stream.value and r.get('dimension_mapping') and r.get('status')!='comparison_error']
        field.options=choices;reset_field()
    def draw(event=None):
        with output:
            clear_output(wait=True)
            try:
                row=state['row'];data=load_plane(row,time_index.value,{d:w.value for d,w in extra.items()},x.value,y.value,begin.value,end.value)
                display(map_figure(data,mode.value=='Missing masks'));plt.close('all')
                display(series_figure(data));plt.close('all')
                finite=np.isfinite(data['left'])&np.isfinite(data['right'])
                error=abs(data['right'][finite]-data['left'][finite])
                print('Selected map only: max abs=',float(error.max()) if error.size else 'no finite pairs',data['units'],
                      '; NaN-mask mismatches=',np.count_nonzero(np.isnan(data['left'])!=np.isnan(data['right'])))
                display(pd.DataFrame([row.get('metadata_differences',{})]).T.rename(columns={0:'Metadata difference'}))
            except Exception as error:print(type(error).__name__+': '+str(error))
    def stamp(event):
        if 'times' in state:time_label.value=f"Map time: {pd.Timestamp(state['times'][time_index.value])} UTC"
    stream.observe(reset_stream,names='value');field.observe(reset_field,names='value');time_index.observe(stamp,names='value');button.on_click(draw)
    reset_stream()
    return widgets.VBox([stream,field,levels,time_index,time_label,widgets.HBox([y,x]),widgets.HBox([begin,end]),mode,button,status,output])
'''


def create_notebook(root,filename='comparison.ipynb'):
    c=nbformat.v4
    setup=f'''from pathlib import Path
import json
ROOT = Path.cwd() if (Path.cwd()/"results.json").exists() else Path({str(root)!r})
report = json.loads((ROOT/"results.json").read_text())
print("Comparison complete:", report["complete"])
print("Summary updated:", report.get("finished_utc", report["generated_utc"]))
print(report["note"])
'''
    overview=r'''
summary=pd.DataFrame([{
    'stream':r['stream'],'field':r['field'],'status':r['status'],'max_abs':r.get('max_abs'),
    'RMSE':r.get('rmse'),'mean_bias':r.get('bias'),'max_relative_nonzero_ref':r.get('max_relative_nonzero_reference'),
    'finite_pairs':r.get('finite_pairs'),'outside_tolerance':r.get('outside_tolerance'),
    'outside_percent':100*r.get('outside_tolerance',0)/r['finite_pairs'] if r.get('finite_pairs') else None,
    'NaN_mask_mismatches':r.get('nan_mask_mismatches'),'metadata_differences':', '.join(r.get('metadata_differences',{})),
    'scientific_issues':'; '.join(r.get('scientific_issues',[])),'error':r.get('error','')
} for r in report['fields']])
if summary.empty:
    print('No full comparison results yet. Check conversion failures below, or rerun these cells after the job finishes.')
else:
    display(pd.crosstab(summary.stream,summary.status))
    filter_stream=widgets.Dropdown(options=['All']+sorted(summary.stream.unique().tolist()),description='Stream')
    filter_status=widgets.Dropdown(options=['All']+sorted(summary.status.unique().tolist()),description='Status')
    sort=widgets.Dropdown(options=['outside_percent','max_relative_nonzero_ref','RMSE','max_abs','field'],description='Sort')
    table=widgets.Output()
    def refresh_table(event=None):
        view=summary
        if filter_stream.value!='All':view=view[view.stream==filter_stream.value]
        if filter_status.value!='All':view=view[view.status==filter_status.value]
        with table:
            clear_output(wait=True)
            display(view.sort_values(sort.value,ascending=sort.value=='field',na_position='last'))
    for control in (filter_stream,filter_status,sort):control.observe(refresh_table,names='value')
    display(widgets.VBox([widgets.HBox([filter_stream,filter_status,sort]),table]));refresh_table()
    worst=summary.dropna(subset=['outside_percent']).sort_values('outside_percent',ascending=False).head(25)
    if not worst.empty:
        fig,ax=plt.subplots(figsize=(12,5));ax.bar(range(len(worst)),worst.outside_percent)
        ax.set_xticks(range(len(worst)),worst.stream+'/'+worst.field,rotation=75,ha='right')
        ax.set(title='NZRA compatibility: finite values outside rtol=1e-6, atol=0 (largest 25)',ylabel='Percent of finite paired samples')
        display(fig);plt.close(fig)
'''
    timing=r'''
rows=[]
for stream,item in report['streams'].items():
    print(stream.upper(),item.get('status'),item.get('error',''))
    for tool in ('C','Rust'):
        r=item.get(tool)
        if r:
            rows.append({'stream':stream,'tool':tool,'wall_seconds':r['wall_seconds'],
                         'CPU_seconds':r.get('user_seconds',0)+r.get('system_seconds',0),
                         'peak_RSS_MiB':r.get('peak_rss_kib',0)/1024,
                         'output_GB':item.get('output_bytes',{}).get(tool,0)/1e9,
                         'returncode':r['returncode']})
            print('  Log:',r['log'])
            print('  Command:',r['command'])
    if item.get('traceback'):print(item['traceback'])
metrics=pd.DataFrame(rows)
display(metrics)
if not metrics.empty:
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,column,label in zip(axes,['wall_seconds','peak_RSS_MiB','output_GB'],['Elapsed seconds','Peak RSS MiB','Output GB']):
        metrics.pivot(index='stream',columns='tool',values=column).plot.bar(ax=ax,rot=0)
        ax.set(ylabel=label,title='NZRA '+label)
    display(fig);plt.close(fig)
'''
    n=c.new_notebook(cells=[c.new_markdown_cell('''# xarray-um versus the installed C um2netcdf

**All available fields, times and levels** in one staged MLEV, SFC and PLEV file are attempted by both tools. Both use the same STASH XML and thermodynamic-grid interpolation. The initialisations differ between streams, but each pair uses an identical input file. Inputs and production workflows are untouched.

Select **Python (xarray-um comparison)** and run the cells in order. If the job is still running, results are provisional; rerun these cells after completion. The notebook reads saved NetCDFs and summaries; it never starts conversions.

Status distinguishes exact values, values within `rtol=1e-6, atol=0`, mask/value mismatches, scientific metadata differences and comparison/conversion errors. These are compatibility checks against another converter, not observational scientific validation. Units differ between fields: use dimensionless violation fractions for cross-field comparisons, not absolute errors across unlike units.

Plots read selected slices on demand. Use an allocated JupyterHub session for large interactive reads. This notebook needs NumPy, pandas, xarray, netCDF4, Matplotlib and ipywidgets; a dedicated Pixi kernel is prepared. Generic ipywidgets support is sufficient; Bokeh is not required.'''),c.new_code_cell(setup),c.new_code_cell(VIEWER),c.new_markdown_cell('## Full-field verification summary\nThe table and error fractions cover every compared plane, not only the interactive selection. Numeric coordinate tolerance: `rtol=1e-6, atol=1e-5`; timestamps and reporting bounds must match exactly after explicitly bounded float32 reference-time canonicalisation to whole UM lookup seconds. C tile ordinals are explicitly matched to native pseudo-level IDs. Geographic differences are retained and checked independently against each file’s CF projection. An absent finite-pair statistic is not a zero error.'),c.new_code_cell(overview),c.new_markdown_cell('## Inspect matched maps and point time series\nSelect stream, field, time and levels. Enter native rotated coordinates and an inclusive UTC point-series range, then click **Plot comparison**. Source maps share colour limits; difference limits are symmetric. The reported point is the nearest grid point; no spatial interpolation is performed during comparison.'),c.new_code_cell('comparison_widget()'),c.new_markdown_cell('## Conversion resources, failures and provenance\nConversions run C then Rust sequentially, with single-threaded numerical libraries. Cache state and ordering are uncontrolled; timings are not a controlled speed benchmark. Metadata and storage-encoding differences are recorded independently of value agreement. Failed streams are not silently omitted.'),c.new_code_cell(timing),c.new_code_cell('print(json.dumps({k:v for k,v in report.items() if k not in ("fields","streams")},indent=2))\nfor stream,item in report["streams"].items():\n    print(stream, json.dumps(item.get("input_identity",{}),indent=2))'),c.new_markdown_cell('## Detailed metadata and coordinate checks'),c.new_code_cell('if report["fields"]:\n    details=widgets.Dropdown(options=[(r["stream"]+"/"+r["field"],i) for i,r in enumerate(report["fields"])],description="Field")\n    detail_out=widgets.Output()\n    def show_details(event=None):\n        with detail_out:\n            clear_output(wait=True)\n            print(json.dumps(report["fields"][details.value],indent=2))\n    details.observe(show_details,names="value")\n    display(details,detail_out);show_details()')])
    n.metadata['kernelspec']={'display_name':'Python (xarray-um comparison)','language':'python','name':'xarray-um-comparison'}
    for cell in n.cells:
        if cell.cell_type=='code':compile(cell.source,'<comparison notebook>','exec')
    nbformat.validate(n)
    with (root/filename).open('x') as f:nbformat.write(n,f)


def sample_figure(root,stream,rows):
    import json
    namespace={'report':json.loads((root/'results.json').read_text())}
    exec(VIEWER,namespace)
    preferred={'mlev':'turb_kinetic_energy','sfc':'sfc_soil_moisture','plev':'std_lev_temp'}
    row=next((r for r in rows if r['field']==preferred.get(stream) and r.get('dimension_mapping')),next(r for r in rows if r.get('dimension_mapping') and r.get('status')!='comparison_error'))
    import xarray as xr
    with xr.open_dataset(namespace['report']['streams'][stream]['reference'],decode_coords='all') as ds:
        a=ds[row['field']];role=namespace['axis_role']
        x=next(d for d in a.dims if role(a[d])=='x');y=next(d for d in a.dims if role(a[d])=='y')
        extras={d:0 for d in a.dims if role(a[d]) not in ('x','y','time')}
        px=float(a[x].to_numpy()[a.sizes[x]//2]);py=float(a[y].to_numpy()[a.sizes[y]//2])
    data=namespace['load_plane'](row,0,extras,px,py)
    namespace['map_figure'](data).savefig(root/f'{stream}_sample_maps.png',dpi=110)
    namespace['series_figure'](data).savefig(root/f'{stream}_sample_series.png',dpi=110)
    namespace['plt'].close('all')
