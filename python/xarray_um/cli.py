"""Compatibility command for the NZRA workflow's converter invocation."""
import argparse
import logging
from pathlib import Path
import uuid
import dask
import xarray as xr
from .backend import UMBackend

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('-c','--stash-xml',type=Path)
    parser.add_argument('-i',dest='interpolate',action='store_true')
    parser.add_argument('-o','--output',required=True,type=Path)
    parser.add_argument('-s',dest='stash_codes',nargs='+',type=int)
    parser.add_argument('-r',dest='float32',action='store_true',help='32-bit output (also the NZRA compatibility default)')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.input.resolve()==args.output.resolve():parser.error('Input and output must differ')
    if args.output.exists():raise FileExistsError(args.output)
    temporary=args.output.with_name(args.output.name+'.'+uuid.uuid4().hex+'.writing')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with dask.config.set(scheduler='synchronous'):
            with xr.open_dataset(args.input,engine='um',stash_xml=args.stash_xml,
                                 stash_codes=args.stash_codes,grid='thermodynamic' if args.interpolate else 'native',chunks={}) as ds:
                encoding={}
                for name,field in ds.data_vars.items():
                    if field.ndim:
                        encoding[name]={'dtype':'float32','zlib':True,'complevel':3,'shuffle':True,
                                        'chunksizes':tuple(n if field[d].attrs.get('axis') in ('X','Y') else 1 for d,n in field.sizes.items())}
                logging.info('Writing %s fields=%s',args.output,list(ds.data_vars))
                ds.to_netcdf(temporary,engine='netcdf4',encoding=encoding)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    logging.info('Published %s',args.output)

if __name__=='__main__':main()
