import python_utils.python_utils.basic as basic
import python_utils.python_utils.nbrun as nbrun
import argparse, os, imp, pdb, sys, itertools
from ipyparallel.util import interactive

def get_args():
    parser = argparse.ArgumentParser(description='runs get_predictions for all leaf paths, all fitter groups')
    parser.add_argument('--path', type=str, help='path of root', required=True, nargs='+')
    parser.add_argument('--outer_mode', type=str, help='outer_parallelism mode', default='serial')
    parser.add_argument('--inner_mode', type=str, help='inner_parallelism mode', default='serial')
    parser.add_argument('--profile', type=str, help='ipyparallel profile', default='')
    parser.add_argument('--compute', type=str, help='whether to compute results', required=True)
    parser.add_argument('--recompute', type=str, help='whether to recompute results', required=True)
    parser.add_argument('--by_fitter', type=str, help='whether compute predictions 1 fitter at a time', default=False)
    parser.add_argument('--fitter_groups', type=str, help='names of fitter_groups to run experiments for', nargs='+', required=True)
    return args

def get_paths(args):
    return basic.hardcoded_crawl([args.path])

def runner(args, paths):
    # returns fxn of paths
    
    mapper = basic.basic_map_getter(args.outer_mode, args.profile)
    notebook_name = 'get_predictions.ipynb'
    notebook_path = basic.parent_find(args.path, notebook_name)
    
    def horse(args, path, fitter_group):
        nbrun.run_notebook(notebook_path, out_path=path, nb_suffix=';fitter_groups=%s' % (args.fitter_groups,), nb_kwargs={'path':path,, 'mode':args.inner_mode, 'profile':args.profile, 'compute':compute, 'recompute':recompute, 'by_fitter':args.by_fitter 'fitter_groups':args.fitter_groups}, hide_input=True, insert_pos=1, timeout=-1)
        
    mapper(horse, itertools.product([args,], paths, args.fitter_groups))
    
args = get_args()
paths = get_paths(args)
runner(args, paths)