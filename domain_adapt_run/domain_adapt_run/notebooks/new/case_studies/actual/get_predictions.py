import numpy as np, pandas as pd, itertools, collections, matplotlib.pyplot as plt
import python_utils.python_utils.basic as basic

# add run number to caching, make cache folder higher up

def run(path, mode, profile, compute, recompute, by_fitter, fitter_groups):
    # accepts path
    fitter_groups_d = basic.parent_import(path, 'fitter_groups_d').fiter_groups_d(path)
    fitter_infos = sum([fitter_groups_d[fitter_group].items() for fitter_group in fitter_groups], [])
    raw_data_getter = basic.parent_import_wrapper(path, 'raw_data_getter')
    cv_getter = basic.parent_import_wrapper(path, 'get_cv_getter')
    subgroup_name = basic.parent_import_wrapper(path, 'get_subgroup_name')
    invariants = basic.parent_import_wrapper(path, 'get_invariants')
    num_trials = invariants['num_trials']
    
    cache_folder = basic.parent_import(path, 'caching').cache_folder
    
    return _run(cache_folder, mode, profile, compute, recompute, by_fitter, fitter_infos, raw_data_getter, cv_getter, num_trials, subgroup_name)

def _run(cache_folder, mode, profile, compute, recompute, by_fitter, fitter_infos, raw_data_getter, cv_getter, num_trials, subgroup_name):
    # accepts arguments that were determined by path 
    
    def get_path(identifier, fitter_info, raw_data_getter, cv_getter, trial, subgroup_name):
        fitter_name, fitter = fitter_info
        return '%s/fitter=%s/subgroup=%s/i=%d' % (cache_folder, fitter_name, subgroup_name, trial)
    
    predictions_path = lambda path: '%s_predictions' % path
    info_path = lambda path: '%s_info' % path
    
    def reader(path):
        return (pd.DataFrame.read_csv(prediction_path(path)), pd.Series.read_csv(info_path(path)))
    
    def writer((predictions, info), path):
        predictions.to_csv(predictions_path(path))
        info.to_csv(info_path(path))
        
    suffix = ''
    
    map_getter = basic.map_getter(mode, profile, compute, recompute)
    mapper = map_getter(reader, writer, get_path, suffix, splat=True)
    return __run(mapper, by_fitter, fitter_infos, raw_data_getter, cv_getter, num_trials, subgroup_name)

def __run(mapper, by_fitter, fitter_infos, raw_data_getter, cv_getter, num_trials, subgroup_name):    
   
    def get_predictions(fitter_info, raw_data_getter, cv_getter, trial, subgroup_name):
        fitter_name, fitter = fitter_info
        xs_is, ys_is, all_zs_is, xs_oos, ys_oos, all_zs_oos = cv_getter(xs, ys, all_zs, i)
        zs_is, zs_oos = all_zs_is[subgroup_name], all_zs_oos[subgroup_name]
        fitter = fitter_getter()
        predictor = fitter.fit(xs_is.values.astype(float), ys_is.values.astype(float), zs_is.astype(float))
        ys_oos_hat = predictor.predict(xs_oos.values.astype(float), zs_oos.values.astype(float))
        info = predictor.info
        return (pd.concat((zs_oos, pd.DataFrame({'ys':ys_oos}), pd.DataFrame({'ys_hat':ys_oos_hat})), axis=1), info)    
    
    if by_fitter:
        ans = []
        for fitter_info in fitter_infos:
            ans.append(mapper(get_predictions, itertools.product(xrange(num_trials), [fitter_info], [raw_data_getter], [cv_getter], [subgroup_name])))
        return ans
    else:
        return mapper(get_predictions, itertools.product(xrange(num_trials), fitter_infos, [raw_data_getter], [cv_getter], [subgroup_name]))