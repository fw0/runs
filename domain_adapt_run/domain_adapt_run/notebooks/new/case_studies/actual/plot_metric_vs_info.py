import numpy as np, pandas as pd, itertools, collections, matplotlib.pyplot as plt
import python_utils.python_utils.basic as basic
import get_predictions

def run(path, inner_mode, profile, fitter_groups):
    # accepts path
    fitter_groups_d = basic.parent_import(path, 'fitter_groups_d').fiter_groups_d(path)
    fitter_infos = sum([fitter_groups_d[fitter_group].items() for fitter_group in fitter_groups], [])
    raw_data_getter = basic.parent_import_wrapper(path, 'raw_data_getter')
    cv_getter = basic.parent_import_wrapper(path, 'get_cv_getter')
    subgroup_name = basic.parent_import_wrapper(path, 'get_subgroup_name')
    invariants = basic.parent_import_wrapper(path, 'get_invariants')
    num_trials = invariants['num_trials']
    metrics_d = invariants['metrics_d']
    
    cache_folder = basic.parent_import(path, 'caching').cache_folder
    
    return _run(cache_folder, inner_mode, profile, fitter_infos, raw_data_getter, cv_getter, num_trials, subgroup_name, metrics_d)

def _run(cache_folder, inner_mode, profile, fitter_infos, raw_data_getter, cv_getter, num_trials, subgroup_name, metrics_d):
    
    # assumes analysis is done in serial
    
    def plot(ax, data, val_key, x_pos_key):
    
        method_names = data.keys()
        positions = [np.mean(data[method_name][x_pos_key]) for method_name in method_names]
        vals = list(np.array([data[method_name][val_key] for method_name in method_names]))
    
        d = ax.boxplot(x=vals, positions=positions, labels=method_names, patch_artist=True, showfliers=False, manage_xticks=False)

        for (box,method_name, color) in zip(d['boxes'], method_names, color_wheel):
            box.set_label(method_name)
            box.set_color(color)
            ax.add_patch(box)
        
        ax.set_xlabel(x_pos_key)
        ax.set_ylabel(val_key)
        ax.legend()
    
    compute = False
    recompute = False 
    by_fitter = False
    d = {}
    info_names = None
    for fitter_info in fitter_infos:
        fitter_name, fitter = fitter_info
        d[fitter_name] = defaultdict(list)
        for (predictions, info) in get_predictions._run(cache_folder, inner_mode, profile, compute, recompute, by_fitter, [fitter_info], raw_data_getter, cv_getter, num_trials, subgroup_name):
            info_names = info.keys()
            for (info,val) in info.iteritems():
                d[fitter_name][key].append(val)
            ys, ys_hat = predictions['ys'], predictions['ys_hat']
            for (metric_name,metric) in metrics_d.iteritems():
                d[fitter_name][metric_name].append(metric(ys, ys_hat))
        
    metric_names = metrics_d.keys()
    for info_name in info_names:
        for metric_name in metric_names:
            fig, ax = plt.subplots()
            plot(ax, d, metric_name, info_name)
            ax.set_title('%s vs. %s' % (metric_name,info_name))
    
    from IPython.display import display_html
    means_df = pd.DataFrame({fitter_name:{key:np.mean(vals) for (key,vals) in vals_d.iteritems()} for (fitter_name,vals_d) in d.iteritems()})
    std_df = pd.DataFrame({fitter_name:{key:np.std(vals) for (key,vals) in vals_d.iteritems()} for (fitter_name,vals_d) in d.iteritems()})
    print 'means:'
    display_html(means_df.to_html(), raw=True)
    print 'stds:'
    display_html(std_df.to_html(), raw=True)