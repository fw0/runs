import numpy as np
import domain_adapt.domain_adapt.new.cv as cv
import python_utils.python_utils.caching as caching
from IPython.display import display_pretty, display_html
import pandas as pd


def run(random_data_getter, num_trials, which_loss, num_folds, use_test, fitter_infos):

    summary_d = {}

    for (fitter_name, fitter) in fitter_infos:
    
        all_losses = []

        for i in xrange(num_trials):

            xs_train, xs_test, ys_train, ys_test = random_data_getter(seed=i)

            #trial_losses = cv.get_oos_loss(which_loss, num_folds, use_test, fitter, xs_train, xs_test, ys_train, ys_test=ys_test if use_test else None)
            trial_losses = cv.get_oos_loss(which_loss, num_folds, use_test, fitter, xs_train, xs_test, ys_train, ys_test=ys_test)


            all_losses.append(np.mean(trial_losses))

            caching.fig_archiver.log_text('fitter: %s, iteration %d: so far: %.2f(%.2f)' % (fitter_name, i, np.mean(all_losses), np.std(all_losses)))

        summary_d[fitter_name] = {'mean': np.mean(all_losses), 'std': np.std(all_losses)}

    display_html(pd.DataFrame(summary_d).to_html(), raw=True)
