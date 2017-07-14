import numpy as np
import domain_adapt.domain_adapt.new.cv as cv
import python_utils.python_utils.caching as caching


def run(random_data_getter, num_trials, which_loss, num_folds, use_test, fitter):

    all_losses = []
    
    for i in xrange(num_trials):

        xs_train, xs_test, ys_train, ys_test = random_data_getter(seed=i)

        trial_losses = cv.get_oos_loss(which_loss, num_folds, use_test, fitter, xs_train, xs_test, ys_train, ys_test=ys_test if use_test else None)

        all_losses.append(np.mean(trial_losses))

        caching.fig_archiver.log_text('iteration %d: %.2f(%.2f)' % (i, np.mean(all_losses), np.std(all_losses)))

    return all
