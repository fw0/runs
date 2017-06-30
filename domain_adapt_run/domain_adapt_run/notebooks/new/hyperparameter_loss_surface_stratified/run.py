import matplotlib.pyplot as plt
from IPython.display import display_pretty, display_html
import python_utils.python_utils.basic as basic
import python_utils.python_utils.caching as caching
import pandas as pd
import pdb
import itertools
import functools
import autograd.numpy as np

def horse_helper(get_data, get_fitter, num_iterations, (inner_iteration,hyperparam_tuples)):
    # calculates results for 1 hyperparameter setting
    hyperparam_dict = dict(hyperparam_tuples)
    fitter = get_fitter(**hyperparam_dict)
    test_losses = []
    train_losses = []
    for i in xrange(num_iterations):
        xs_train, xs_test, ys_train, ys_test = get_data(seed=i)
        predictor = fitter(xs_train, xs_test, ys_train, ys_test)
        ys_test_hat = predictor(xs_test)
        test_errors = ys_test - ys_test_hat
        test_loss = np.dot(test_errors, test_errors) / len(test_errors)
        test_losses.append(test_loss)
        ys_train_hat = predictor(xs_train)
        train_errors = ys_train - ys_train_hat
        train_loss = np.dot(train_errors, train_errors) / len(train_errors)
        train_losses.append(train_loss)
        caching.fig_archiver.fig_text(['single iteration', 'inner iteration: %d' % inner_iteration, hyperparam_tuples, 'iteration: %d' % i, 'test_loss: %.2f' % test_loss, 'train_loss: %.2f' % train_loss])
    caching.fig_archiver.fig_text(['single hyperparam', hyperparam_tuples, 'test mean loss: %.2f' % np.mean(test_losses), 'test std loss: %.2f' % np.std(test_losses), 'test losses: %s' % test_losses, 'train mean loss: %.2f' % np.mean(train_losses), 'train std loss: %.2f' % np.std(train_losses), 'train losses: %s' % train_losses])
    return (hyperparam_tuples, {'test_mean_loss':np.mean(test_losses),'test_std_loss':np.std(test_losses),'test_losses':test_losses, 'train_mean_loss':np.mean(train_losses),'train_std_loss':np.std(train_losses),'train_losses':train_losses})


def run(get_data, get_fitter, care_hyperparam_tupless, nocare_hyperparam_tupless, num_iterations, mapper):
    
    nocare_hyperparam_tupless = list(nocare_hyperparam_tupless)
    
    horse = functools.partial(horse_helper, get_data, get_fitter, num_iterations)
    
    for (outer_iteration, care_hyperparam_tuples) in enumerate(care_hyperparam_tupless):
        this_hyperparam_tupless = [care_hyperparam_tuples + nocare_hyperparam_tuples for nocare_hyperparam_tuples in nocare_hyperparam_tupless]
        this_results = mapper(horse, enumerate(this_hyperparam_tupless))
        this_results_display = ['%s: test_mean_loss(std): %.2f(%.2f) train_mean_loss(std): %.2f(%.2f)' % (hyperparam_tuples, result_dict['test_mean_loss'], result_dict['test_std_loss'], result_dict['train_mean_loss'], result_dict['train_std_loss']) for (hyperparam_tuples, result_dict) in this_results]
        import pdb
        best_hyperparam_tuples, best_result_dict = min(this_results, key=lambda (hyperparam_tuples,result_dict): result_dict['test_mean_loss'])
        caching.fig_archiver.fig_text(['single care_hyperparam_tuples', care_hyperparam_tuples, 'outer iteration: %d' % outer_iteration, 'best: %s: test_mean_loss(std_loss): %.2f(%.2f) train_mean_loss(std_loss): %.2f(%.2f)' % (best_hyperparam_tuples, best_result_dict['test_mean_loss'], best_result_dict['test_std_loss'], best_result_dict['train_mean_loss'], best_result_dict['train_std_loss']), 'all:'] + this_results_display)
        this_results_df = pd.DataFrame.from_dict(dict(this_results))
        display_html(this_results_df.to_html(), raw=True)