from sklearn.metrics import roc_curve, auc
from scipy import interp
import pdb, time
import numpy as np
import pandas as pd
import itertools
from IPython.display import display_html
import matplotlib.pyplot as plt
import python_utils.python_utils.basic as basic

def run(num_trials, fitter_infos, cv, data_infos):

    trial_lw = 0.15
    trial_alpha = 0.23
    agg_lw = 1.2
    agg_alpha = 0.8
    fpr_points = np.linspace(0.,1.,101)
    color_wheel = itertools.cycle(['r','cyan','green','blue','purple','orange','yellow','magenta','brown','black'])

    time_df = pd.DataFrame()

    for ((xs, ys, x_names), data_name) in data_infos:
        fig, ax = plt.subplots()
        for ((fitter, fitter_name), color) in zip(fitter_infos,color_wheel):
            fitter_tprs = []
            fitter_aucs = []
            ts = time.time()
            for i in xrange(num_trials):
                print i, fitter_name
                xs_train, ys_train, xs_test, ys_test = cv(xs, ys, i)
                try:
                    predictor = fitter.fit(xs_train, ys_train, x_names)
                except:
                    predictor = fitter.fit(xs_train, ys_train)
                print predictor
                ys_test_hat = predictor.predict_proba(xs_test)
                if len(ys_test_hat.shape) == 2:
                    ys_test_hat = ys_test_hat[:,1]
                fpr, tpr, thresholds = roc_curve(ys_test, ys_test_hat)
                fitter_aucs.append(auc(fpr, tpr))
                ax.plot(fpr, tpr, color=color, lw=trial_lw, alpha=trial_alpha)
                fitter_tprs.append(interp(fpr_points, fpr, tpr))
            te = time.time()
            time_df.loc[fitter_name,data_name] = te - ts
            fitter_auc = np.mean(fitter_aucs)
            ax.plot(fpr_points, np.mean(fitter_tprs, axis=0), color=color, label='%s %.3f' % (fitter_name,fitter_auc), lw=agg_lw)
            ax.legend()
            ax.set_xlabel('fpr')
            ax.set_ylabel('tpr')
            ax.set_title('%s ROCs' % data_name)
            basic.display_fig_inline(fig)
        display_html(time_df.to_html(), raw=True)
        