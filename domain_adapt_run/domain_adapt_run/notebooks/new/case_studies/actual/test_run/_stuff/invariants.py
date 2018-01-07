def invariants(path):
    import numpy as np
    return {
        'metrics_d':{
            'accuracy':lambda ys_hat, ys: ((ys_hat > 0) == (ys > 0)).sum() / float(len(ys)),
            }
        'num_trials': 10,
        }