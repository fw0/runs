def fitter_groups_d(path):
    import domain_adapt.domain_adapt.new.subgroup as subgroup
    from sklearn.linear_model import LogisticRegressionCV
    return {
        'baselines':[
            ('all_logreg', subgroup.all_fitter(LogisticRegressionCV())),
            ('subgroup_logreg', subgroup.subgroup_fitter(LogisticRegressionCV())),
            ],
        }