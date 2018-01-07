import python_utils.python_utils.basic as basic
import os, pdb
import _stuff

children_list = [
    {
        'all': {
            'invariants.py': basic.writeable_object(_stuff.invariants.invariants, display_name='invariants'),
            'cv_getter.py': basic.writeable_object(_stuff.cv_getter.subsample_cv_getter, display_name='cv_getter'),
            'fitter_groups.py': basic.writeable_object(_stuff.fitter_groups_d.fitter_groups_d, display_name='fitter_groups_d'),
        }
    },
    {
        'foster_violent':{'raw_data.py': basic.writeable_object(_stuff.raw_data.foster_violent_data, display_name='raw_data')},
    },
    {
        'male':{'subgroup_name.py': basic.writeable_object(_stuff.subgroup_name.male_subgroup_name, display_name='subgroup_name'), 'constants.py': basic.writeable_object(_stuff.constants)},
        'black':{'subgroup_name.py': basic.writeable_object(_stuff.subgroup_name.black_subgroup_name, display_name='subgroup_name'), 'constants.py': basic.writeable_object(_stuff.constants)},
    },
]

tree = basic.identical_tree(basic.node(), children_list)
this_path = os.path.dirname(os.path.realpath(__file__))
basic.write_directory_tree(this_path, tree)