import os

cache_folder = '%s/%s' % (os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'cache')
which_hash = 'sha256'
shelf_file = '%s/%s' % (cache_folder, 'shelf')
