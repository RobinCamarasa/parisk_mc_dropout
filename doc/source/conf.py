import os
import sys
import datetime
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))
)

project = '##__NAME__##'
author = '##__AUTHOR__##'
copyright = '{}, {}'.format(
    datetime.datetime.now().year,
    author
)


# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']
