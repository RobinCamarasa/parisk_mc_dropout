"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-03-23

**Project** : parisk_mc_dropout

**Contains the global settings of the parisk_mc_dropout project**

"""
import os

# Set project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# Set data root
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')

# Set result root
RESULT_ROOT = os.path.join(PROJECT_ROOT, 'results')

# Set device
DEVICE = 'cuda'
