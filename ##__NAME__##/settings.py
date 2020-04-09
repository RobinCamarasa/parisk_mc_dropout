"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-03-23

**Project** : ##__NAME__##

**Contains the global settings of the ##__NAME__## project**

"""
import os

# Set project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Set text_file.txt root
TEST_ROOT = os.path.join(PROJECT_ROOT, 'test')

# Set ressources root
RESSOURCES_ROOT = os.path.join(PROJECT_ROOT, 'ressources')

