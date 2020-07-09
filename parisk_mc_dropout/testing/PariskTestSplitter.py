"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-04-14

**Project** : parisk_mc_dropout

**Implement test data splitting**

"""
import os
import pandas as pd

from data_science_framework.data_spy.loggers.experiment_loggers import timer, data_saver
from data_science_framework.scripting.file_structure_manager import get_dir_structure

from parisk_mc_dropout.settings import DATA_ROOT


class PariskTestSplitter:
    """
    Class that handle data splitting

    :param data_id: Version of your data
    :param nsplit: Number of split of the dataset
    :param validation: List of the subsplit used for validation
    :param test: List of the subsplit used for test
    :param centers: Contains the list of the hospital center under study
    """
    def __init__(self):
        # Initialize parameters
        self.MUMC = 4
        self.UMCU = 4
        self.AMC = 4


    @data_saver
    def split_data(self, *args, **kwargs) -> dict:
        """
        Function that splits dataset into train, validation and test

        :return: Dictionnary of dataframes
        """
        # Get dataframe
        df = pd.read_csv(
            os.path.join(DATA_ROOT, 'parisk_data_split.csv')
        )

        return {
            'MUMC': df[df.subset == 4],
            'UMCU': df[df.subset == 5],
            'AMC': df[df.subset == 6],
        }

