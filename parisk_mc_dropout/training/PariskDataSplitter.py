import os
import pandas as pd

from data_science_framework.data_spy.loggers.experiment_loggers import timer, data_saver
from data_science_framework.scripting.file_structure_manager import get_dir_structure
from data_science_framework.data_splitter import DataSplitter

from parisk_mc_dropout.settings import DATA_ROOT


class PariskDataSplitter:
    """
    Class that handle data splitting

    :param data_id: Version of your data
    :param nsplit: Number of split of the dataset
    :param validation: List of the subsplit used for validation
    :param training: List of the subsplit used for training
    :param test: List of the subsplit used for test
    :param centers: Contains the list of the hospital center under study
    """
    def __init__(
            self, validation: int = 3, nb_training: int = 69
    ):
        # Initialize parameters
        self.validation = validation
        self.nb_training = nb_training

    def process_parameters(self) -> None:
        """
        Method that processes parameters

        :return: None
        """
        # Get data ressources
        self.train = [i for i in range(4) if i != self.validation]


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
        train_df = df[df.subset.isin(self.train)].sample(
            n=self.nb_training, random_state=1
        )
        validation_df = df[df.subset == self.validation]
        return {
            'train': train_df,
            'validation': validation_df
        }

