"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-04-14

**Project** : parisk_mc_dropout

**Implements testing**

"""
import click
from data_science_framework.data_analyser.tester import MCTester
from data_science_framework.data_analyser.analyser import Analyser, DataSaver, ConfusionMatricesAnalyser
from data_science_framework.data_spy.options.option_manager import parameters_to_options, initialize_experiment_parameters
from data_science_framework.data_spy.loggers.experiment_loggers import global_logger
from data_science_framework.data_augmentation.segmentation_augmentation import SegmentationToTorch
from data_science_framework.scripting.file_structure_manager import create_error_less_directory
from parisk_mc_dropout.settings import *
from parisk_mc_dropout.testing import PariskTestSplitter, PariskExperimentLoader
from parisk_mc_dropout.training import PariskDataset
from parisk_mc_dropout.testing.vizualizing_utils import *
from pyfiglet import Figlet
import os
import pandas as pd
import torch


# Define Test experiment object
EXPERIMENT_OBJECTS = {
    'experimentobject': PariskExperimentLoader(experiment_id=22)
}

# Tested functions
FUNCTIONS = [unc]

@click.command()
@parameters_to_options(
    experiment_objects=EXPERIMENT_OBJECTS
)
def main(**option_values):
    # Set random seed
    torch.manual_seed(5)
    np.random.seed(5)

    # Initialize object
    initialize_experiment_parameters(EXPERIMENT_OBJECTS, option_values)

    print(
        Figlet().renderText(
            'Test exp {}'.format(EXPERIMENT_OBJECTS['experimentobject'].experiment_id)
        )
    )

    # Create repositories
    experiment_folder = os.path.join(
        RESULT_ROOT, 'testing_unc_metrics', '#training_{}'.format(EXPERIMENT_OBJECTS['experimentobject'].experiment_id)
    )
    create_error_less_directory(experiment_folder)

    # Get dataframe
    dfs = PariskTestSplitter().split_data(
        save=True,
        folder=experiment_folder,
        subdirectories=['split_data'],
        tag='testing_unc_metrics'
    )

    # Create repositories
    for f in FUNCTIONS:
        create_error_less_directory(os.path.join(experiment_folder, f.__name__))

    # Loop over subsets
    for key, value in dfs.items():
        # Create generator
        generator_ = PariskDataset()
        generator_.process_parameters(
            dataframe=value,
            transformations=EXPERIMENT_OBJECTS['experimentobject'].generator.transformations
        )
        csv_path_ = os.path.join(RESULT_ROOT, 'testing_unc_metrics', '{}.csv'.format(key))
        cumulated_output_ = {}

        # Loop over tested functions
        for f in FUNCTIONS:
            # Save path
            path_ = os.path.join(experiment_folder, f.__name__)
            output_ = f(
                key, generator_,
                EXPERIMENT_OBJECTS['experimentobject'].model,
                path_
            )
            cumulated_output_.update(output_)
        cumulated_output_serie_ = pd.Series(
            cumulated_output_, name=EXPERIMENT_OBJECTS['experimentobject'].experiment_id
        )


if __name__ == '__main__':
    main()
