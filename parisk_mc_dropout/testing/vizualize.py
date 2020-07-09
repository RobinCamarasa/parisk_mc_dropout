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
from parisk_mc_dropout.testing.utils import *
from pyfiglet import Figlet
import os
import pandas as pd
import plotly.express as px
import plotly


def main():
    # Load parameters
    parameters = pd.read_csv(
        os.path.join(RESULT_ROOT, 'training_subset', '#glogger_training.csv')
    ).set_index('index')

    # Generate dropout type
    dropout = []
    drop_list = parameters['model_dropout'].unique().tolist()
    drop_type = parameters['model_dropout_type'].unique().tolist()
    for type_ in drop_type:
        for drop in drop_list:
            dropout.append('{}_{}'.format(drop, type_))

    nb_images = parameters['datasplitter_nb_training'].unique().tolist()
    nb_images = [i for i in nb_images if i != 20]

    # Load results
    for hospital_id in ['AMC', 'MUMC', 'UMCU']:
        results = pd.read_csv(
            os.path.join(RESULT_ROOT, 'testing_subset', '{}.csv'.format(hospital_id))
        ).set_index('index')
        tmp_ = results.join(parameters)
        data_ = np.zeros((len(nb_images), len(dropout)))
        for column in list(results):
            for _, row in tmp_.iterrows():
                try:
                    i = dropout.index('{}_{}'.format(row['model_dropout'], row['model_dropout_type']))
                    j = nb_images.index(row['datasplitter_nb_training'])
                    data_[j, i] = row[column]
                except Exception as e:
                    pass
            fig = px.imshow(
                data_,
                labels=dict(x='Dropout type', y='nb_training_images', color=column, colorscale='Viridis'),
                x=dropout,
                y=['{} images'.format(nb_images_) for nb_images_ in nb_images]
            )
            create_error_less_directory(os.path.join(RESULT_ROOT, 'testing_subset', hospital_id))
            plotly.offline.plot(
                fig, filename=os.path.join(
                    RESULT_ROOT, 'testing_subset', hospital_id, '{}.html'.format(column),
                ), auto_open=False
            )

if __name__ == '__main__':
    main()
