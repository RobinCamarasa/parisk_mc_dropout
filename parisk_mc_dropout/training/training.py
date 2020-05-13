"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-05

**Project** : parisk_mc_dropout

** File that creates the command related to training **
"""
import os

import click

from data_science_framework.data_spy.options.option_manager import parameters_to_options, \
    initialize_experiment_parameters
from data_science_framework.data_spy.loggers.experiment_loggers import global_logger
from data_science_framework.pytorch_utils.models import MCUnet2Axis
from data_science_framework.pytorch_utils.trainer import VanillaTrainer
from data_science_framework.pytorch_utils.optimizer import AdadeltaOptimizer
from data_science_framework.pytorch_utils.losses import DiceLoss, BinaryCrossEntropyLoss, \
        WeightedCrossEntropy, GeneralizedDiceLoss
from data_science_framework.data_augmentation.segmentation_augmentation import \
        SegmentationGTExpander, SegmentationInputExpander, SegmentationCropHalf,\
        SegmentationROISelector, SegmentationNormalization, SegmentationToTorch,\
        SegmentationFlip, SegmentationGTDropClasses, SegmentationGTCustomExpander,\
        SegmentationRotation
from data_science_framework.pytorch_utils.metrics import SegmentationAccuracyMetric,\
    SegmentationDiceMetric, SegmentationBCEMetric
from data_science_framework.pytorch_utils.callbacks import ModelCheckpoint,\
        ModelPlotter, MetricsWritter, ConfusionMatrixCallback, DataDisplayer

from torch.utils.tensorboard import SummaryWriter

from parisk_mc_dropout.training import PariskDataset, PariskDataSplitter
from parisk_mc_dropout.settings import RESULT_ROOT, PROJECT_ROOT, DATA_ROOT, DEVICE
import pickle


# Defining the objects required for your experiments
EXPERIMENT_OBJECTS = {
    'datasplitter': PariskDataSplitter(),
    'traingenerator': PariskDataset(batchsize=1),
    'validationgenerator': PariskDataset(batchsize=1),
    'normalization': SegmentationNormalization(),
    'flip': SegmentationFlip(
        flip_x=True, random=True
    ),
    'inputexpander': SegmentationInputExpander(
        tile_shape_x=128, tile_shape_y=128, tile_shape_z=16
    ),
    'gtcustomexpander': SegmentationGTCustomExpander(),
    'roiselector': SegmentationROISelector(
        shape_x=128, shape_y=128, shape_z=16,
        centered=True, background=True
    ),
    'model': MCUnet2Axis(
        in_channels=5, out_channels=3, depth=3,
        n_features=16, activation='softmax', dropout=0.3
    ),
    'optimizer': AdadeltaOptimizer(weight_decay=0.0001),
    'loss': DiceLoss(),
    'trainer': VanillaTrainer(nb_epochs=600),
}
@click.command()
@parameters_to_options(
    experiment_objects=EXPERIMENT_OBJECTS
)
@global_logger(
    folder=os.path.join(RESULT_ROOT, 'training_subset'), tag='training',
    project_root=os.path.join(PROJECT_ROOT, 'parisk_mc_dropout')
)
def main(experiment_folder, **option_values):
    """
    Function that launches a training
    """
    # Initialize experiment parameters
    initialize_experiment_parameters(
        experiment_objects=EXPERIMENT_OBJECTS,
        option_values=option_values
    )

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(experiment_folder, 'tensorboard'))
    writer.add_hparams(
            {
                key: str(value)
                for key, value in option_values.items()
            },
            {},
    )

    # Split data
    print('- Split data')

    EXPERIMENT_OBJECTS['datasplitter'].process_parameters()
    splitted_dataframes = EXPERIMENT_OBJECTS['datasplitter'].split_data(
        save=True,
        folder=experiment_folder,
        subdirectories=['split_data'],
        tag='training'
    )

    # Update generators
    EXPERIMENT_OBJECTS['traingenerator'].process_parameters(
        dataframe=splitted_dataframes['train'],
        transformations= [
            EXPERIMENT_OBJECTS['inputexpander'],
            EXPERIMENT_OBJECTS['gtcustomexpander'],
            SegmentationCropHalf(),
            EXPERIMENT_OBJECTS['normalization'],
            EXPERIMENT_OBJECTS['roiselector'],
            EXPERIMENT_OBJECTS['flip'],
            SegmentationToTorch(device=DEVICE)
        ]
    )
    EXPERIMENT_OBJECTS['validationgenerator'].process_parameters(
        dataframe=splitted_dataframes['validation'],
        transformations= [
            EXPERIMENT_OBJECTS['inputexpander'],
            EXPERIMENT_OBJECTS['gtcustomexpander'],
            SegmentationCropHalf(),
            EXPERIMENT_OBJECTS['normalization'],
            EXPERIMENT_OBJECTS['roiselector'],
            EXPERIMENT_OBJECTS['flip'],
            SegmentationToTorch(device=DEVICE)
        ]
    )

    # Set VanillaTrainer
    EXPERIMENT_OBJECTS['trainer'].set_objects_attributes(
        writer=writer,
        model=EXPERIMENT_OBJECTS['model'],
        trainning_generator=EXPERIMENT_OBJECTS['traingenerator'],
        validation_generator=EXPERIMENT_OBJECTS['validationgenerator'],
        callbacks= [
            ModelCheckpoint(
                writer=writer,
                metric=SegmentationDiceMetric(),
                save_folder=experiment_folder,
                metric_to_minimize=True
            ),
            ModelPlotter(
                writer=writer, model=EXPERIMENT_OBJECTS['model']
            ),
            DataDisplayer(writer=writer),
            ConfusionMatrixCallback(
                writer=writer
            )
        ]
    )
    EXPERIMENT_OBJECTS['trainer'].set_loss(
        EXPERIMENT_OBJECTS['loss']
    )
    EXPERIMENT_OBJECTS['trainer'].set_optimizer(
        optimizer=AdadeltaOptimizer()
    )

    # Save objects
    for key, value in EXPERIMENT_OBJECTS.items():
        if 'generator' in key:
            path = os.path.join(experiment_folder, '{}.pkl'.format(key))
            with open(path, 'wb') as handle:
                pickle.dump(value, handle)

    # Run training
    print('- Run training')
    EXPERIMENT_OBJECTS['trainer'].run()


if __name__ == '__main__':
    main()
