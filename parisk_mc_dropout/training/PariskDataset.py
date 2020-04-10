"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-06

**Project** : parisk_mc_dropout

** Class that implements DataGenerator **

"""
import os

from parisk_mc_dropout.training import PariskDataSplitter
from parisk_mc_dropout.settings import DEVICE, DATA_ROOT
from data_science_framework.data_spy.loggers.experiment_loggers import timer
from torch.utils.data import Dataset
import torch
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import glob
from parisk_mc_dropout.settings import DATA_ROOT

from data_science_framework.data_augmentation.segmentation_augmentation import \
    SegmentationTransformation
from typing import List


class PariskDataset(Dataset):
    """
    Class that generates data

    :param batchsize: Size of the batch
    """

    def __init__(self, batchsize: int = 1):
        self.batchsize = batchsize

    def process_parameters(
            self, dataframe,
            transformations: List[SegmentationTransformation]
    ) -> None:
        """
        Method that processes parameters

        :param dataframe: Dataframe containing the patients studied by this datagenerator
        :return: None
        """
        self.dataframe = dataframe
        self.transformations = transformations

    def __getitem__(self, idx, *args, **kwargs):
        idx = [self.batchsize * idx + i for i in range(self.batchsize)]

        # Get dataframe
        dataframe_ = self.dataframe.reset_index().iloc[idx]

        # Get data
        input_batch = []
        gt_batch = []
        for index, row in dataframe_.iterrows():
            input_batch.append(
                [
                    nib.load(
                        glob.glob(
                            os.path.join(
                                DATA_ROOT, 'input_images', '*{}*{}*'.format(
                                    row['patient_id'], row['modality_{}'.format(i)]
                                )
                            )
                        )[0]
                    )
                    for i in range(1, 6)
                ]
            )
            gt_batch.append(
                [
                    nib.load(
                        glob.glob(
                            os.path.join(
                                DATA_ROOT, 'gt_images', '*{}*'.format(
                                    row['patient_id']
                                )
                            )
                        )[0]
                    )
                ]
            )
        niifty_template = input_batch[0][0]
        input_images = input_batch[0]

        # Apply transformations
        for i, transformation in enumerate(self.transformations):
            input_batch, gt_batch = transformation.transform_batch(
                input_batch, gt_batch
            )

        # Return batch
        return input_batch, gt_batch, {
            'niifty_template': niifty_template
        }

    def __len__(self):
        return int(len(self.dataframe)/self.batchsize)
