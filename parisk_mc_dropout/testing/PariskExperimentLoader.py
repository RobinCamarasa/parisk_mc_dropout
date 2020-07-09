"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-04-14

**Project** : parisk_mc_dropout

**Implement the experiment loader**

"""
import pickle
from parisk_mc_dropout.settings import *
import torch


class PariskExperimentLoader:
    """
    Class that implements Parisk PariskExperimentLoader

    :param experiment_id: Id of the experiment considered
    """
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.experiment_path =  os.path.join(
            RESULT_ROOT, 'training_subset', '#training_{}'.format(experiment_id)

        )
        # Load model
        with open(os.path.join(self.experiment_path, 'model.pkl'), 'rb') as handle:
            self.model = pickle.load(handle)
        self.model.load_state_dict(torch.load(os.path.join(self.experiment_path, 'best_model_dice.pt')))

        # Load model
        with open(os.path.join(self.experiment_path, 'validationgenerator.pkl'), 'rb') as handle:
            self.generator = pickle.load(handle)
