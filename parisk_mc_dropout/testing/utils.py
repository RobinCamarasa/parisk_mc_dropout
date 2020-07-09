"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-04-14

**Project** : parisk_mc_dropout

**Implements utils function for testing**

"""
import tqdm
import json
import nibabel as nib
import os
from parisk_mc_dropout.training.PariskDataset import PariskDataset
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, average_precision_score
from scipy.stats import entropy
from data_science_framework.scripting.file_structure_manager import create_error_less_directory
from data_science_framework.pytorch_utils.losses import DiceLoss
import torch
import matplotlib.pyplot as plt
import time


def dice(
    model_output: np.ndarray, gt: np.ndarray,
    class_: np.ndarray
):
    """
    Compute dice

    :param model_output: Array of shape (n_i, n_c, n_x, n_y, n_z)
    where n_i is the number of monte-carlo samples, n_c the number
    of classes, n_
    """
    # Compute mean over Monte-Carlo samples
    model_output = model_output.mean(axis=0)

    # Get output for specific class
    model_output = model_output[class_, :] == np.max(
        model_output, axis=0
    )
    return f1_score(
        model_output.ravel(), gt[class_, :].ravel()
    )


def get_unc_utils(
    model_output: np.ndarray,
    gt: np.ndarray
):
    """
    Get uncertainty utils

    :param model_output: 
    """
    # Model mean
    model_mean = model_output.mean(axis=0)

    #Gt as one channel
    gt_one_channel = np.argmax(
        gt, axis=0
    )

    # Compute misclassification
    misclassification = model_mean.argmax(axis=0) != gt_one_channel

    # Transform output in histograms
    bins = np.linspace(0, 1, 101)

    # Compute histogram
    output_hists = np.apply_along_axis(
        lambda a: np.histogram(
            a, bins=bins
        )[0]/100,
        axis=0,
        arr=model_output
    )
    # Order classes
    order_classes = np.apply_along_axis(
        lambda a: np.argsort(
            a
        ),
        axis=0,
        arr=model_mean
    ).reshape((1,)+model_mean.shape)

    # Concatenate arrays
    concatenated_arrays = np.swapaxes(np.concatenate(
        (output_hists, order_classes),
        axis=0
    ), 0, 1).reshape(
        (gt.shape[0] * 101,) +\
        model_mean.mean(axis=0).shape
    )
    return misclassification, concatenated_arrays


def varprauc(model_output, gt):
    return model_output.std(axis=0).mean(axis=0)


def battprauc(
        model_output,
        concatenated_arrays
    ):
    return np.apply_along_axis(
        lambda arr:
        np.sum(
            np.sqrt(
                arr[int(arr[-1]) * 101:int(arr[-1]) * 101 + 100] *\
                arr[int(arr[-102]) * 101:int(arr[-102]) * 101  + 100]
            )
        ),
        arr=concatenated_arrays,
        axis=0
    )


def kl_sym(
    p, q,
    epsilon=0.0000001
):
    p[p==0] = epsilon
    q[q==0] = epsilon
    return 1/2 * (
        entropy(p, q) +\
        entropy(
            q, p
        )
    )


def klprauc(
        model_output,
        concatenated_arrays,
        epsilon=0.0000001
    ):
    return np.apply_along_axis(
        lambda arr:
        -kl_sym(
            arr[int(arr[-1]) * 101:int(arr[-1]) * 101 + 100],
            arr[int(arr[-102]) * 101:int(arr[-102]) * 101  + 100]
        ),
        arr=concatenated_arrays,
        axis=0
    )

def entprauc(
        model_output,
        concatenated_arrays,
        epsilon=0.0000001
    ):
    return np.apply_along_axis(
        lambda arr:
        -np.sum(
            arr[int(arr[-1]) * 101:int(arr[-1]) * 101 + 100] *\
            np.log(
                arr[int(arr[-1]) * 101:int(arr[-1]) * 101 + 100] +\
                epsilon
            ) +\
            arr[int(arr[-102]) * 101:int(arr[-102]) * 101 + 100] *\
            np.log(
                arr[int(arr[-102]) * 101:int(arr[-102]) * 101 + 100] +\
                epsilon
            ) +\
            arr[int(arr[-203]) * 101:int(arr[-203]) * 101 + 100] *\
            np.log(
                arr[int(arr[-203]) * 101:int(arr[-203]) * 101 + 100] +\
                epsilon
            )
        )/3,
        arr=concatenated_arrays,
        axis=0
    )


def eval_unc_map(unc_map, misclassification):
    return average_precision_score(
        misclassification.ravel(),
        unc_map.ravel()
    )


UNCMETRIC = [
    entprauc, varprauc, battprauc, klprauc
]
METRICS = [
    dice
]

def unc(subset, generator, model, path):
    print('- Compute {} on {} : '.format('metrics', subset))
    cumul = {}
    model.n_iter = 50
    display=True
    for input_, gt_, _ in tqdm.tqdm(generator):
        with torch.no_grad():
            torch.manual_seed(5)
            np.random.seed(5)
            model_output = model.mc_forward(input_)[:, 0, :]\
                    .detach().cpu().numpy()
            gt = gt_[0, :].cpu().detach().numpy()
            misclassification, concatenated_arrays = get_unc_utils(
                model_output, gt
            )
            if display:
                nib.save(
                    nib.Nifti1Image(
                        misclassification.astype(float),
                        header=None,
                        affine=None
                    ),
                    os.path.join(
                        path,
                        'misclassification.nii.gz'
                    )
                )
                for i in range(input_.shape[1]):
                    nib.save(
                        nib.Nifti1Image(
                            input_[0, i, :].detach().cpu().numpy(),
                            header=None,
                            affine=None
                        ),
                        os.path.join(
                            path,
                            'input_{}.nii.gz'.format(i)
                        )
                    )

            for metrics in METRICS:
                for class_ in range(model_output.shape[1]):
                    for metric in METRICS:
                        if display:
                            nib.save(
                                nib.Nifti1Image(
                                    model_output[:, class_, :]\
                                            .mean(axis=0),
                                    affine=None,
                                    header=None
                                ),
                                os.path.join(
                                    path,
                                    'class_{}.nii.gz'.format(class_)
                                )
                            )
                        key_ = '{}_{}'.format(metric.__name__, class_)
                        if not key_ in list(cumul):
                            cumul[key_] = []
                        cumul[key_].append(
                            metric(
                                model_output, gt, class_
                            )
                        )
            for metric in UNCMETRIC:
                unc_map = metric(
                    model_output,
                    concatenated_arrays
                )
                if display:
                    nib.save(
                        nib.Nifti1Image(
                            unc_map,
                            header=None,
                            affine=None
                        ),
                        os.path.join(
                            path,
                            '{}.nii.gz'.format(
                                metric.__name__
                            )
                        )
                    )
                if not metric.__name__ in list(cumul):
                    cumul[metric.__name__] = []
                eval_ = eval_unc_map(
                    unc_map,
                    misclassification
                )
                cumul[metric.__name__].append(
                    eval_
                )
            if display:
                with open(
                    os.path.join(path, 'values.json'), 'w'
                ) as handle:
                    json.dump(cumul, handle)
        display=False
    output = {
        '{}_values'.format(k): v
        for k, v in cumul.items()
    }
    return output

