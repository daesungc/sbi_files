import os
from pathlib import Path

import torch
from sbi.inference import SNPE, prepare_for_sbi
from sbi.utils import RestrictedPrior, get_density_thresholder, get_nn_models

import sbi_files.sbi_utils as sbi_utils


def density_estimator_build_fun(hidden_features=50, num_transforms=5):
    """
    Wrapper to build a density estimator for SNPE.
    See https://www.mackelab.org/sbi/reference/#sbi.utils.get_nn_models.posterior_nn for original function and documentation.

    Parameters
    ----------
    hidden_features : int, optional
        Number of hidden features, by default 50
    num_transforms : int, optional
        Number of transforms when a flow is used, by default 5

    """
    return get_nn_models.posterior_nn(
        model="maf", hidden_features=hidden_features, num_transforms=num_transforms
    )


def setup_folders(training_loc):
    # Set up training folder
    Path(training_loc + "log/").mkdir(parents=True, exist_ok=True)
    Path(training_loc + "simdat/").mkdir(parents=True, exist_ok=True)
    Path(training_loc + "posteriors/").mkdir(parents=True, exist_ok=True)
    Path(training_loc + "priors/").mkdir(parents=True, exist_ok=True)


def train_tsnpe(
    simulator, prior, training_loc, x_o, rounds=10, num_sims=10_000, summary_writer=None
):
    setup_folders(training_loc)

    simulator, prior = prepare_for_sbi(simulator, prior)

    inference = SNPE(prior, logging_level="INFO", summary_writer=summary_writer)

    proposal = prior
    for r in range(rounds):
        if Path(training_loc + "priors/prior_round%i.pkl" % r).is_file():
            pass
        else:
            data_path = training_loc + "simdat/round%i.pt" % r
            if Path(data_path).is_file():
                theta, x = torch.load(data_path)
            else:
                theta = proposal.sample((num_sims,))
                x = simulator(theta)
                torch.save((theta, x), data_path)

            _ = inference.append_simulations(theta, x).train(
                force_first_round_loss=True
            )
            posterior = inference.build_posterior().set_default_x(x_o)
            sbi_utils.save_pr(
                posterior, training_loc + "posteriors/posterior_round%i.pkl" % r
            )

            accept_reject_fn = get_density_thresholder(posterior, quantile=1e-4)
            proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")
            sbi_utils.save_pr(proposal, training_loc + "priors/prior_round%i.pkl" % r)
