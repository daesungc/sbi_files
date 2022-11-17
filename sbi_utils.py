import dill as pickle


def load_pr(pr_name):
    """
    Load prior/posterior (hence p*r but can't have * in python name ;( )

    Parameters
    ----------
    pr_name : str
            File name of the prior/posterior including leading path if applicable.

    Returns
    -------
            sbi NeuralPosterior or RestrictedPrior
            Parameter space object. Can be posterior or RestrictedPrior depending on the file.
    """
    with open(pr_name, "rb") as f:
        posterior = pickle.load(f)

    return posterior


def save_pr(pr, savename):
    """
    Save prior/posterior (hence p*r but can't have * in python name ;( )

    Parameters
    ----------
    pr : sbi NeuralPosterior or RestrictedPrior
            Parameter space object such as the NeuralPosterior or RestrictedPrior.
            (Technically it's just a generic pickle wrapper)
    savename : str
            File name to save including leading path if applicable.
    """
    with open(savename, "wb") as handle:
        pickle.dump(pr, handle)
