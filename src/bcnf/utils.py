import os


def get_dir(*args: str, create: bool = False) -> str:
    """
    Get the path to the data directory.

    Parameters
    ----------
    args : str
        The path to the data directory.
    create : bool, optional
        Whether to create the directory if it does not exist, by default False.

    Returns
    -------
    str
        The path to the data directory.
    """
    if any([not isinstance(arg, str) for arg in args]):
        raise TypeError("All arguments must be strings.")

    if create:
        os.makedirs(os.path.join(os.path.dirname(__file__), '..', '..', *args), exist_ok=True)

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', *args))
