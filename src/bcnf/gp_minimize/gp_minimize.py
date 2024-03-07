# type: ignore

import numbers
import warnings

import numpy as np
from sklearn.utils import check_random_state
from skopt.utils import cook_estimator, eval_callbacks, normalize_dimensions

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from skopt.callbacks import VerboseCallback, check_callback
from skopt.optimizer import Optimizer


def gp_minimize_fixed(
        func, dimensions, base_estimator=None,
        n_calls=100, n_random_starts=None,
        n_initial_points=10,
        initial_point_generator="random",
        acq_func="gp_hedge", acq_optimizer="auto", x0=None, y0=None,
        random_state=None, verbose=False, callback=None,
        n_points=10000, n_restarts_optimizer=5, xi=0.01, kappa=1.96,
        noise="gaussian", n_jobs=1, model_queue_size=None):

    # Check params
    rng = check_random_state(random_state)
    space = normalize_dimensions(dimensions)

    if base_estimator is None:
        base_estimator = cook_estimator(
            "GP", space=space, random_state=rng.randint(0, np.iinfo(np.int32).max),
            noise=noise)

    return base_minimize_fixed(
        func, space, base_estimator=base_estimator,
        acq_func=acq_func,
        xi=xi, kappa=kappa, acq_optimizer=acq_optimizer, n_calls=n_calls,
        n_points=n_points, n_random_starts=n_random_starts,
        n_initial_points=n_initial_points,
        initial_point_generator=initial_point_generator,
        n_restarts_optimizer=n_restarts_optimizer,
        x0=x0, y0=y0, random_state=rng, verbose=verbose,
        callback=callback, n_jobs=n_jobs, model_queue_size=model_queue_size)


def base_minimize_fixed(
        func, dimensions, base_estimator,
        n_calls=100, n_random_starts=None,
        n_initial_points=10,
        initial_point_generator="random",
        acq_func="EI", acq_optimizer="lbfgs",
        x0=None, y0=None, random_state=None, verbose=False,
        callback=None, n_points=10000, n_restarts_optimizer=5,
        xi=0.01, kappa=1.96, n_jobs=1, model_queue_size=None):
    specs = {"args": locals(),
             "function": "base_minimize"}

    acq_optimizer_kwargs = {
        "n_points": n_points, "n_restarts_optimizer": n_restarts_optimizer,
        "n_jobs": n_jobs}
    acq_func_kwargs = {"xi": xi, "kappa": kappa}

    # Initialize optimization
    # Suppose there are points provided (x0 and y0), record them

    # check x0: list-like, requirement of minimal points
    if x0 is None:
        x0 = []
    elif not isinstance(x0[0], (list, tuple)):
        x0 = [x0]
    if not isinstance(x0, list):
        raise ValueError("`x0` should be a list, but got %s" % type(x0))

    # Check `n_random_starts` deprecation first
    if n_random_starts is not None:
        warnings.warn(("n_random_starts will be removed in favour of "
                       "n_initial_points. It overwrites n_initial_points."),
                      DeprecationWarning)
        n_initial_points = n_random_starts

    if n_initial_points <= 0 and not x0:
        raise ValueError("Either set `n_initial_points` > 0,"
                         " or provide `x0`")
    # check y0: list-like, requirement of maximal calls
    if isinstance(y0, Iterable):
        y0 = list(y0)
    elif isinstance(y0, numbers.Number):
        y0 = [y0]
    required_calls = n_initial_points + (len(x0) if not y0 else 0)
    if n_calls < required_calls:
        raise ValueError(
            "Expected `n_calls` >= %d, got %d" % (required_calls, n_calls))

    # calculate the total number of initial points
    # HACK: Is this a bug in skopt?
    # If y0 is not provided, treat the x0 as additional initial points
    if not y0:
        n_initial_points = n_initial_points + len(x0)

    print(f'{n_initial_points} initial points will be randomly generated')

    # Build optimizer

    # create optimizer class
    optimizer = Optimizer(dimensions, base_estimator,
                          n_initial_points=n_initial_points,
                          initial_point_generator=initial_point_generator,
                          n_jobs=n_jobs,
                          acq_func=acq_func, acq_optimizer=acq_optimizer,
                          random_state=random_state,
                          model_queue_size=model_queue_size,
                          acq_optimizer_kwargs=acq_optimizer_kwargs,
                          acq_func_kwargs=acq_func_kwargs)
    # check x0: element-wise data type, dimensionality
    assert all(isinstance(p, Iterable) for p in x0)
    if not all(len(p) == optimizer.space.n_dims for p in x0):
        raise RuntimeError("Optimization space (%s) and initial points in x0 "
                           "use inconsistent dimensions." % optimizer.space)
    # check callback
    callbacks = check_callback(callback)
    if verbose:
        callbacks.append(VerboseCallback(
            n_init=len(x0) if not y0 else 0,
            n_random=n_initial_points,
            n_total=n_calls))

    # Record provided points

    # create return object
    result = None
    # evaluate y0 if only x0 is provided
    if x0 and y0 is None:
        y0 = list(map(func, x0))
        n_calls -= len(y0)
    # record through tell function
    if x0:
        if not (isinstance(y0, Iterable) or isinstance(y0, numbers.Number)):
            raise ValueError(
                "`y0` should be an iterable or a scalar, got %s" % type(y0))
        if len(x0) != len(y0):
            raise ValueError("`x0` and `y0` should have the same length")
        print(f'Telling optimizer about {len(x0)} initial points')
        result = optimizer.tell(x0, y0)
        result.specs = specs
        if eval_callbacks(callbacks, result):
            return result

    # Optimize
    for n in range(n_calls):
        next_x = optimizer.ask()
        next_y = func(next_x)
        result = optimizer.tell(next_x, next_y)
        result.specs = specs
        if eval_callbacks(callbacks, result):
            break

    return result
