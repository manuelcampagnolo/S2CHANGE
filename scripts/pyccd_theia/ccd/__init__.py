import time
import logging

# from ccd.procedures import fit_procedure as __determine_fit_procedure
from ccd.procedures import standard_procedure
import numpy as np
from ccd import app, math_utils, qa
import importlib
from ccd.version import __version__
from ccd.version import __algorithm__ as algorithm
from ccd.version import __name

log = logging.getLogger(__name)


def attr_from_str(value):
    """Returns a reference to the full qualified function, attribute or class.

    Args:
        value = Fully qualified path (e.g. 'ccd.models.lasso.fitted_model')

    Returns:
        A reference to the target attribute (e.g. fitted_model)
    """
    module, target = value.rsplit('.', 1)
    try:
        obj = importlib.import_module(module)
        return getattr(obj, target)
    except (ImportError, AttributeError) as e:
        log.debug(e)
        return None


def __attach_metadata(procedure_results):
    """
    Attach some information on the algorithm version, what procedure was used,
    and which inputs were used

    Returns:
        A dict representing the change detection results

    {algorithm: 'pyccd:x.x.x',
     processing_mask: (bool, bool, ...),
     snow_prob: float,
     water_prob: float,
     cloud_prob: float,
     change_models: [
         {start_day: int,
          end_day: int,
          break_day: int,
          observation_count: int,
          change_probability: float,
          curve_qa: int,
          ndvi:      {magnitude: float,
                     rmse: float,
                     coefficients: (float, float, ...),
                     intercept: float},
          green:    {magnitude: float,
                     rmse: float,
                     coefficients: (float, float, ...),
                     intercept: float},
          red:     {magnitude: float,
                     rmse: float,
                     coefficients: (float, float, ...),
                     intercept: float},
          nir:      {magnitude: float,
                     rmse: float,
                     coefficients: (float, float, ...),
                     intercept: float},
          swir1:    {magnitude: float,
                     rmse: float,
                     coefficients: (float, float, ...),
                     intercept: float},
          swir2:    {magnitude: float,
                     rmse: float,
                     coefficients: (float, float, ...),
                     intercept: float},
          thermal:  {magnitude: float,
                     rmse: float,
                     coefficients: (float, float, ...),
                     intercept: float}}
                    ]
    }
    """
    change_models, processing_mask = procedure_results

    return {'algorithm': algorithm,
            'processing_mask': [int(_) for _ in processing_mask],
            'change_models': change_models}
            # 'cloud_prob': probs[0],
            # 'snow_prob': probs[1],
            # 'water_prob': probs[2]}


def __split_dates_spectra(matrix):
    """ Slice the dates and spectra from the matrix and return """
    return matrix[0], matrix[1:6]


def __sort_dates(dates):
    """ Sort the values chronologically """
    return np.argsort(dates)


def __check_inputs(dates, spectra):
    """
    Make sure the inputs are of the correct relative size to each-other.
    
    Args:
        dates: 1-d ndarray
        quality: 1-d ndarray
        spectra: 2-d ndarray
    """
    # Make sure we only have one dimension
    assert dates.ndim == 1
    # Make sure quality is the same
    # assert dates.shape == quality.shape
    # Make sure there is spectral data for each date
    assert dates.shape[0] == spectra.shape[1]

# def detect(dates, ndvis, greens, reds, nirs, swir1s, swir2s, params=None):
def detect(dates, ndvis, greens, swir2s, params=None):
    """Entry point call to detect change

    No filtering up-front as different procedures may do things
    differently

    Args:
        dates:    1d-array or list of ordinal date values
        ndvis:    1d-array or list of ndvis  values
        greens:   1d-array or list of green band values
        reds:     1d-array or list of red band values
        nirs:     1d-array or list of nir band values
        swir1s:   1d-array or list of swir1 band values
        swir2s:   1d-array or list of swir2 band values
 
        params: python dictionary to change module wide processing
            parameters

    Returns:
        Tuple of ccd.detections namedtuples
    """
    t1 = time.time()

    proc_params = app.get_default_params()

    if params:
        proc_params.update(params)

    dates = np.asarray(dates)
    
    # spectra = np.stack((ndvis, greens, reds,  nirs, swir1s, swir2s))

    spectra = np.stack((ndvis, greens, swir2s))

    __check_inputs(dates, spectra)

    indices = __sort_dates(dates)
    dates = dates[indices]
    spectra = spectra[:, indices]

    # load the fitter_fn
    fitter_fn = attr_from_str(proc_params.FITTER_FN)

    results = standard_procedure(dates, spectra, fitter_fn, proc_params)
    log.debug('Total time for algorithm: %s', time.time() - t1)

    # call detect and return results as the detections namedtuple
    return __attach_metadata(results)