"""Functions for providing the over-arching methodology. Tying together the
individual components that make-up the change detection process. This module
should really contain any method that could be considered procedural. Methods
must accept the processing parameters, then use those values for the more
functional methods that they call. The hope is that this will eventually get
converted more and more away from procedural and move more towards the
functional paradigm.

Any methods determined by the fit_procedure call must accept same 5 arguments,
in the same order: dates, observations, fitter_fn, quality, proc_params.

The results of this process is a list-of-lists of change models that correspond
to observation spectra. A processing mask is also returned, outlining which
observations were utilized and which were not.

Pre-processing routines are essential to, but distinct from, the core change
detection algorithm. See the `ccd.qa` for more details related to this
step.

For more information please refer to the pyccd Algorithm Description Document.

.. _Algorithm Description Document:
   https://drive.google.com/drive/folders/0BzELHvbrg1pDREJlTF8xOHBZbEU
"""
import logging
import numpy as np

from ccd import qa
from ccd.change import enough_samples, enough_time,\
    update_processing_mask, stable, determine_num_coefs, calc_residuals, \
    find_closest_doy, change_magnitude, detect_change, detect_outlier, \
    adjustpeek, adjustchgthresh, returnThresholdFromProb
from ccd.models import results_to_changemodel, tmask
from ccd.math_utils import adjusted_variogram, euclidean_norm

log = logging.getLogger(__name__)


def standard_procedure(dates, observations, fitter_fn, proc_params):
    """
    Runs the core change detection algorithm.

    Step 1: initialize -- Find an initial stable time-frame to build from.

    Step 2: lookback -- The initlize step may have iterated the start of the
    model past the previous break point. If so then we need too look back at
    previous values to see if they can be included within the new
    initialized model.

    Step 3: catch -- Fit a general model to values that may have been skipped
    over by the previous steps.

    Step 4: lookforward -- Expand the time-frame until a change is detected.

    Step 5: Iterate.

    Step 6: catch -- End of time series considerations.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: 2-d array of observed spectral values corresponding
            to each time.
        fitter_fn: a function used to fit observation values and
            acquisition dates for each spectra.

        proc_params: dictionary of processing parameters

    Returns:
        list: Change models for each observation of each spectra.
        1-d ndarray: processing mask indicating which values were used
            for model fitting
    """
    # TODO do this better
    meow_size = proc_params.MEOW_SIZE
    defpeek = proc_params.PEEK_SIZE

    log.debug('Build change models - dates: %s, obs: %s, '
              'meow_size: %s, peek_size: %s',
              dates.shape[0], observations.shape, meow_size, defpeek)

    processing_mask = np.ones(dates.shape[0], dtype=bool)

    obs_count = np.sum(processing_mask)

    log.debug('Processing mask initial count: %s', obs_count)

    # Accumulator for models. This is a list of ChangeModel named tuples
    results = []

    if obs_count <= meow_size:
        return results, processing_mask

    # TODO Temporary setup on this to just get it going
    peek_size = adjustpeek(dates[processing_mask], defpeek)
    proc_params.PEEK_SIZE = peek_size
    proc_params.CHANGE_THRESHOLD = returnThresholdFromProb(proc_params.CHISQUAREPROB, len(proc_params.DETECTION_BANDS))
    proc_params.CHANGE_THRESHOLD = adjustchgthresh(peek_size, defpeek,
                                                    proc_params.CHANGE_THRESHOLD, proc_params.CHISQUAREPROB, len(proc_params.DETECTION_BANDS))

    log.debug('Peek size: %s', proc_params.PEEK_SIZE)
    log.debug('Chng thresh: %s', proc_params.CHANGE_THRESHOLD)

    # Compute and store outlier threshold
    proc_params.OUTLIER_THRESHOLD = returnThresholdFromProb(0.999999, len(proc_params.DETECTION_BANDS))

    # Initialize the window which is used for building the models
    model_window = slice(0, meow_size)
    previous_end = 0

    # Only capture general curve at the beginning, and not in the middle of
    # two stable time segments
    start = True

    # Calculate the variogram/madogram that will be used in subsequent
    # processing steps. See algorithm documentation for further information.
    variogram = adjusted_variogram(dates[processing_mask],
                                    observations[:, processing_mask])
    log.debug('Variogram values: %s', variogram)

    # Only build models as long as sufficient data exists.
    while model_window.stop <= dates[processing_mask].shape[0] - meow_size:
        # Step 1: Initialize
        log.debug('Initialize for change model #: %s', len(results) + 1)
        if len(results) > 0:
            start = False

        # Make things a little more readable by breaking this apart
        # catch return -> break apart into components
        initialized = initialize(dates, observations, fitter_fn, model_window,
                                  processing_mask, variogram, proc_params)

        model_window, init_models, processing_mask = initialized
        
        # print('After Initialization - Processing Mask:', processing_mask)

        # Catch for failure
        if init_models is None:
            log.debug('Model initialization failed')
            break

        # Step 2: Lookback
        if model_window.start > previous_end:
            lb = lookback(dates, observations, model_window, init_models,
                          previous_end, processing_mask, variogram, proc_params)

            model_window, processing_mask = lb
            
            # print('After Lookback - Processing Mask:', processing_mask)

        # Step 3: catch
        # If we have moved > peek_size from the previous break point
        # then we fit a generalized model to those points.
        if model_window.start - previous_end > peek_size and start is True:
            results.append(catch(dates,
                                  observations,
                                  fitter_fn,
                                  processing_mask,
                                  slice(previous_end, model_window.start),
                                  proc_params))
            start = False

        # Handle specific case where if we are at the end of a time series and
        # the peek size is greater than what remains of the data.
        if model_window.stop + peek_size > dates[processing_mask].shape[0]:
            break

        # Step 4: lookforward
        log.debug('Extend change model')
        lf = lookforward(dates, observations, model_window, fitter_fn,
                          processing_mask, variogram, proc_params)

        result, processing_mask, model_window = lf
        results.append(result)
        
        # print('After Lookforward - Processing Mask:', processing_mask)

        log.debug('Accumulate results, {} so far'.format(len(results)))

        # Step 5: Iterate
        previous_end = model_window.stop
        model_window = slice(model_window.stop, model_window.stop + meow_size)

    # Step 6: Catch
    # We can use previous start here as that value should be equal to
    # model_window.stop due to the constraints on the the previous while
    # loop.
    if previous_end + peek_size < dates[processing_mask].shape[0]:
        model_window = slice(previous_end, dates[processing_mask].shape[0])
        results.append(catch(dates, observations, fitter_fn,
        processing_mask, model_window,
        proc_params=proc_params))
    
    # print('After Final Catch - Processing Mask:', processing_mask)

    log.debug("change detection complete")

    return results, processing_mask


def initialize(dates, observations, fitter_fn, model_window, processing_mask,
               variogram, proc_params):
    """
    Determine a good starting point at which to build off of for the
    subsequent process of change detection, both forward and backward.

    Args:
        dates: 1-d ndarray of ordinal day values
        observations: 2-d ndarray representing the spectral values
        fitter_fn: function used for the regression portion of the algorithm
        model_window: start index of time/observation window
        processing_mask: 1-d boolean array identifying which values to
            consider for processing
        variogram: 1-d array of variogram values to compare against for the
            normalization factor
        proc_params: dictionary of processing parameters

    Returns:
        slice: model window that was deemed to be a stable start
        namedtuple: fitted regression models
    """
    # TODO do this better
    meow_size = proc_params.MEOW_SIZE
    day_delta = proc_params.DAY_DELTA
    detection_bands = proc_params.DETECTION_BANDS
    tmask_bands = proc_params.TMASK_BANDS
    change_thresh = proc_params.CHANGE_THRESHOLD
    tmask_scale = proc_params.T_CONST
    avg_days_yr = proc_params.AVG_DAYS_YR
    fit_max_iter = proc_params.LASSO_MAX_ITER
    alpha = proc_params.ALPHA

    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    log.debug('Initial %s', model_window)
    models = None
    while model_window.stop + meow_size < period.shape[0]:
        # Finding a sufficient window of time needs to run
        # each iteration because the starting point
        # will increment if the model isn't stable, incrementing only
        # the window stop in lock-step does not guarantee a 1-year+
        # time-range.
        if not enough_time(period[model_window], day_delta):
            model_window = slice(model_window.start, model_window.stop + 1)
            continue
        # stop = find_time_index(dates, model_window, meow_size, day_delta)
        # model_window = slice(model_window.start, stop)
        log.debug('Checking window: %s', model_window)

        # Count outliers in the window, if there are too many outliers then
        # try again.
        tmask_outliers = tmask.tmask(period[model_window],
                                     spectral_obs[:, model_window],
                                     variogram, tmask_bands, tmask_scale,
                                     avg_days_yr)

        tmask_count = np.sum(tmask_outliers)

        log.debug('Number of Tmask outliers found: %s', tmask_count)

        # Subset the data to the observations that currently under scrutiny
        # and remove the outliers identified by the tmask.
        tmask_period = period[model_window][~tmask_outliers]

        # TODO should probably look at a different fit procedure to handle
        # the following case.
        if tmask_count == model_window.stop - model_window.start:
            log.debug('Tmask identified all values as outliers')

            model_window = slice(model_window.start, model_window.stop + 1)
            continue

        # Make sure we still have enough observations and enough time after
        # the tmask removal.
        if not enough_time(tmask_period, day_delta) or \
                not enough_samples(tmask_period, meow_size):

            log.debug('Insufficient time or observations after Tmask, '
                      'extending model window')

            model_window = slice(model_window.start, model_window.stop + 1)
            continue

        # Update the persistent mask with the values identified by the Tmask
        if any(tmask_outliers):
            processing_mask = update_processing_mask(processing_mask,
                                                     tmask_outliers,
                                                     model_window)

            # The model window now actually refers to a smaller slice
            model_window = slice(model_window.start,
                                 model_window.stop - tmask_count)
            # Update the subset
            period = dates[processing_mask]
            spectral_obs = observations[:, processing_mask]

        log.debug('Generating models to check for stability')
        models = [fitter_fn(period[model_window], spectrum,
                            fit_max_iter, avg_days_yr, 4, alpha)
                  for spectrum in spectral_obs[:, model_window]]

        # If a model is not stable, then it is possible that a disturbance
        # exists somewhere in the observation window. The window shifts
        # forward in time, and begins initialization again.
        if not stable(models, period[model_window], variogram,
                      change_thresh, detection_bands):

            model_window = slice(model_window.start + 1, model_window.stop + 1)
            
            # remove a sobreposição de curvas mas estraga os outros resultados:
            # model_window = slice(model_window.stop, model_window.stop + meow_size)
            
            log.debug('Unstable model, shift window to: %s', model_window)
            models = None
            continue

        else:
            log.debug('Stable start found: %s', model_window)
            break

    return model_window, models, processing_mask


def lookforward(dates, observations, model_window, fitter_fn, processing_mask,
                variogram, proc_params):
    """Increase observation window until change is detected or
    we are out of observations.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: spectral values, list of spectra -> values
        model_window: span of indices that is represented in the current
            process
        fitter_fn: function used to model observations
        processing_mask: 1-d boolean array identifying which values to
            consider for processing
        variogram: 1-d array of variogram values to compare against for the
            normalization factor
        proc_params: dictionary of processing parameters

    Returns:
        namedtuple: representation of the time segment
        1-d bool ndarray: processing mask that may have been modified
        slice: model window
    """
    # TODO do this better
    peek_size = proc_params.PEEK_SIZE
    coef_min = proc_params.COEFFICIENT_MIN
    coef_mid = proc_params.COEFFICIENT_MID
    coef_max = proc_params.COEFFICIENT_MAX
    num_obs_fact = proc_params.NUM_OBS_FACTOR
    detection_bands = proc_params.DETECTION_BANDS
    change_thresh = proc_params.CHANGE_THRESHOLD
    outlier_thresh = proc_params.OUTLIER_THRESHOLD
    avg_days_yr = proc_params.AVG_DAYS_YR
    fit_max_iter = proc_params.LASSO_MAX_ITER
    alpha=proc_params.ALPHA
    min_years=proc_params.MIN_YEARS

    # Step 4: lookforward.
    # The second step is to update a model until observations that do not
    # fit the model are found.
    log.debug('lookforward initial model window: %s', model_window)

    # The fit_window pertains to which locations are used in the model
    # regression, while the model_window identifies the locations in which
    # fitted models apply to. They are not always the same.
    fit_window = model_window

    # Initialized for a check at the first iteration.
    models = None

    # Simple value to determine if change has occured or not. Change may not
    # have occurred if we reach the end of the time series.
    change = 0

    # Initial subset of the data
    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    # Used for comparison purposes
    fit_span = period[model_window.stop - 1] - period[model_window.start]
    
    #print('antes do while model_window.stop:',model_window.stop)
    #print('antes do while peek size:', peek_size)
    #print('antes do while period shape:', period.shape[0])

    # stop is always exclusive
    
    # while model_window.stop + peek_size <= period.shape[0]:
    while model_window.stop < period.shape[0]:
        #print('model_window.stop:',model_window.stop)
        #print('peek size:', peek_size)
        #print('period shape:', period.shape[0])
        num_coefs = determine_num_coefs(period[model_window], coef_min,
                                        coef_mid, coef_max, num_obs_fact)

        peek_window = slice(model_window.stop, model_window.stop + peek_size)

        # Used for comparison against fit_span
        model_span = period[model_window.stop - 1] - period[model_window.start]

        log.debug('Detecting change for %s', peek_window)

        # If we have less than 24 observations covered by the model_window
        # or it the first iteration, then we always fit a new window
        # If the number of observations that the current fitted models
        # expand past a threshold, then we need to fit new ones.
        if not models or model_window.stop - model_window.start < 24 or model_span >= min_years * fit_span:
            fit_span = period[model_window.stop - 1] - period[
                model_window.start]

            fit_window = model_window
            log.debug('Retrain models')
            models = [fitter_fn(period[fit_window], spectrum,
                                fit_max_iter, avg_days_yr, num_coefs, alpha)
                      for spectrum in spectral_obs[:, fit_window]]

        residuals = np.array([calc_residuals(period[peek_window],
                                             spectral_obs[idx, peek_window],
                                             models[idx], avg_days_yr)
                              for idx in range(observations.shape[0])])

        if model_window.stop - model_window.start <= 24:
            comp_rmse = [models[idx].rmse for idx in detection_bands]

        # More than 24 points
        else:
            # We want to use the closest residual values to the peek_window
            # values based on seasonality.
            
            # closest_indexes = find_closest_doy(period, peek_window.stop - 1,
            #                                    fit_window, 24)

            closest_indexes = find_closest_doy(period, min(peek_window.stop - 1, len(dates) - 1), 
                                               fit_window, 24)

            # Calculate an RMSE for the seasonal residual values, using 8
            # as the degrees of freedom.
            comp_rmse = [euclidean_norm(models[idx].residual[closest_indexes]) / 4
                         for idx in detection_bands]

        # Calculate the change magnitude values for each observation in the
        # peek_window.
        magnitude = change_magnitude(residuals[detection_bands, :],
                                     variogram[detection_bands],
                                     comp_rmse)

        if detect_change(magnitude, change_thresh):
            log.debug('Change detected at: %s', peek_window.start)

            # Change was detected, return to parent method
            change = 1
            break
        elif detect_outlier(magnitude[0], outlier_thresh):
            log.debug('Outlier detected at: %s', peek_window.start)

            # Keep track of any outliers so they will be excluded from future
            # processing steps
            processing_mask = update_processing_mask(processing_mask,
                                                     peek_window.start)

            # Because only one value was excluded, we shouldn't need to adjust
            # the model_window.  The location hasn't been used in
            # processing yet. So, the next iteration can use the same windows
            # without issue.
            period = dates[processing_mask]
            spectral_obs = observations[:, processing_mask]
            continue

        # Check before incrementing the model window, otherwise the reporting
        # can get a little messy.
        
        # if model_window.stop + peek_size > period.shape[0]:
        #     print('Condição de parada ativada: model_window.stop + peek_size > period.shape[0]')
        #     break

        model_window = slice(model_window.start, model_window.stop + 1)

    result = results_to_changemodel(fitted_models=models,
                                start_day=period[model_window.start],
                                end_day=period[model_window.stop - 1],
                                break_day=period[peek_window.start],
                                magnitudes=np.median(residuals, axis=1),
                                observation_count=(
                                    model_window.stop - model_window.start),
                                change_probability=change)

    return result, processing_mask, model_window


def lookback(dates, observations, model_window, models, previous_break,
             processing_mask, variogram, proc_params):
    """
    Special case when there is a gap between the start of a time series model
    and the previous model break point, this can include values that were
    excluded during the initialization step.

    Args:
        dates: list of ordinal days
        observations: spectral values across bands
        model_window: current window of values that is being considered
        models: currently fitted models for the model_window
        previous_break: index value of the previous break point, or the start
            of the time series if there wasn't one
        processing_mask: index values that are currently being masked out from
            processing
        variogram: 1-d array of variogram values to compare against for the
            normalization factor
        proc_params: dictionary of processing parameters

    Returns:
        slice: window of indices to be used
        array: indices of data that have been flagged as outliers
    """
    # TODO do this better
    peek_size = proc_params.PEEK_SIZE
    detection_bands = proc_params.DETECTION_BANDS
    change_thresh = proc_params.CHANGE_THRESHOLD
    outlier_thresh = proc_params.OUTLIER_THRESHOLD
    avg_days_yr = proc_params.AVG_DAYS_YR
    #alpha = proc_params.ALPHA

    log.debug('Previous break: %s model window: %s', previous_break, model_window)
    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    while model_window.start > previous_break:
        # Three conditions to see how far we want to look back each iteration.
        # 1. If we have more than 6 previous observations
        # 2. Catch to make sure we don't go past the start of observations
        # 3. Less than 6 observations to look at

        # Important note about python slice objects, start is inclusive and
        # stop is exclusive, regardless of direction/step
        if model_window.start - previous_break > peek_size:
            peek_window = slice(model_window.start - 1, model_window.start - peek_size, -1)
        elif model_window.start - peek_size <= 0:
            peek_window = slice(model_window.start - 1, None, -1)
        else:
            peek_window = slice(model_window.start - 1, previous_break - 1, -1)

        log.debug('Considering index: %s using peek window: %s',
                  peek_window.start, peek_window)

        residuals = np.array([calc_residuals(period[peek_window],
                                             spectral_obs[idx, peek_window],
                                             models[idx], avg_days_yr)
                              for idx in range(observations.shape[0])])

        # log.debug('Residuals for peek window: %s', residuals)

        comp_rmse = [models[idx].rmse for idx in detection_bands]

        log.debug('RMSE values for comparison: %s', comp_rmse)

        magnitude = change_magnitude(residuals[detection_bands, :],
                                     variogram[detection_bands],
                                     comp_rmse)

        if detect_change(magnitude, change_thresh):
            log.debug('Change detected for index: %s', peek_window.start)
            # change was detected, return to parent method
            break
        elif detect_outlier(magnitude[0], outlier_thresh):
            log.debug('Outlier detected for index: %s', peek_window.start)
            processing_mask = update_processing_mask(processing_mask,
                                                     peek_window.start)

            period = dates[processing_mask]
            spectral_obs = observations[:, processing_mask]

            # Because this location was used in determining the model_window
            # passed in, we must now account for removing it.
            model_window = slice(model_window.start - 1, model_window.stop - 1)
            continue

        log.debug('Including index: %s', peek_window.start)
        model_window = slice(peek_window.start, model_window.stop)

    return model_window, processing_mask


def catch(dates, observations, fitter_fn, processing_mask, model_window, proc_params):
    """
    Handle special cases where general models just need to be fitted and return
    their results.

    Args:
        dates: list of ordinal day numbers relative to some epoch,
            the particular epoch does not matter.
        observations: spectral values, list of spectra -> values
        model_window: span of indices that is represented in the current
            process
        fitter_fn: function used to model observations
        processing_mask: 1-d boolean array identifying which values to
            consider for processing

    Returns:
        namedtuple representing the time segment

    """
    log.debug('Fitting catch model')

    # if you want to change the `catch` implementation, modify it here
    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]
    model_period = period[model_window]
    model_spectral = spectral_obs[:, model_window]

    # Find indices where observations are equal to 65535
    # invalid_indices = np.where(observations == 65535)

    # # Ensure that invalid indices are within the valid range of dates
    # invalid_indices = invalid_indices[0][(invalid_indices[0] >= 0) & (invalid_indices[0] < len(dates))]

    # Create a boolean mask for invalid indices
    # invalid_mask = np.zeros_like(processing_mask, dtype=bool)
    # invalid_mask[invalid_indices] = True

    # Update processing_mask based on invalid_mask
    # processing_mask &= ~invalid_mask

    if np.all(processing_mask == False):
        return None
    
    # TODO do this better
    avg_days_yr = proc_params.AVG_DAYS_YR
    fit_max_iter = proc_params.LASSO_MAX_ITER
    num_coef = proc_params.COEFFICIENT_MIN
    alpha = proc_params.ALPHA


    log.debug('Catching observations: %s', model_window)
    period = dates[processing_mask]
    spectral_obs = observations[:,processing_mask]

    # Subset the data based on the model window
    model_period = period[model_window]
    model_spectral = spectral_obs[:, model_window]

    #print(alpha,'alpha')

    models = [fitter_fn(model_period, spectrum, fit_max_iter, avg_days_yr,num_coef, alpha)
              for spectrum in model_spectral]
    
    if model_window.stop >= period.shape[0]:
        break_day = period[-1]
    else:
        break_day = period[model_window.stop]
    
    #print("Model Window Start:", model_window.start)
    #print("Model Window Stop:", model_window.stop)
    #print("Period Shape:", period.shape[0])


    result = results_to_changemodel(
    fitted_models=models,
    start_day=period[model_window.start],
    end_day=period[model_window.stop-1],
    break_day=break_day,
    magnitudes=np.zeros(shape=(6,)),
    observation_count=(model_window.stop - model_window.start),
    change_probability=0
    )

    return result