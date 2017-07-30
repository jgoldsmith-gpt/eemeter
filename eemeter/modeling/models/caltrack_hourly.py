import pandas as pd
import numpy as np
import patsy
import eemeter.modeling.exceptions as model_exceptions
from eemeter.modeling.models.caltrack_helpers import \
    _fit_intercept, _fit_cdd_only, _fit_hdd_only, _fit_full

class CaltrackHourlyModel(object):
    '''
    '''
    def __init__(
            self, fit_cdd=True,
            grid_search=False,
            modeling_period_interpretation='baseline',
            **kwargs):  # ignore extra args

        self.fit_cdd = fit_cdd
        self.grid_search = grid_search
        self.model_freq = pd.tseries.frequencies.Hour()
        self.params = None
        self.X = None
        self.y = None
        self.estimated = None
        self.r2 = None
        self.rmse = None
        self.cvrmse = None
        self.nmbe = None
        self.n = None
        self.input_data = None
        self.fit_bp_hdd, self.fit_bp_cdd = None, None
        self.modeling_period_interpretation = modeling_period_interpretation

        if grid_search:
            self.bp_cdd = range(65, 76)
            self.bp_hdd = range(55, 66)
        else:
            self.bp_cdd, self.bp_hdd = [70, ], [60, ]

    def __repr__(self):
        return 'CaltrackHourly'


    def add_hdd_cdd(self, df):
        """

        Parameters
        ----------
        df : Pandas Dataframe with Columns: tempF and energy, index and row numbers
        as DatetimeIndex.
        EXAMPLE:
                                    energy  temp
        2017-06-01 00:00:00+00:00   582.00  73.4
        2017-06-01 01:00:00+00:00   695.50  71.6
        2017-06-01 02:00:00+00:00   671.00  69.8
        2017-06-01 03:00:00+00:00   685.00  69.8
        2017-06-01 04:00:00+00:00   708.50  71.6
        Returns
        -------

        """
        df = df[~df.index.duplicated(keep='last')].sort_index()

        # Create arrays to hold computed CDD and HDD for each
        # balance point temperature.
        cdd = {i: [0] for i in self.bp_cdd}
        hdd = {i: [0] for i in self.bp_hdd}

        # If there isn't any data, throw an exception
        if len(df.index) == 0:
            raise model_exceptions.DataSufficiencyException("No energy trace data")

        for bp in self.bp_cdd:
            cdd[bp] = pd.Series(
                np.maximum(df.tempF - bp, 0),
                index=df.index)
        for bp in self.bp_hdd:
            hdd[bp] = pd.Series(
                np.maximum(bp - df.tempF, 0),
                index=df.index)

        df_dict = {'upd': df.energy, 'usage': df.energy}
        df_dict.update({'CDD_' + str(bp): cdd[bp] for bp in cdd.keys()})
        df_dict.update({'HDD_' + str(bp): hdd[bp] for bp in hdd.keys()})
        output = pd.DataFrame(df_dict, index=df.index)
        return output


    def fit(self, input_data):

        self.input_data = input_data
        self.df = self.add_hdd_cdd(input_data)
        # Fit the intercept-only model
        (
            int_formula,
            int_mod,
            int_res,
            int_rsquared,
            int_qualified
        ) = _fit_intercept(self.df)

        # CDD-only
        if self.fit_cdd:
            (
                cdd_formula,
                cdd_mod,
                cdd_res,
                cdd_rsquared,
                cdd_qualified,
                cdd_bp
            ) = _fit_cdd_only(self.df)
        else:
            cdd_formula = None
            cdd_mod = None
            cdd_res = None
            cdd_rsquared = 0
            cdd_qualified = False
            cdd_bp = None

        # HDD-only
        (
            hdd_formula,
            hdd_mod,
            hdd_res,
            hdd_rsquared,
            hdd_qualified,
            hdd_bp
        ) = _fit_hdd_only(self.df)

        # CDD+HDD
        if self.fit_cdd:
            (
                full_formula,
                full_mod,
                full_res,
                full_rsquared,
                full_qualified,
                full_hdd_bp,
                full_cdd_bp
            ) = _fit_full(self.df)
        else:
            full_formula = None
            full_mod = None
            full_res = None
            full_rsquared = 0
            full_qualified = False
            full_hdd_bp = None
            full_cdd_bp = None

        # Now we take the best qualified model.
        if (
            full_qualified or
            hdd_qualified or
            cdd_qualified or
            int_qualified
        ) is False:
            raise model_exceptions.ModelFitException(
                "No candidate model fit to data successfully")

        use_full = (full_qualified and (
            full_rsquared > max([
                int(hdd_qualified) * hdd_rsquared,
                int(cdd_qualified) * cdd_rsquared,
                int(int_qualified) * int_rsquared,
            ])
        ))

        use_hdd_only = (hdd_qualified and (
            hdd_rsquared > max([
                int(full_qualified) * full_rsquared,
                int(cdd_qualified) * cdd_rsquared,
                int(int_qualified) * int_rsquared,
            ])
        ))

        use_cdd_only = (cdd_qualified and (
            cdd_rsquared > max([
                int(full_qualified) * full_rsquared,
                int(hdd_qualified) * hdd_rsquared,
                int(int_qualified) * int_rsquared,
            ])
        ))

        fit_bp_hdd, fit_bp_cdd = None, None

        if use_full:
            # Use the full model
            y, X = patsy.dmatrices(
                full_formula, self.df, return_type='dataframe')
            estimated = full_res.fittedvalues
            r2, rmse = full_rsquared, np.sqrt(full_res.ssr/full_res.nobs)
            model_obj, model_res, formula = full_mod, full_res, full_formula
            fit_bp_hdd, fit_bp_cdd = full_hdd_bp, full_cdd_bp

        elif use_hdd_only:
            y, X = patsy.dmatrices(
                hdd_formula, self.df, return_type='dataframe')
            estimated = hdd_res.fittedvalues
            r2, rmse = hdd_rsquared, np.sqrt(hdd_res.ssr/hdd_res.nobs)
            model_obj, model_res, formula = hdd_mod, hdd_res, hdd_formula
            fit_bp_hdd = hdd_bp

        elif use_cdd_only:
            y, X = patsy.dmatrices(
                cdd_formula, self.df, return_type='dataframe')
            estimated = cdd_res.fittedvalues
            r2, rmse = cdd_rsquared, np.sqrt(cdd_res.ssr/cdd_res.nobs)
            model_obj, model_res, formula = cdd_mod, cdd_res, cdd_formula
            fit_bp_cdd = cdd_bp

        else:
            # Use Intercept-only
            y, X = patsy.dmatrices(
                int_formula, self.df, return_type='dataframe')
            estimated = int_res.fittedvalues
            r2, rmse = int_rsquared, np.sqrt(int_res.ssr/int_res.nobs)
            model_obj, model_res, formula = int_mod, int_res, int_formula

        if y.mean != 0:
            cvrmse = rmse / float(y.values.ravel().mean())
            nmbe = np.nanmean(model_res.resid) / float(y.values.ravel().mean())
        else:
            cvrmse = np.nan
            nmbe = np.nan

        n = estimated.shape[0]

        self.y, self.X = y, X
        self.estimated = estimated
        self.r2, self.rmse = r2, rmse
        self.model_obj, self.model_res, self.formula = model_obj, model_res, formula
        self.cvrmse = cvrmse
        self.nmbe = nmbe
        self.fit_bp_hdd, self.fit_bp_cdd = fit_bp_hdd, fit_bp_cdd
        self.n = n
        self.params = {
            "coefficients": self.model_res.params.to_dict(),
            "formula": self.formula,
            "cdd_bp": self.fit_bp_cdd,
            "hdd_bp": self.fit_bp_hdd,
            "X_design_info": self.X.design_info,
        }

        output = {
            "r2": self.r2,
            "model_params": self.params,
            "rmse": self.rmse,
            "cvrmse": self.cvrmse,
            "nmbe": self.nmbe,
            "n": self.n,
        }
        return output
