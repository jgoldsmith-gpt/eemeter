from collections import defaultdict
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error

class MovingHourlyAverage(object):
    def __init__(self, modeling_period_interpretation="baseline"):
        self.weekday_hrly_avg = defaultdict(float)
        self.weekend_hrly_avg = defaultdict(float)
        self.modeling_period_interpretation = modeling_period_interpretation
        self.rmse = None

    def fit(self, df):
        if (df.index.freq != 'H'):
            raise ValueError("Index Freq not set to hour")

        weekday_hrly_sum = defaultdict(float)
        weekend_hrly_sum = defaultdict(float)


        weekend_hrly_datapoints = defaultdict(int)
        weekday_hrly_datapoints = defaultdict(int)
        df = df[np.isfinite(df['value'])]



        for index, row in df.iterrows():
            day_of_week, hour = index.dayofweek, index.hour
            if day_of_week < 5:
                weekday_hrly_datapoints[str(hour)] += 1
                weekday_hrly_sum[ str(hour)] += float(row['value'])
            else:
                weekend_hrly_datapoints[str(hour)] += 1
                weekend_hrly_sum[str(hour)] += float(row['value'])


        for key, value in weekend_hrly_sum.items():
            denominator = weekend_hrly_datapoints[key]
            if denominator <= 0:
                self.weekend_hrly_avg[key] = 0.0
            else:
                self.weekend_hrly_avg[key] = value / weekend_hrly_datapoints[key]

        for key, value in weekday_hrly_sum.items():
            denominator = weekday_hrly_datapoints[key]
            if denominator <= 0:
                self.weekday_hrly_avg[key] = 0.0
            else:
                self.weekday_hrly_avg[key] = value / weekday_hrly_datapoints[key]

        pd = self.predict(df)
        rmse = math.sqrt(mean_squared_error(df['value'], pd))
        if len(self.weekday_hrly_avg) == 0 and len(self.weekend_hrly_avg) ==0 :
            raise ValueError("Fit failed Completely")
        self.rmse = rmse

        """
        params = {
            "coefficients": {'Intercept' : 1.0},
            "formula": "Moving Avearge",
            "X_design_info": "Moving average",
        }


        output = {
            "r2": 1.0,
            "model_params": params,
            "rmse": rmse,
            "cvrmse": 1.0,
            "n": len(df),
        }
        #print ("Fit Called******************"), output
        return output
        """

    def predict(self, df, params=None, summed=True):
        '''
        '''
        #if (df.index.freq != 'H'):
        #    raise ValueError("Index Freq not set to hour")

        predicted = []
        for index, row in df.iterrows():
            day_of_week, hour = index.dayofweek, index.hour
            if day_of_week < 5:
                pred_value = self.weekday_hrly_avg[str(hour)]
                predicted.append(pred_value)
            else:
                pred_value = self.weekend_hrly_avg[str(hour)]

                predicted.append(pred_value)


        prediction = pd.Series(predicted, index=df.index)
        return prediction
        """
        variance = pd.Series([1.0 for xx in range(len(predicted))], index=df.index)
        #return prediction, 1.0

        if summed:
            #predicted = prediction.sum()
            return prediction.sum(), 1.0
        else:
            return predicted, 1.0
        """
