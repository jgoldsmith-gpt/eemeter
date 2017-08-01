from collections import defaultdict
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error

class MovingHourlyAverage(object):
    def __init__(self):
        self.weekday_hrly_avg = defaultdict(float)
        self.weekend_hrly_avg = defaultdict(float)

    def fit(self, df):
        if (df.index.freq != 'H'):
            raise ValueError("Index Freq not set to hour")

        weekday_hrly_sum = defaultdict(float)
        weekend_hrly_sum = defaultdict(float)

        weekend_hrly_datapoints = defaultdict(int)
        weekday_hrly_datapoints = defaultdict(int)

        for index, row in df.iterrows():
            day_of_week, hour = index.dayofweek, index.hour
            if day_of_week < 5:
                weekday_hrly_datapoints[str(hour)] += 1
                weekday_hrly_sum[str(hour)] += float(row['energy'])
            else:
                weekend_hrly_datapoints[str(hour)] += 1
                weekend_hrly_sum[str(hour)] += float(row['energy'])

        for key, value in weekend_hrly_sum.items():
            self.weekend_hrly_avg[key] = value / weekend_hrly_datapoints[key]

        for key, value in weekday_hrly_sum.items():
            self.weekday_hrly_avg[key] = value / weekday_hrly_datapoints[key]

        pd, var = self.predict(df)
        rmse = math.sqrt(mean_squared_error(df['energy'], pd))
        output = {
            "r2": 1.0,
            "model_params": {},
            "rmse": rmse,
            "cvrmse": 1.0,
            "n": len(df),
        }
        return output


    def predict(self, df, params=None, summed=True):
        '''
        '''
        if (df.index.freq != 'H'):
            raise ValueError("Index Freq not set to hour")

        predicted = []
        for index, row in df.iterrows():
            day_of_week, hour = index.dayofweek, index.hour
            if day_of_week < 5:
                predicted.append(self.weekday_hrly_avg[str(hour)])
            else:
                predicted.append(self.weekday_hrly_avg[str(hour)])


        prediction = pd.Series(predicted, index=df.index)
        variance = np.var(prediction)
        return predicted, variance