import numpy as np

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from memoized_property import memoized_property

from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer

class Trainer():

    MLFLOW_URI = "https://mlflow.lewagon.ai/"
    EXPERIMENT_NAME = "[REMOTE] [reallylongaddress] TaxiFareModel + 0.0.2"

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def set_pipeline(self, estimator='LinearRegression'):
        """defines the pipeline as a class attribute"""
        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        # Add the model of your choice to the pipeline
        if estimator == 'LinearRegression':
            pipeline = Pipeline([
                ('preproc', preproc_pipe),
                ('model', LinearRegression())
            ])
        elif estimator == 'KNN':
            pipeline = Pipeline([
                ('preproc', preproc_pipe),
                ('model', KNeighborsRegressor())
            ])
        elif estimator == 'Lasso':
            pipeline = Pipeline([
                ('preproc', preproc_pipe),
                ('model', Lasso())
            ])
        elif estimator == 'Ridge':
            pipeline = Pipeline([
                ('preproc', preproc_pipe),
                ('model', Ridge())
            ])
        elif estimator == 'SGD':
            pipeline = Pipeline([
                ('preproc', preproc_pipe),
                ('model', SGDRegressor())
            ])

        else:
            raise Exception("unknown model type")

        self.mlflow_log_param('model', estimator)

        return pipeline

    def run(self, estimator = 'LinearRegression'):
        """set and train the pipeline"""
        #values LinearRegression, KNN, Lasso, Ridge, SGD
        self.pipeline = self.set_pipeline(estimator=estimator)
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric('rmse', rmse)
        return rmse

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        save_result = joblib.dump(self.pipeline, './model.joblib')
        print(f'save_result: {save_result}')

if __name__ == "__main__":
    #load data
    df = get_data(10000)
    print(f'df.shape: {df.shape}')

    # clean data
    df = clean_data(df)

    # set X and y
    X = df.drop(columns=['fare_amount'])
    y = df['fare_amount']

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.1)

    # train
    estimators = ['LinearRegression', 'KNN', 'Lasso', 'Ridge', 'SGD']
    for estimator in estimators:
        print(f'running: {estimator}')
        trainer = Trainer(X_train, y_train)
        trainer.run(estimator)

        # evaluate
        rmse = trainer.evaluate(X_test, y_test)
        print(f'rmse: {rmse}')

        #save the model
        trainer.save_model()
