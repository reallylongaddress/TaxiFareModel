from re import sub
import numpy as np
import pandas as pd

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score

from memoized_property import memoized_property

from TaxiFareModel.data import feature_engineering, get_data, clean_data
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer

class Trainer():

    MLFLOW_URI = "https://mlflow.lewagon.ai/"
    EXPERIMENT_NAME = "[REMOTE] [reallylongaddress] TaxiFareModel + 0.0.8"

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.mlflow_log_param('train_rows_data', len(self.X))

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
            knn_weight = 'uniform'
            # knn_weight = 'distance'

            # knn_n_neighbors = 5
            knn_n_neighbors = 10
            # knn_n_neighbors = 15

            self.mlflow_log_param('knn_weights', knn_weight)
            self.mlflow_log_param('knn_n_neighbors', knn_n_neighbors)

            pipeline = Pipeline([
                ('preproc', preproc_pipe),
                ('model', KNeighborsRegressor(n_neighbors=knn_n_neighbors, weights=knn_weight))
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

        cv_results = cross_val_score(self.pipeline,
                                     self.X, self.y,
                                     cv=3,
                                     n_jobs=-1,
                                     scoring='neg_root_mean_squared_error')

        #-1 since we have to use the neg(_root_mean_squared_error) above
        self.mlflow_log_metric('rmse_train_cv', np.mean(cv_results)*-1)

    def evaluate(self, X_val, y_var):
        self.mlflow_log_param('test_rows_data', len(X_val))

        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_val)
        rmse = compute_rmse(y_pred, y_var)
        self.mlflow_log_metric('rmse_test', rmse)

        cv_results = cross_val_score(self.pipeline,
                                     X_val, y_var,
                                     cv=3,
                                     n_jobs=-1,
                                     scoring='neg_root_mean_squared_error')

        #-1 since we have to use the neg(_root_mean_squared_error) above
        self.mlflow_log_metric('rmse_test_cv', np.mean(cv_results)*-1)

        return rmse

    def save_model(self, estimator):
        """ Save the trained model into a model.joblib file """
        save_result = joblib.dump(self.pipeline, f'./model_{estimator}.joblib')
        print(f'save_result: {save_result}')

    def predict(self, df_test):
        y_pred = self.pipeline.predict(df_test)
        y_pred = pd.concat([df_test["key"],pd.Series(y_pred)],axis=1)
        y_pred.columns = ['key', 'fare_amount']

        return y_pred

    def save_submission(self, submission_data, estimator):
        submission_data.to_csv(f'./submission_{estimator}.csv', index=False)

if __name__ == "__main__":
    #load data
    df_train, df_test = get_data(nrows=500_000)

    # clean data
    df_train = clean_data(df_train)

    #feature enginering
    df_train = feature_engineering(df_train)

    # set X and y
    X = df_train.drop(columns=['fare_amount'])
    y = df_train['fare_amount']

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=.9)

    # train
    estimators = ['KNN']
    for estimator in estimators:
        print(f'running: {estimator}')
        trainer = Trainer(X_train, y_train)
        trainer.run(estimator)

        # evaluate
        rmse = trainer.evaluate(X_val, y_val)
        print(f'validate rmse: {rmse}')

        y_pred = trainer.predict(df_test)

        # print(y_pred_list)
        trainer.save_submission(y_pred, estimator)

        #save the model
        trainer.save_model(estimator)
