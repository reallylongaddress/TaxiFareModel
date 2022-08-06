import pandas as pd
import s3fs

from TaxiFareModel.utils import calculate_direction, minkowski_distance_gps, haversine_distance
from TaxiFareModel.utils import


#AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"


def get_data(nrows=10_000):

    '''returns a DataFrame with nrows from s3 bucket'''
    # df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)

    #was having issues with AWS, access data locally
    df_train = pd.read_csv('./raw_data/train.csv', nrows=nrows)
    return df_train

def clean_data(df):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


def fe_is_airport(row, airport_radius):
    if row['pickup_distance_to_lga']<airport_radius or \
       row['dropoff_distance_to_lga']<airport_radius or \
       row['pickup_distance_to_jfk']<airport_radius or \
       row['dropoff_distance_to_jfk']<airport_radius :
        return 1
    return 0

def feature_engineering(df):

    airport_radius = 2

    # manhattan distance <=> minkowski_distance(x1, x2, y1, y2, 1)
    df['manhattan_dist'] = minkowski_distance_gps(df['pickup_latitude'], df['dropoff_latitude'],
                                                  df['pickup_longitude'], df['dropoff_longitude'], 1)
    # euclidian distance <=> minkowski_distance(x1, x2, y1, y2, 2)
    df['euclidian_dist'] = minkowski_distance_gps(df['pickup_latitude'], df['dropoff_latitude'],
                                                  df['pickup_longitude'], df['dropoff_longitude'], 2)

    df['delta_lon'] = df.pickup_longitude - df.dropoff_longitude
    df['delta_lat'] = df.pickup_latitude - df.dropoff_latitude
    df['direction'] = calculate_direction(df.delta_lon, df.delta_lat)

    #how are are pickup/dropoff from jfk airport?
    jfk_center = (40.6441666667, -73.7822222222)

    df["jfk_lat"], df["jfk_lng"] = jfk_center[0], jfk_center[1]

    args_pickup =  dict(start_lat="jfk_lat", start_lon="jfk_lng",
                        end_lat="pickup_latitude", end_lon="pickup_longitude")
    args_dropoff =  dict(start_lat="jfk_lat", start_lon="jfk_lng",
                         end_lat="dropoff_latitude", end_lon="dropoff_longitude")

    df['pickup_distance_to_jfk'] = haversine_distance(df, **args_pickup)
    df['dropoff_distance_to_jfk'] = haversine_distance(df, **args_dropoff)

    #how are are pickup/dropoff from lga airport?
    lga_center = (40.776927, -73.873966)

    df["lga_lat"], df["lga_lng"] = lga_center[0], lga_center[1]

    args_pickup =  dict(start_lat="lga_lat", start_lon="lga_lng",
                        end_lat="pickup_latitude", end_lon="pickup_longitude")
    args_dropoff =  dict(start_lat="lga_lat", start_lon="lga_lng",
                         end_lat="dropoff_latitude", end_lon="dropoff_longitude")

    # jfk = (-73.7822222222, 40.6441666667)
    df['pickup_distance_to_lga'] = haversine_distance(df, **args_pickup)
    df['dropoff_distance_to_lga'] = haversine_distance(df, **args_dropoff)

    #which pickups/dropoffs can be considered airport runs?
    df['is_airport'] = df.apply(lambda row: fe_is_airport(row, airport_radius), axis=1)

    df.drop(columns=['jfk_lat', 'jfk_lng', 'lga_lat', 'lga_lng',
                     'pickup_distance_to_jfk', 'dropoff_distance_to_jfk',
                     'pickup_distance_to_lga', 'dropoff_distance_to_lga',
                     'delta_lon', 'delta_lat'], inplace=True)

    return df

# if __name__ == '__main__':
#     df = get_data()
