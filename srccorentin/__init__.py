# Package containing all the steps of the concert purchase prediction pipeline:
# - data_loading:       read the CSV / JSON files
# - music_features:     listening features built from the music app JSON
# - recency_features:   how recently users listened to music / artists
# - csv_features:       date, price, geography and missing-value features
# - behaviour_features: rankings, intentions and price sensitivity
# - preprocessing:      X / y preparation, train/validation split, ColumnTransformer
# - models:             Random Forest, Logistic Regression, LightGBM pipelines
# - evaluation:         metrics, threshold optimisation, feature importance plots
