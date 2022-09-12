"""
Optuna example that demonstrates a pruner for Keras.
In this example, we optimize the validation accuracy of hand-written digit recognition using
Keras and MNIST, where the architecture of the neural network and the learning rate of optimizer
is optimized. Throughout the training of neural networks, a pruner observes intermediate
results and stops unpromising trials.
You can run this example as follows:
    $ python keras_integration.py
For a similar Optuna example that demonstrates Keras without a pruner on a regression dataset,
see the following link:
    https://github.com/optuna/optuna/blob/master/examples/mlflow/keras_mlflow.py
"""
import warnings

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import keras
import optuna
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential, Model
from optuna.integration import KerasPruningCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import np_utils

from lib.price_change import features_generation, generate_price_change


BATCHSIZE = 32
CLASSES = 3
EPOCHS = 50


def create_model(trial):
    # We optimize the number of layers, hidden units and dropout in each layer and
    # the learning rate of RMSProp optimizer.

    # We define our MLP.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    model = Sequential()
    for i in range(n_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        model.add(Dense(num_hidden, activation="relu"))
        dropout = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        model.add(Dropout(rate=dropout))
    model.add(Dense(CLASSES, activation="softmax"))

    # We compile our model with a sampled learning rate.
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    model.compile(
        loss='mse',         #"categorical_crossentropy",
        optimizer=keras.optimizers.RMSprop(lr=lr),
        metrics=["accuracy"],
    )

    return model


def objective(trial):
    # Clear clutter from previous session graphs.
    keras.backend.clear_session()

    fitted_data_path = 'fitted_data/NN'
    ticker = 'eurusd'
    n_clusters = 250
    bars_count = 100000
    features_columns = ['candle_size', 'body_size', 'lower_shadow', 'upper_shadow',
                             'color', 'body_candle_size', 'lower_shadow_size', 'upper_shadow_size',
                             'is_equal', 'is_higher', 'is_lower', 'is_inside', 'is_outside']

    # Data preparation
    train_data = pd.read_csv(f'../../data/{ticker}.csv')[-2 * bars_count:-bars_count]
    data_piece = None
    train_data.columns = map(str.lower, train_data.columns)
    train_data['date'] = pd.to_datetime(train_data['date'])
    train_data['hour'] = train_data.date.dt.hour
    df_with_generated_features = features_generation(train_data)
    df_with_generated_features = df_with_generated_features[(df_with_generated_features.hour >= 10) &
                                                            (df_with_generated_features.hour <= 19)]
    df_candle_features = df_with_generated_features[features_columns].dropna()
    target = generate_price_change(train_data, 50, 0.00035)

    scaler = StandardScaler(with_mean=True)
    df_candle_features_scaled = scaler.fit_transform(df_candle_features)
    kmeans = KMeans(n_clusters, random_state=0).fit(df_candle_features_scaled)

    df_candle_features['label'] = kmeans.labels_
    df_candle_features['price_change'] = target.loc[df_candle_features.index]

    # Make shifts
    shifts = 9
    for shft in range(1, shifts + 1):
        df_candle_features[f'label_shift_{shft}'] = df_candle_features['label'].shift(shft)

    df_candle_features = df_candle_features.dropna()
    y = df_candle_features['price_change']
    X = df_candle_features.drop(['price_change'], axis=1)
    label_columns = [col for col in df_candle_features.columns if 'label' in col]
    is_columns = ['is_equal', 'is_higher', 'is_lower', 'is_inside', 'is_outside']
    X[is_columns] = X[is_columns].astype(bool)

    onehot_encoder = OneHotEncoder(sparse=False)
    X_enc = onehot_encoder.fit_transform(X[label_columns])

    label_encoder = LabelEncoder()
    y_labelled = label_encoder.fit_transform(y)
    y_enc = np_utils.to_categorical(y_labelled)

    X_for_NN = np.concatenate((X_enc, X[is_columns].values), axis=1)

    x_train, x_valid, y_train, y_valid = train_test_split(X_for_NN, y_enc, test_size=0.3, random_state=1, shuffle=False)

    # Generate our trial model.
    model = create_model(trial)

    # Fit the model on the training data.
    # The KerasPruningCallback checks for pruning condition every epoch.
    model.fit(
        x_train,
        y_train,
        batch_size=BATCHSIZE,
        callbacks=[KerasPruningCallback(trial, "val_accuracy")],
        epochs=EPOCHS,
        validation_data=(x_valid, y_valid),
        verbose=1,
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(x_valid, y_valid, verbose=0)
    return score[1]


if __name__ == "__main__":
    warnings.warn(
        "Recent Keras release (2.4.0) simply redirects all APIs "
        "in the standalone keras package to point to tf.keras. "
        "There is now only one Keras: tf.keras. "
        "There may be some breaking changes for some workflows by upgrading to keras 2.4.0. "
        "Test before upgrading. "
        "REF:https://github.com/keras-team/keras/releases/tag/2.4.0"
    )
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100, timeout=600)
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
