import tensorflow as tf
import functools
import sys

sys.path.append("..")
import global_variables as G


def build_input_features():
    # functor to preprocess continuous features
    def process_continuous_data(mean, var, data):
        data = (tf.cast(data, tf.float32) - mean) / var
        return tf.reshape(data, [-1, 1])
    
    # Compared with the codes given in tf2.0 documentary, 
    # this can guarantee the correctness of the feature order
    feature_columns = []
    for col in G.COLUMNS:
        if col in G.CATEGORIES.keys():
            print(' [*] processing column key={} \t\ttype=CATEGORICAL'.format(col))
            cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
                key=col, vocabulary_list=G.CATEGORIES[col])
            feature = tf.feature_column.indicator_column(cat_col)
        elif col in G.MEANVARS.keys():
            print(' [*] processing column key={} \t\ttype=NUMERICAL'.format(col))
            norm_fn = functools.partial(process_continuous_data, *G.MEANVARS[col])
            feature = tf.feature_column.numeric_column(key=col, normalizer_fn=norm_fn)
        else:
            continue
        feature_columns.append(feature)

    # preprocessing layer in keras
    preprocessing_layer = tf.keras.layers.DenseFeatures(feature_columns)
    return preprocessing_layer


def build_shared_fc(highest_flight, lowest_flight, middle_flight, flight_global):
    # create single layer embedding for each flight feature
    flight_input = tf.keras.layers.Input(shape=(G.EACH_FLIGHT_DIM))
    out = tf.keras.layers.Dense(64, activation='relu')(flight_input)
    shared_model = tf.keras.models.Model(flight_input, out)
    
    highest_out = shared_model(highest_flight)
    lowest_out = shared_model(lowest_flight)
    middle_out = shared_model(middle_flight)

    flight_global_out = tf.keras.layers.Dense(32, activation='relu')(flight_global)
    concatenated = tf.keras.layers.concatenate([
        highest_out, lowest_out, middle_out, flight_global_out
    ], axis=-1)
    return concatenated


def build_lstm_model(seq_features, num_units=64, allow_cudnn_kernel=True):
    """build_lstm_model
    CuDNN is only available at layer level, and not at the cell level.
    This means `LSTM(units)` will use the CuDNN kernel,
    while `RNN(LSTMCell(units))` will run on non-CuDNN kernel.
    """
    input_dim = seq_features.shape[-1]
    if allow_cudnn_kernel:
        lstm = tf.keras.layers.LSTM(num_units, input_shape=(None, input_dim))(seq_features)
    else:
        lstm = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(units),
            input_shape=(None, input_dim))(seq_features)
    return lstm


class UnionFeatureModel(tf.keras.Model):
    """UnionFeatureModel
    Only consider sequential features and addressing the prediction using LSTM
    """
    name = 'UnionFeatureModel'
    
    def __init__(self):
        super(UnionFeatureModel, self).__init__()
        self.preprocessing_layer = build_input_features()
        self.fc_shared_1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc_shared_2 = tf.keras.layers.Dense(128, activation='relu')
        if tf.test.is_built_with_cuda():
            self.lstm = tf.keras.layers.LSTM(
                units=64, input_shape=(None, 128), return_state=True)
        else:
            self.lstm = tf.keras.layers.RNN(
                tf.keras.layers.LSTMCell(32), 
                input_shape=(None, 64), return_state=True)
        self.shared_fc = tf.keras.layers.Dense(32, activation='relu')
        self.fc_global = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_op):
        proc_xs = self.preprocessing_layer(input_op)
        seq_xs = tf.split(proc_xs, G.GLOBAL_SPLITS, axis=-1)
        embedded_xs = []
        for seq in seq_xs[: -1]:
            embedded = self.fc_shared_2(self.fc_shared_1(proc_xs))
            embedded_xs.append(embedded)
        shared_embedded = self.shared_fc(seq_xs[-1])
        stacked = tf.stack(embedded_xs, axis=1)
        lstm_hidden = self.lstm(stacked)[-2]
        concat = tf.keras.layers.concatenate([lstm_hidden, shared_embedded], axis=-1)
        return self.fc_global(concat)


def build_union_feature_model():
    return UnionFeatureModel()


if __name__ == '__main__':
    pass