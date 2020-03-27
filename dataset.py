import tensorflow as tf
import os

import global_variables as G


def create_dataset(csv_path, batch_size=64, num_epochs=20, shuffle=True):
    csv_files = os.listdir(csv_path)
    csv_files = [os.path.join(csv_path, fn) for fn in csv_files]
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=csv_files,
        batch_size=batch_size,
        label_name=G.LABEL_NAME,
        select_columns=G.COLUMNS,
        na_value='?',
        num_epochs=num_epochs,
        shuffle=shuffle)
    return dataset


def create_tmp_dataset(csv_path, batch_size=3, num_epochs=1, shuffle=False):
    csv_files = os.listdir(csv_path)
    csv_files = [os.path.join(csv_path, fn) for fn in csv_files]
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=csv_files,
        batch_size=batch_size,
        select_columns=G.COLUMNS,
        label_name=G.LABEL_NAME,
        num_epochs=num_epochs,
        shuffle=shuffle)
    return dataset


def build_input_features(input_op):
    import functools
    # preprocessing categorical features
    categorical_columns = []
    for feature, vocab in G.CATEGORIES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
                key=feature, vocabulary_list=vocab)
        categorical_columns.append(tf.feature_column.indicator_column(cat_col))

    # preprocessing continuous features
    def process_continuous_data(mean, var, data):
        data = (tf.cast(data, tf.float32) - mean) / var
        return tf.reshape(data, [-1, 1])
    numerical_columns = []
    for feature in G.MEANVARS.keys():
        norm_fn = functools.partial(process_continuous_data, *G.MEANVARS[feature])
        num_col = tf.feature_column.numeric_column(feature, normalizer_fn=norm_fn)
        numerical_columns.append(num_col)

    # preprocessing layer in keras
    preprocessing_layer = tf.keras.layers.DenseFeatures(
            categorical_columns + numerical_columns)

    # start to build input feature models
    preprocessed = preprocessing_layer(input_op)
    return preprocessed


if __name__ == '__main__':
    dataset = create_tmp_dataset('./data')
    batch_xs, batch_ys = next(iter(dataset))
    proc_xs = build_input_features(batch_xs)

    print('\n')
    print('batch_xs:', batch_xs)
    print('\n')
    print('proc_xs:', proc_xs)
    print('\n')
    print('batch_ys:', batch_ys)