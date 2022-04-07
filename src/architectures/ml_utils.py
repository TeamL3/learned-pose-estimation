import numpy as np
import pickle
import neptune.new as neptune
from pose_utils import DEG_TO_RAD
from pose_utils import RAD_TO_SCALED
from pose_utils import MAX_DEPTH
from pose_utils import METERS_TO_SCALED
from pose_utils import INTENSITY_TO_SCALED


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow.keras as keras


def get_neptune_run(run_id, proj="ljburtz/relative-pose"):
    """ get the neptune run object stored with the run_id and the params dictionary that defined it's U-net training parameters
    """
    # initialize this latest run, but disable restart of monitoring logs
    run = neptune.init(
        project=proj,
        run=run_id,
        # capture_stdout=False,
        # capture_stderr=False,
        capture_hardware_metrics=False,
    )
    # get parameters dictionary
    param_struct = run.get_structure()['parameters']
    params = {}
    from neptune.new.attributes.atoms.string import String
    from neptune.new.attributes.atoms.float import Float
    from neptune.new.attributes.atoms.integer import Integer
    for param in param_struct.keys():
        if type(param_struct[param]) == String or type(param_struct[param]) == Float or type(param_struct[param]) == Integer:
            params[param] = run[f'parameters/{param}'].fetch()
    return params, run


def get_neptune_latest_run(proj="ljburtz/sar-unet2"):
    """ get id of run on Neptune that has the latest creation_time. Then use that id to get the corresponding neptune run object and associated run parameters
    """
    runs_df = neptune.get_project(name=proj).fetch_runs_table().to_pandas()
    latest_run_id = runs_df.sort_values('sys/creation_time', ascending=False)['sys/id'].values[0]
    return get_neptune_run(latest_run_id, proj=proj)


def get_best_metrics(history, loss_metric='loss', accuracy_metric='position_diff', val_prefix='val_'):
    # loss = history.history[loss_metric]
    val_loss = history.history[val_prefix + loss_metric]
    # acc = history.history[accuracy_metric]
    val_acc = history.history[val_prefix + accuracy_metric]

    min_loss = min(val_loss)
    min_loss_epoch = val_loss.index(min_loss)

    min_acc = min(val_acc)
    min_acc_epoch = val_acc.index(min_acc)

    return min_loss, min_loss_epoch, min_acc, min_acc_epoch


def load_dataset(path, compression=None):
    with open(path + '/element_spec', 'rb') as in_:
        es = pickle.load(in_)

    loaded = tf.data.experimental.load(
        path, es, compression=compression
    )
    return loaded


def get_keras_model_full(custom_objects, model_path='/tmp/model_end.h5', compiled=True):
    """ given the path to the .h5 saved model, and custom objects, return the corresponding keras model (and by default: compile it)
    """
    keras.backend.clear_session()
    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects
    )
    if compiled:
        model.compile(
            loss=custom_objects['pose_loss'],
            optimizer="adam",
            metrics=[*custom_objects.values()][:-1]
        )
    return model


def get_keras_model_inference(config_path='/tmp/model_relative_pose_config.json', weigths_path="/tmp/model_relative_pose_weights.h5"):
    """ lightweight model loading for inference purposes
    config path should point to the json file created by model.to_json() and
    weights path should point to the .h5 file created by model.save_weights after training.
    Note: the model returned by this method cannot be used for additional training or model.evaluate() because custom objects are not included
    """
    import json
    with open(config_path) as in_:
        json_config = json.load(in_)
    model = keras.models.model_from_json(json_config)
    model.load_weights(weigths_path)
    return model


def position_loss(y_true, y_pred):
    position_true = y_true[:, :2]
    position_pred = y_pred[:, :2]
    pos_diff = tf.norm(position_true - position_pred, axis=-1)
    position_loss = tf.reduce_mean(pos_diff, axis=-1)

    # d_true = y_true[:, 0]
    # theta_true = y_true[:, 1]
    # d_pred = y_pred[:, 0]
    # theta_pred = y_pred[:, 1]
    #
    # pos_diff = tf.sqrt(tf.square(d_true) + tf.square(d_pred) - 2 * tf.multiply(tf.multiply(d_true, d_pred), tf.cos(theta_true - theta_pred)))
    # position_loss = tf.reduce_mean(pos_diff, axis=-1)
    return position_loss


def distance_loss(y_true, y_pred):
    d_true = y_true[:, 0]
    d_pred = y_pred[:, 0]
    d_diff = tf.square(d_true - d_pred)
    distance_loss = tf.reduce_mean(d_diff, axis=-1)
    return distance_loss


def theta_loss(y_true, y_pred):
    theta_true = y_true[:, 1]
    theta_pred = y_pred[:, 1]
    delta = theta_true - theta_pred
    # batch compatible modulo 2 (range of theta is [-1,1])
    delta = tf.where(tf.math.less(delta, -1.), tf.math.add(delta, 2.), delta)
    delta = tf.where(tf.math.greater(delta, 1.), tf.math.add(delta, -2.), delta)
    theta_diff = tf.square(delta)
    theta_loss = tf.reduce_mean(theta_diff, axis=-1)
    return theta_loss


def orientation_loss(y_true, y_pred):
    orientation_true = y_true[:, 2]
    orientation_pred = y_pred[:, 2]
    ori_diff = tf.square(orientation_true - orientation_pred)
    orientation_loss = tf.reduce_mean(ori_diff, axis=-1)
    return orientation_loss


# --- Metrics for humans ---
def position_diff(y_true, y_pred):
    position_true = y_true[:, :2]
    position_pred = y_pred[:, :2]
    position_diff = tf.norm(position_true - position_pred, axis=-1) / METERS_TO_SCALED
    return tf.reduce_mean(position_diff, axis=-1)


def distance_diff(y_true, y_pred):
    d_true = y_true[:, 0]
    d_pred = y_pred[:, 0]
    d_diff = tf.abs(d_true - d_pred) / METERS_TO_SCALED
    return tf.reduce_mean(d_diff, axis=-1)


def theta_diff(y_true, y_pred):
    theta_true = y_true[:, 1]
    theta_pred = y_pred[:, 1]
    delta = theta_true - theta_pred
    # batch compatible modulo 2 (range of theta is [-1,1])
    delta = tf.where(tf.math.less(delta, -1.), tf.math.add(delta, 2.), delta)
    delta = tf.where(tf.math.greater(delta, 1.), tf.math.add(delta, -2.), delta)
    theta_diff = tf.abs(delta) / RAD_TO_SCALED / DEG_TO_RAD
    return tf.reduce_mean(theta_diff, axis=-1)


def orientation_diff(y_true, y_pred):
    orientation_true = y_true[:, 2]
    orientation_pred = y_pred[:, 2]
    orientation_diff = tf.abs(orientation_true - orientation_pred) / RAD_TO_SCALED / DEG_TO_RAD
    return tf.reduce_mean(orientation_diff, axis=-1)


def predict_and_scale(model, ds, ds_batched, n_pred, batch_size):
    d_list = []
    theta_list = []
    yaw_list = []
    d_true = []
    theta_true = []
    yaw_true = []
    for (d, theta, yaw), (image, label) in zip(model.predict(ds_batched.take(n_pred)), ds.take(n_pred * batch_size)):
        d_list.append(d / METERS_TO_SCALED)
        theta_list.append(theta / RAD_TO_SCALED)
        yaw_list.append(yaw / RAD_TO_SCALED)

        d_true.append(label.numpy()[0] / METERS_TO_SCALED)
        theta_true.append(label.numpy()[1] / RAD_TO_SCALED)
        yaw_true.append(label.numpy()[2] / RAD_TO_SCALED)
    return d_true, theta_true, yaw_true, d_list, theta_list, yaw_list
