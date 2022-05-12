import tensorflow as tf
import numpy as np


def adam_simpleclip(weight: tf.Variable, grad, moment, args):
    lr = args["lr"]
    beta1, beta2 = args["beta1"], args["beta2"]
    smoothFactor = args["smoothFactor"]
    clip_grad = args["clip_grad"]
    clip_lambda = args["clip_lambda"]
    grad = tf.clip_by_value(grad, -clip_grad, clip_grad)
    m1, m2 = moment
    m1 = m1*beta1+grad*(1-beta1)
    m2 = m2*beta2+tf.square(grad)*(1-beta2)
    grad = m1/(tf.sqrt(m2)+smoothFactor)
    weight.assign_sub(lr*grad)
    return grad


def adam_clip(weight: tf.Variable, grad, moment, args):
    lr = args["lr"]
    beta1, beta2 = args["beta1"], args["beta2"]
    smoothFactor = args["smoothFactor"]
    clip_grad = args["clip_grad"]
    clip_lambda = args["clip_lambda"]
    grad = tf.clip_by_value(grad, -clip_grad, clip_grad)
    m1, m2 = moment
    m1 = m1*beta1+grad*(1-beta1)
    m2 = m2*beta2+tf.square(grad)*(1-beta2)
    grad = m1/(tf.sqrt(m2)+smoothFactor)
    grad_module = tf.sqrt(tf.reduce_sum(tf.square(grad)))
    if grad_module >= clip_lambda:
        grad = grad/(grad_module+0.1*clip_lambda)*clip_lambda
    weight.assign_sub(lr*grad)
    return grad


def adam_adaclip(weight: tf.Variable, grad, moment, args):
    lr = args["lr"]
    beta1, beta2 = args["beta1"], args["beta2"]
    smoothFactor = args["smoothFactor"]
    clip_grad = args["clip_grad"]
    clip_lambda = args["clip_lambda"]
    grad = tf.clip_by_value(grad, -clip_grad, clip_grad)
    m1, m2 = moment
    m1 = m1*beta1+grad*(1-beta1)
    m2 = m2*beta2+tf.square(grad)*(1-beta2)
    grad = m1/(tf.sqrt(m2)+smoothFactor)
    grad_lambda = tf.math.sqrt(tf.reduce_mean(tf.square(
        grad))/(tf.reduce_sum(tf.square(weight))+smoothFactor)*np.prod(grad.shape))
    if grad_lambda >= clip_lambda:
        grad = grad/(grad_lambda+0.1*clip_lambda)*clip_lambda
    weight.assign_sub(lr*grad)
    return grad


summary = {
    "adam_simpleclip": adam_simpleclip,
    "adam_clip": adam_clip,
    "adam_adaclip": adam_adaclip
}
