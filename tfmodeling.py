# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Parameter", "UnitVector"]

import numpy as np
import tensorflow as tf


def get_param_for_value(value, min_value, max_value):
    if ((min_value is not None and np.any(value <= min_value)) or
            (max_value is not None and np.any(value >= max_value))):
        raise ValueError("value must be in the range (min_value, max_value)")
    if min_value is None and max_value is None:
        return value
    result = 0.0
    if min_value is not None:
        result += np.log(value - min_value)
    if max_value is not None:
        result -= np.log(max_value - value)
    return result


def get_value_for_param(param, min_value, max_value, _np=np):
    if min_value is None and max_value is None:
        return param
    if min_value is None:
        return max_value - _np.exp(-param)
    if max_value is None:
        return min_value + _np.exp(param)
    return min_value + (max_value - min_value) / (1.0 + _np.exp(-param))


class Parameter(object):

    def __init__(self, value, bounds=None, name=None, dtype=None):
        self.name = name
        self.bounds = bounds
        with tf.name_scope(name, "Parameter"):
            if bounds is None:
                self._parameters = [tf.Variable(value, name="parameter",
                                                dtype=dtype)]
                self._value = self._parameters[0]
                self._log_jac = tf.constant(0.0, dtype=self._value.dtype)
            else:
                self._parameters = [tf.Variable(
                    get_param_for_value(value, *bounds), name="parameter",
                    dtype=dtype)]
                self._value = get_value_for_param(self._parameters[0], *bounds,
                                                  _np=tf)
                self._log_jac = tf.constant(0.0, dtype=self._value.dtype)
                if bounds[0] is not None:
                    self._log_jac += tf.reduce_sum(
                        tf.log(self._value - bounds[0]))
                if bounds[1] is not None:
                    self._log_jac += tf.reduce_sum(
                        tf.log(bounds[1] - self._value))

    @property
    def value(self):
        return self._value

    @property
    def parameters(self):
        return self._parameters

    @property
    def log_jacobian(self):
        return self._log_jac


class UnitVector(Parameter):

    def __init__(self, x, y, name=None, dtype=None):
        self.name = name
        with tf.name_scope(name, "UnitVector"):
            self.x = tf.Variable(x, dtype=dtype, name="x")
            self.y = tf.Variable(y, dtype=dtype, name="y")
            norm = tf.square(self.x) + tf.square(self.y)

            self._parameters = [self.x, self.y]
            self._value = tf.stack((x/norm, y/norm))
            self._log_jac = -0.5 * tf.reduce_sum(norm)



