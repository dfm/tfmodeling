# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Parameter", "UnitVector", "Model"]

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

    def __init__(self, value, bounds=None, name=None, dtype=None,
                 frozen=False):
        self.changed = True
        self.frozen = frozen
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

    def __getattr__(self, key):
        try:
            return getattr(self.value, key)
        except AttributeError:
            raise AttributeError(key)

    @property
    def value(self):
        return self._value

    def get_parameters(self, include_frozen=False):
        if self.frozen and not include_frozen:
            return []
        return self._parameters

    @property
    def log_jacobian(self):
        return self._log_jac

    def freeze(self):
        self.changed = True
        self.frozen = True

    def thaw(self):
        self.changed = True
        self.frozen = False


class UnitVector(Parameter):

    def __init__(self, x, name=None, dtype=None, frozen=False):
        self.changed = True
        self.frozen = frozen
        self.name = name
        with tf.name_scope(name, "UnitVector"):
            self.x = tf.Variable(x, dtype=dtype, name="x")
            norm = tf.reduce_sum(tf.square(self.x), axis=-1)

            self._parameters = [self.x]
            self._value = self.x / tf.expand_dims(tf.sqrt(norm), -1)
            self._log_jac = -0.5 * tf.reduce_sum(norm)


class Model(object):

    def __init__(self, target, parameters, feed_dict=None, session=None):
        self._changed = True
        self._parameters = parameters
        self._feed_dict = dict() if feed_dict is None else feed_dict
        self._session = session

        self.target = target
        for p in self._parameters:
            try:
                self.target += p.log_jacobian
            except AttributeError:
                pass

    @property
    def changed(self):
        return self._changed or any(p.changed for p in self.get_parameters())

    def get_parameters(self, include_frozen=False):
        params = []
        for par in self._parameters:
            try:
                params += par.get_parameters(include_frozen=include_frozen)
            except AttributeError:
                params.append(par)
        return params

    @property
    def session(self):
        if self._session is None:
            self._session = tf.get_default_session()
        return self._session

    def value(self, vector):
        feed_dict = self.vector_to_feed_dict(vector)
        return self.session.run(self.target, feed_dict=feed_dict)

    def gradient(self, vector):
        feed_dict = self.vector_to_feed_dict(vector)
        return np.concatenate([
            np.reshape(g, s) for s, g in zip(
                self.sizes,
                self.session.run(self.grad_target, feed_dict=feed_dict))
        ])

    def update(self):
        if not self.changed:
            return
        self.parameters = self.get_parameters()
        self.grad_target = tf.gradients(self.target, self.parameters)
        values = self.session.run(self.parameters, feed_dict=self._feed_dict)
        self.shapes = [np.shape(v) for v in values]
        self.sizes = [np.size(v) for v in values]
        for p in self.parameters:
            p.changed = False
        self._changed = False

    def vector_to_feed_dict(self, vector, specs=None):
        self.update()
        i = 0
        fd = dict(self._feed_dict)
        for var, shape, size in zip(self.parameters, self.shapes, self.sizes):
            fd[var] = np.reshape(vector[i:i+size], shape)
            i += size
        return fd

    def feed_dict_to_vector(self, feed_dict):
        self.update()
        return np.concatenate([
            np.reshape(feed_dict[v], s)
            for v, s in zip(self.parameters, self.sizes)])

    def current_vector(self):
        self.update()
        values = self.session.run(self.parameters, feed_dict=self._feed_dict)
        return np.concatenate([
            np.reshape(v, s)
            for v, s in zip(values, self.sizes)])

    def get_values_for_chain(self, chain, var_list=None, names=None):
        self.update()

        if var_list is None:
            if names is None:
                names = [p.name for p in self._parameters]
            var_list = [p.value if hasattr(p, "value") else p
                        for p in self._parameters]
        elif names is None:
            names = [v.name for v in var_list]

        # Work out the dtype for the output chain
        dtype = [(n, float, np.shape(v))
                 for n, v in zip(names,
                                 self.session.run(var_list,
                                                  feed_dict=self._feed_dict))]

        # Allocate the output chain
        N = len(chain)
        out_chain = np.empty(N, dtype=dtype)

        # Loop over the chain and get the value at each sample
        for i, s in enumerate(chain):
            fd = self.vector_to_feed_dict(s)
            for n, v in zip(names, self.session.run(var_list, feed_dict=fd)):
                out_chain[n][i] = v
        return out_chain
