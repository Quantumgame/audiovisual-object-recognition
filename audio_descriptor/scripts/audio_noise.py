"""
Module for generate white and pink noise
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import numpy
import numpy.random as rng

def iterwhite():
    """
    Generate a sequence of samples of white noise.

    Generates a never-ending sequence of floating-point values.
    """
    while True:
        for n in rng.randn(100):
            yield n


def iterpink(depth=20):
    """
    Generate a sequence of samples of pink noise.

    Based on the Voss-McCartney algorithm, discussion and code examples at
    http://www.firstpr.com.au/dsp/pink-noise/

    depth: Use this many samples of white noise to calculate the output. A
      higher number is slower to run, but renders low frequencies with more
      correct power spectra.

    Generates a never-ending sequence of floating-point values. Any continuous
    set of these samples will tend to have a 1/f power spectrum.
    """
    values = rng.randn(depth)
    smooth = rng.randn(depth)
    source = rng.randn(depth)
    sum = values.sum()
    i = 0
    while True:
        yield sum + smooth[i]

        # advance the index by 1. if the index wraps, generate noise to use in
        # the calculations, but do not update any of the pink noise values.
        i += 1
        if i == depth:
            i = 0
            smooth = rng.randn(depth)
            source = rng.randn(depth)
            continue

        # count trailing zeros in i
        c = 0
        while not (i >> c) & 1:
            c += 1

        # replace value c with a new source element
        sum += source[i] - values[c]
        values[c] = source[i]


def _asarray(source, size):
    """
    Reshape signal to the desired size
    :param: function. Returns white or pink noise.
    :param int. Desired array size.
    """
    noise = source()
    if size is None:
        return noise.next()
    #count = reduce(operator.mul, shape)
    return numpy.asarray([noise.next() for _ in range(size)])


def white(shape=None):
    """
    Generate white noise.

    :param shape: If given, returns a numpy array of white noise with this shape. If
      not given, return just one sample of white noise.
    """
    return _asarray(iterwhite, shape)


def pink(shape=None, depth=20):
    """
    Generate an array of pink noise.

    :param shape: If given, returns a numpy array of noise with this shape. If not
      given, return just one sample of noise.
    :param depth: Use this many samples of white noise to calculate pink noise. A
      higher number is slower to run, but renders low frequencies with more
      correct power spectra.
    """
    return _asarray(lambda: iterpink(depth), shape)
