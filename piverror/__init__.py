import numpy as np
from scipy import fft


def energy(velocity_field, dx=0, dy=0):
    """
    Returns the energy of velocity field
    :param velocity_field: array_like
        3d input array.
    :param dx: float64
       single scalar to specify a sample distance for x-axis.
    :param dy: float64
       single scalar to specify a sample distance for y-axis.
    :return: float64
       single scalar describing total energy of velocity field.
    """
    shape = np.shape(velocity_field)
    return np.sum(velocity_field ** 2) / (2 * shape[1] * shape[2])


def enstrophy(velocity_field, dx, dy):
    """
    Returns the enstrophy of velocity field
    :param velocity_field: array_like
        3d input array.
    :param dx: float64
        single scalar to specify a sample distance for x-axis.
    :param dy: float64
        single scalar to specify a sample distance for y-axis.
    :return: float64
        single scalar describing enstrophy of the velocity field.
    """
    shape = np.shape(velocity_field)
    dvy_x = np.gradient(velocity_field[1, :, :], dx, axis=1)
    dvx_y = np.gradient(velocity_field[0, :, :], dy, axis=0)
    rot = dvy_x - dvx_y
    rot_sq = np.sum(rot ** 2)
    return rot_sq / (2 * shape[1] * shape[2])


def butterworth(x, kf=np.pi, n=16):
    return 1 / (1 + (x / kf) ** (2 * n)) ** 0.5


def low_pass(velocity_field, dx, dy, filter_args=(2/3*np.pi, 16)):
    """
    Return low pass part of the decomposition q = \bar q + \tilde q
    :param velocity_field: array_like
        2d or greater input array.
    :param dx: float64
        single scalar to specify a sample distance for x-axis.
    :param dy: float64
        single scalar to specify a sample distance for x-axis.
    :param filter_args: tuple
        arguments for the filtering function.
    :return: array_like
        output array of the same shape as input array describing low pass part of velocity.
    """
    shape = np.shape(velocity_field)
    k_x = 2 * np.pi * fft.fftfreq(shape[-2], dx)
    k_y = 2 * np.pi * fft.fftfreq(shape[-1], dy)
    k_xy = np.sqrt(np.square(k_x[np.newaxis, :]) + np.square(k_y[:, np.newaxis]))
    butterworth_wnd = butterworth(k_xy, *filter_args)
    return np.real_if_close(fft.ifft2(fft.fft2(velocity_field) * butterworth_wnd))


def energy_flux(velocity_field, dx, dy):
    """
    Returns the flux of energy of velocity field
    :param velocity_field: array_like
        3d input array
    :param dx: float64
        single scalar to specify a sample distance for x-axis.
    :param dy: float64
        single scalar to specify a sample distance for y-axis.
    :return: float64
        single scalar describing energy flux of velocity field.
    """
    v_x_l = low_pass(velocity_field[0, :, :], dx, dy)
    v_y_l = low_pass(velocity_field[1, :, :], dx, dy)
    v_xy_l = low_pass(velocity_field[0, :, :] * velocity_field[1, :, :], dx, dy)
    dv_x_l = np.gradient(v_x_l, dy, axis=0)
    flux = -(v_xy_l - v_x_l * v_y_l) * dv_x_l
    return np.sum(flux)


def accuracy(func, velocity_field, error, dx=0, dy=0):
    """
    Returns the relative indirect error of function
    :param func: callable
        function for which the error distribution is calculated.
    :param velocity_field: array_like
        3d input array.
    :param error: array_like
        3d input array of the same shape as a velocity_field.
    :param dx: float64
        single scalar to specify a sample distance for x-axis.
    :param dy: float64
        single scalar to specify a sample distance for x-axis.
    :return: float64
        single scalar denoting a absolute error.
    """
    delta_f = func(velocity_field+error, dx, dy) - func(velocity_field, dx, dy)
    return delta_f


def vel_error(shape, kind, st_dev):
    """
    Returns a random error of certain type.
    :param shape: tuple of int
        shape of output array.
    :param kind: string
        type of error distribution.
    :param st_dev: float64
        standard deviation of error distribution.
    :return: array_like
        3d output array.
    """
    n_x = shape[0]
    n_y = shape[1]
    if kind == 'gauss':
        mis = np.random.normal(0, st_dev, size=(2, n_x, n_y))
        return mis
    if kind == 'uniform':
        mis = st_dev * (np.random.random_sample((2, n_x, n_y)) - 1/2)
        return mis
    if kind == 'discrete':
        mis = np.random.randint(-1, 2, (2, n_x, n_y)) * st_dev
        return mis


def func_error(k, func, velocity_field, shape, kind, st_dev, dx=0, dy=0):
    """
    Returns array of relative errors for k different samples.
    :param k: int
        number of samples of the error.
    :param func: callable
        function for which the error distribution is calculated.
    :param velocity_field: array_like
        3d input array.
    :param shape: tuple of int
        shape of error array.
    :param kind: string
        type of error distribution.
    :param st_dev: float64
        standard deviation of error distribution.
    :param dx: float64
        single scalar to specify a sample distance for x-axis.
    :param dy: float64
        single scalar to specify a sample distance for x-axis.
    :return: array_like
        1d array containing absolute errors.
    """
    ac_abs = np.zeros(k, )
    for i in range(k):
        ac_abs[i] = accuracy(func, velocity_field, vel_error(shape, kind, st_dev) * velocity_field, dx, dy)
    return ac_abs
