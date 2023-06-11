import numpy as np
from scipy import fft
from openpiv import windef
from matplotlib.pyplot import hist as hst
import matplotlib.pyplot as plt


def energy(velocity_field, **kwargs):
    """
    Returns the energy of velocity field
    :param velocity_field: array_like
        3d input array.
    :return: float64
       single scalar describing total energy of velocity field.
    """
    return np.mean(velocity_field ** 2)


def enstrophy(velocity_field, dx, dy, **kwargs):
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
    dvy_x = np.gradient(velocity_field[1, :, :], dx, axis=1)
    dvx_y = np.gradient(velocity_field[0, :, :], dy, axis=0)
    rot = dvy_x - dvx_y
    return np.mean(rot ** 2) / 2


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
    k_x = 2 * np.pi * fft.fftfreq(shape[-1], dx)
    k_y = 2 * np.pi * fft.fftfreq(shape[-2], dy)
    k_xy = np.sqrt(np.square(k_x[np.newaxis, :]) + np.square(k_y[:, np.newaxis]))
    butterworth_wnd = butterworth(k_xy, *filter_args)
    return np.real_if_close(fft.ifft2(fft.fft2(velocity_field) * butterworth_wnd))


def energy_flux(velocity_field, dx, dy, filter_args=(2/3*np.pi, 16)):
    """
    Returns the energy flux for given velocity field at a specific wave number.
    :param velocity_field: array_like
        3d input array.
    :param dx: float64
        single scalar to specify a sample distance for x-axis.
    :param dy: float64
        single scalar to specify a sample distance for y-axis.
    :param filter_args: tuple
        arguments for the filtering function.
    :return: float64
        single scalar describing energy flux.
    """
    v_x_v_y_l = low_pass(velocity_field[0, :, :] * velocity_field[1, :, :], dx, dy, filter_args)
    v2_x_l = low_pass(velocity_field[0, :, :] ** 2, dx, dy, filter_args)
    v2_y_l = low_pass(velocity_field[1, :, :] ** 2, dx, dy, filter_args)
    v_x_l = low_pass(velocity_field[0, :, :], dx, dy, filter_args)
    v_y_l = low_pass(velocity_field[1, :, :], dx, dy, filter_args)
    dv_x_ldy = np.gradient(v_x_l, dy, axis=0)
    dv_y_ldx = np.gradient(v_y_l, dx, axis=1)
    dv_x_ldx = np.gradient(v_x_l, dx, axis=1)
    dv_y_ldy = np.gradient(v_y_l, dy, axis=0)
    flux = (v_x_v_y_l - v_x_l * v_y_l) * (dv_x_ldy + dv_y_ldx) + \
           (v2_y_l - v_y_l ** 2) * dv_y_ldy + (v2_x_l - v_x_l ** 2) * dv_x_ldx
    return np.mean(flux)


def max_energy_flux(velocity_field, dx, dy):
    """
    Returns the wave number that corresponds to energy flux maximum.
    :param velocity_field: array_like
        3d input array.
    :param dx: float64
        single scalar to specify a sample distance for x-axis.
    :param dy: float64
        single scalar to specify a sample distance for y-axis.
    :return: float64
        single scalar describing the wave number at which the maximum of energy flux.
    """
    k_f = np.linspace(0.1, 2 * np.pi, 1000)
    flux_k = [energy_flux(velocity_field, dx, dy, (k_f[i], 16)) for i in range(1000)]
    return k_f[np.argmax(flux_k)]


def double_laplace_rand(mu, sigma, size):
    """
    Generates samples from the double Laplace distribution with specified mean and standard deviation.
    :param mu: float
        the mean of distribution.
    :param sigma: float
        the standard deviation of distribution.
    :param size: tuple of ints
        shape of random numbers array.
    :return: array_like
        random numbers array of given shape.
    """
    dist = np.zeros(2*size[0]*size[1])
    for i in range(2*size[0]*size[1]):
        num = np.random.randint(1, 3)
        if num == 1:
            dist[i] = np.random.laplace(mu, sigma)
        if num == 2:
            dist[i] = np.random.laplace(-mu, sigma)
    return np.reshape(dist, (2, size[0], size[1]))


def dist_by_hist(hist, size):
    """
    Returns a random numbers array of the given shape, with a distribution similar to a histogram.
    :param hist: array_like
        given histogram of distribution of random numbers.
    :param size: tuple of ints
        shape of random numbers array.
    :return: array_like
        random numbers array of given shape and distribution.
    """
    dist = np.zeros(2*size[0]*size[1])
    dx = hist[1][1] - hist[1][0]
    cdf = [np.sum(hist[0][0:i]) * dx for i in range(len(hist[0]))]
    unif = np.random.uniform(0, 1, size=2*size[0]*size[1])
    ind = np.searchsorted(cdf, unif)
    np.place(ind, ind == 500, 499)
    end = int(*np.shape(hist[0]))
    for i in range(2*size[0]*size[1]):
        dist[i] = hist[1][0:end][ind[i] - 1] \
                  + ((hist[1][0:end][ind[i]] - hist[1][0:end][ind[i]-1]) / (cdf[ind[i]] - cdf[ind[i] - 1])) \
                  * (unif[i] - cdf[ind[i]-1])
    np.place(dist, np.isposinf(dist), hist[1][end])
    np.place(dist, np.isneginf(dist), hist[1][0])
    return np.reshape(dist, (2, size[0], size[1]))


def accuracy(func, velocity_field, error, **kwargs):
    """
    Returns calculation error of a given function for specific velocity field and its error.
    :param func: callable
        function for which the error is calculated.
    :param velocity_field: array_like
        3d input array describing the velocity field.
    :param error: array_like
        3d input array describing the error of velocity field.
    :param kwargs: tuple
        additional function arguments (example: filter scale and order of filter for energy flux).
    :return: float
        single scalar describing error of function for specific velocity field and its error.
    """
    delta_f = func(velocity_field+error, **kwargs) - func(velocity_field, **kwargs)
    return delta_f


def vel_error(shape, kind, mu=0.0, st_dev=1.0, hist=None):
    """
    Returns a random error of certain type.
    :param shape: tuple of int
        shape of output array.
    :param kind: string
        type of error distribution.
    :param mu: float64
        mean of error distribution.
    :param st_dev: float64
        standard deviation of error distribution.
    :param hist: histogram
    :return: array_like
        3d output array.
    """
    n_x, n_y = shape[0], shape[1]
    if kind == 'gauss':
        mis = np.random.normal(mu, st_dev, size=(2, n_x, n_y))
        return mis
    if kind == 'laplace':
        mis = np.random.laplace(mu, st_dev/np.sqrt(2), size=(2, n_x, n_y))
        return mis
    if kind == 'double_laplace':
        mis = double_laplace_rand(mu, st_dev/np.sqrt(2), (n_x, n_y))
        return mis
    if kind == 'cdf':
        mis = dist_by_hist(hist, (n_x, n_y))
        return mis


def func_error(k, func, velocity_field, shape, kind, mu=0.0, st_dev=1.0, hist=None, **kwargs):
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
    :param mu: float64
        mean of error distribution.
    :param kwargs: tuple
        additional function arguments (example: filter scale and order of filter for energy flux).
    :param hist: array_like
        amplitude of velocity error for each velocity vector.
    :return: array_like
        1d array containing absolute errors.
    """
    ac_abs = [accuracy(func, velocity_field,
                       vel_error(shape, kind, mu=mu, st_dev=st_dev, hist=hist), **kwargs) for i in range(k)]
    return np.array(ac_abs)


def mean(func, k, sigma, velocity_field, kind, **kwargs):
    """
    Returns the standard deviation dependence of mean of distribution.
    :param func: callable
        function for which the standard deviation dependence of mean of distribution is calculated.
    :param k: int
        number of samples of the error.
    :param sigma: array_like
        1d input array containing standard deviation of error distribution.
    :param velocity_field: array_like
        3d input array.
    :param kind: string
        type of error distribution.
    :param kwargs: tuple
        additional function arguments (example: filter scale and order of filter for energy flux).
    :return: array_like
        1d output array containing mean of distribution for each sigma element.
    """
    n_x, n_y = np.shape(velocity_field[1, :, :])
    return [np.mean(func_error(k, func, velocity_field, (n_x, n_y), kind, st_dev=sigma, **kwargs)) for sigma in sigma]


def standard_deviation(func, k, sigma, velocity_field, kind, **kwargs):
    """
    Returns the standard deviation dependence of standard deviation of distribution.
    :param func: callable
        function for which the standard deviation dependence of variance of distribution is calculated.
    :param k: int
        number of samples of the error.
    :param sigma: array_like
        1d input array containing standard deviation of error distribution.
    :param velocity_field: array_like
        3d input array.
    :param kind: string
        type of error distribution.
    :param kwargs: tuple
        additional function arguments (example: filter scale and order of filter for energy flux).
    :return: array_like
        1d output array containing standard deviation of distribution for each sigma element.
    """
    n_x, n_y = np.shape(velocity_field[1, :, :])
    return [np.std(func_error(k, func, velocity_field, (n_x, n_y), kind, st_dev=sigma, **kwargs)) for sigma in sigma]


def first_n_max(n, func):
    """
    Returns first n maximum of a given 2d function.
    :param n: int
        number of first peaks.
    :param func: array_like
        2d input array
    :return: list
        containing tuples of matrix indices corresponding to maximum.
    """
    index_of_max = []
    while len(index_of_max) < n:
        maximum = np.max(func)
        index = np.where(func == maximum)
        indexes = list(zip(index[0], index[1]))
        shape = np.array(indexes).shape[0]
        for j in range(shape):
            func[indexes[j]] = 0
            index_of_max.append(indexes[j])
    return index_of_max


def pic2vel(image_a, image_b):
    """
    Returns the velocity field obtained by multipass PIV algorithm.
    :param image_a: array_like
        a two dimensions array of integers containing grey levels of the first experimental PIV frame.
    :param image_b: array_like
        a two dimensions array of integers containing grey levels of the second experimental PIV frame.
    :return: array_like
        u : 2d np.ndarray
            a two-dimensional array containing the u velocity component, in pixels/frame.
        v : 2d np.ndarray
            a two-dimensional array containing the v velocity component, in pixels/frame.
        x : 2d np.ndarray
            a two-dimensional array containing the x coordinates of the interrogation window centers, in pixels.
        y : 2d np.ndarray
            a two-dimensional array containing the y coordinates of the interrogation window centers, in pixels.
    """
    settings = windef.PIVSettings()
    settings.num_iterations = 3
    settings.windowsizes = (64, 32, 32)
    settings.overlap = (32, 16, 16)
    settings.scaling_factor = 1
    settings.dt = 1
    settings.subpixel_method = 'gaussian'
    settings.interpolation_order = 3
    settings.sig2noise_validate = False
    settings.validation_first_pass = False
    settings.std_threshold = 7
    settings.min_max_u_disp = (-8, 8)
    settings.min_max_v_disp = (-8, 8)
    settings.median_threshold = 3
    settings.replace_vectors = True
    settings.filter_method = 'localmean'
    settings.max_filter_iteration = 4
    settings.filter_kernel_size = 3
    x, y, u, v, flags = windef.simple_multipass(image_a, image_b, settings)
    y = y[::-1, :]
    v = (-1) * v
    return np.array([u, v]), x, y


def deformation_field(velocity_field):
    """
    Returns the deformation field according to the given velocity field by Fourier filtering.
    :param velocity_field: array_like
        3d input array.
    :return: array_like
        3d output array.
    """
    fourier_vel_field = np.fft.fft2(velocity_field)
    shape_x = velocity_field.shape[2]
    shape_y = velocity_field.shape[1]
    m, n = np.meshgrid(np.arange(0, shape_x), np.arange(0, shape_y))
    num = int(np.round(shape_x * shape_y * 0.1))
    u_def = 0
    v_def = 0
    for (l, k) in first_n_max(num, np.abs(fourier_vel_field[0, :, :])):
        u_def += fourier_vel_field[0, l, k] \
                 * np.exp(1j * 2 * np.pi * (k * (m / m.shape[1]) + l * (n / n.shape[0])))
    for (l, k) in first_n_max(num, np.abs(fourier_vel_field[1, :, :])):
        v_def += fourier_vel_field[1, l, k] \
                 * np.exp(1j * 2 * np.pi * (k * (m / m.shape[1]) + l * (n / n.shape[0])))
    norm = np.sqrt(energy(velocity_field) / energy(np.array([u_def, v_def])))
    u_def = np.real(norm * u_def)
    v_def = np.real(norm * v_def)
    return np.array([u_def, v_def])


def deform_image(image, x, y, u, v):
    image_new = windef.deform_windows(image, x, y, -u * 0.1, v * 0.1)
    for i in range(9):
        image_new = windef.deform_windows(image_new, x, y, -u * 0.1, v * 0.1)
    return image_new


def pic2error(func, image_a, image_b, scaling_factor, dt, error_type='cdf', **kwargs):
    """
    Returns the error of the given velocity field function.
    :param func: callable
        velocity field function for which the error is calculated.
    :param image_a: array_like
        a two dimensions array of integers containing grey levels of the first experimental PIV frame.
    :param image_b: array_like
        a two dimensions array of integers containing grey levels of the second experimental PIV frame.
    :param scaling_factor: float
        the image scaling factor in pixels per centimeter.
    :param dt: float
        the time delay separating the two frames.
    :param error_type: string
        the generation type of velocity field error.
    :param kwargs: tuple
        additional function arguments (example: filter scale and order of filter for energy flux).
    :return: tuple
        func_value: float
            the value of the function of the velocity field obtained from the given frames.
        func_error_mean: float
            the mean of function error distribution.
        func_error_dev: float
            the standard deviation of function error distribution.
    """
    vel_field, x, y = pic2vel(image_a, image_b)
    dx = (x[0, 1] - x[0, 0]) / scaling_factor
    dy = (y[1, 0] - y[0, 0]) / scaling_factor
    (n_x, n_y) = (x.shape[0], x.shape[1])
    def_vel = deformation_field(vel_field)
    image_a_new = deform_image(image_a, x, y, def_vel[0, :, :], def_vel[1, :, :])
    image_b_new = deform_image(image_b, x, y, -def_vel[0, :, :], -def_vel[1, :, :])
    vel_field_art, x_art, y_art = pic2vel(image_a, image_a_new)
    vel_field_art_inv, x_art_inv, y_art_inv = pic2vel(image_b_new, image_b)
    delta_v = np.concatenate((def_vel - vel_field_art, def_vel - vel_field_art_inv)) / (scaling_factor * dt)
    mu = float(np.mean(delta_v))
    sigma = float(np.std(delta_v))
    dist = hst(delta_v.flatten(), bins=500, density="True")
    func_error_dist = func_error(1000, func, def_vel / (scaling_factor * dt), (n_x, n_y), error_type, mu=mu,
                                 st_dev=sigma, hist=dist, dx=dx, dy=dy, **kwargs)
    func_error_mean = np.mean(func_error_dist)
    func_error_dev = np.std(func_error_dist)
    func_value = func(vel_field / (scaling_factor * dt), dx=dx, dy=dy, **kwargs)
    plt.close()
    return func_value, func_error_mean, func_error_dev
