from piverror import errorfunction
import numpy as np


n = 100
x, y = np.meshgrid(np.linspace(-np.pi, np.pi, n), np.linspace(-np.pi, np.pi, n))
v_x = np.zeros((n, n))
v_y = 6*np.ones((n, n))
v = np.array([v_x, v_y])
norm = np.sum(v ** 2) / n ** 2
v = v / np.sqrt(norm)
h = np.linspace(-np.pi, np.pi, n)[1] - np.linspace(-np.pi, np.pi, n)[0]

v_x_0 = -np.sin(x) * np.cos(y)
v_y_0 = np.sin(y) * np.cos(x)
v0 = np.array([v_x_0, v_y_0])
norm0 = np.sum(v0 ** 2) / n ** 2
v0 = v0 / np.sqrt(norm0)

np.random.seed(1)
error_v = errorfunction.vel_error((n, n), 'gauss', 0.05)
error_v0 = errorfunction.vel_error((n, n), 'uniform', 0.1)

energy_error = [0.00172486, 0.00205277, 0.00168371, 0.00144664, 0.00269484, 0.00131055, 0.00192585,
                0.00158908, 0.00165683, 0.00124036]

enstrophy_error = [0.0587275, 0.0578663, 0.05649499, 0.05914524, 0.06017222, 0.06063226, 0.05766637,
                   0.0597067, 0.05798932, 0.05675752]


def func_error_test(k, func, velocity_field, shape, kind, st_dev, dx=0, dy=0):
    ac_abs = np.zeros(k,)
    np.random.seed(1)
    for i in range(k):
        ac_abs[i] = errorfunction.accuracy(func, velocity_field, errorfunction.vel_error(shape, kind, st_dev) * velocity_field, dx, dy)
    return ac_abs


def test_energy():
    assert np.allclose(errorfunction.energy(v0), 0.5)
    assert np.allclose(errorfunction.energy(v), 0.5)


def test_enstrophy():
    assert np.allclose(errorfunction.enstrophy(v, h, h), 0)


def test_accuracy():
    assert np.allclose(errorfunction.accuracy(errorfunction.energy, v, error_v), 0.0029719427399993448)
    assert np.allclose(errorfunction.accuracy(errorfunction.enstrophy, v0, error_v0, h, h), 0.11295035016368349)


def test_func_error():
    assert np.allclose(func_error_test(10, errorfunction.energy, v, (n, n), 'gauss', 0.05), energy_error)
    assert np.allclose(func_error_test(10, errorfunction.enstrophy, v0, (n, n), 'uniform', 0.1, h, h),
                       enstrophy_error)

