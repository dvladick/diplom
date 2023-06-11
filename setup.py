from setuptools import find_packages, setup
setup(
    name='piverror',
    packages=find_packages(include=['piverror']),
    version='0.1.0',
    description='Library for calculating error due to PIV',
    author='Dvornikov Vladislav',
    license='HSE',
    install_requires=['numpy', 'scipy', 'openpiv', 'matplotlib'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests'
)
