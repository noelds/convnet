from distutils.core import setup

setup(
    name='convnet',
    version='0.1.0',
    author='Nitish Srivastava',
    author_email='',
    packages=['cudamat'],
    url='https://github.com/TorontoDeepLearning/convnet',
    license='LICENSE.txt',
    description=' A C++ based GPU implementation of Convolutional Neural Nets.',
    long_description=open('README.md').read(),
    install_requires=[
        "h5py",
        "matplotlib",
        "numpy",
        "protobuf",
        "scipy",
    ],
)
