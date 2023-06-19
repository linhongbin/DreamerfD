import setuptools
import pathlib


setuptools.setup(
    name='efficient_dreamer',
    version='2.2.0',
    description='Mastering Atari with Discrete World Models',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=['efficient_dreamer', 'efficient_dreamer.common'],
    install_requires=[
         'ruamel.yaml',
        'tensorflow', 'tensorflow_probability'
        ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
