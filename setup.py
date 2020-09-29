from setuptools import setup, find_packages
from codecs import open
from os import path

from mushroom_rl_benchmark import __version__

here = path.abspath(path.dirname(__file__))

requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

extras = {
    'gym': ['gym'],
    'atari': ['atari_py~=0.2.0', 'Pillow', 'opencv-python'],
    'box2d': ['box2d-py~=2.3.5'],
    'bullet': ['pybullet'],
    'mujoco': ['mujoco_py'],
    'plots': ['pyqtgraph']
}

all_deps = []
for group_name in extras:
    if group_name not in ['mujoco', 'plots']:
        all_deps += extras[group_name]
extras['all'] = all_deps

long_description = 'mushroom_rl_benchmark is a simple and easy to use tool to create scientific benchmarks for RL algorithms.' \
                   'It enables running experiments local (sequential, parallel) or with SLURM.'

setup(
    name='mushroom_rl_benchmark',
    version=__version__,
    description='A Python toolkit for creating and running benchmarks with MushroomRL.',
    long_description=long_description,
    url='https://github.com/MushroomRL/mushroom-rl-benchmark',
    author="Benedikt Voelker",
    author_email='mushroom@benedikt-voelker.de',
    license='MIT',
    packages=[package for package in find_packages()
              if package.startswith('mushroom_rl_benchmark')],
    zip_safe=False,
    install_requires=requires_list,
    extras_require=extras,
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 ]
)
