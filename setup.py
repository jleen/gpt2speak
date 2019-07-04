from setuptools import setup

setup(
    name='gpt2speak',
    version='0.1',
    entry_points={
        'console_scripts': ['speak = speak:speak']
    }
)
