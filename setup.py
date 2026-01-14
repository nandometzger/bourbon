from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='bourbon',
    version='1.1.2',
    description='Bourbon: Distilled Population Maps',
    author='Nando Metzger',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'bourbon-predict=bourbon.predict_from_coords:main',
            'bourbon-timeseries=bourbon.predict_timeseries:main',
        ],
    },
    url='https://github.com/nandometzger/bourbon',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
