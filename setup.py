from setuptools import find_packages, setup


setup(
    name='mindset-memory',
    version='0.1.0',
    description='Free-energy dynamics for provenance-preserving agent memory',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    python_requires='>=3.11',
    install_requires=[
        'pydantic>=2.7,<3',
        'numpy>=1.26,<3',
        'scipy>=1.11,<2',
    ],
)
