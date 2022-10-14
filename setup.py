from setuptools import find_packages, setup

with open('requirements.txt') as fh:
    requirements = [line.strip() for line in fh.readlines()]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='ml2dac',
    packages=find_packages('src'),
    package_dir={'':'src'},
    version='0.1.0',
    install_requires=requirements,
    description='Python Library for meta-learning approach to democratize automl for clustering',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='XXX',
    author_email='XXX@XXX.XXX',
    license='MIT',
    python_requires=">=3.9",
)
