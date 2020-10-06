from setuptools import setup, find_packages


with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='gprEmbedding',
    version='0.1.2',
    description='Useful tools to work with Elastic stack in Python',
    long_description_content_type="text/markdown",
    long_description=README,
    license='MIT',
    packages=find_packages(exclude='tests'),
    author='Clemens Hutter',
    author_email='mail@clemenshutter.de',
    keywords=['GaussianProcess', 'scikit-learn', 'Kernel', 'Embedding'],
    # url='https://github.com/ncthuc/elastictools',
    # download_url='https://pypi.org/project/elastictools/'
    install_requires = [
        'numpy',
        'scikit-learn'
    ]
)


if __name__ == '__main__':
    setup(**setup_args)

