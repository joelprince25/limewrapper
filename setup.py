from setuptools import setup, find_packages

setup(
      #Application Details
      name = 'limewrapper',
      version = '0.1',
      author = 'Joel Prince Varghese',
      author_email = 'joelprince25@gmail.com',
      description = 'Wrapper around the lime package to explain model predictions',
      include_package_data = True,

      #Dependencies
      packages = find_packages(),
      install_requires = [
        "pandas",
        "numpy"
        ]
      )
