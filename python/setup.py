from setuptools import setup, find_packages
setup(
  name="textGenerator",
  version="1.0",
  packages=find_packages(),
  scripts=['script.py'],
  install_requires=['docutils>=0.3','numpy','pandas','unicode','string',
                   'torch','random','matplotlib'],

  package_data={
    '':['*.txt','*.rst','*.csv']
  },

  author="Frederic Boileau, Jimmy Leroux, Nicolas Laliberte"
)
