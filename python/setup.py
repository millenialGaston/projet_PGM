from setuptools import setup, find_packages
from setuptools.command.install import install as _install

class Install(_install):
  def run(self):
    _install.do_egg_install(self)
    import nltk
    nltk.download('brown')

setup(
  name="textGenerator",
  version="1.0",
  packages=find_packages(),
  scripts=['script.py'],
  cmdclass={'install' : Install},
  install_requires=['docutils>=0.3','numpy',
                    'pandas','unicode',
                    'string', 'torch',
                    'random','matplotlib',
                    'nltk'],
  setup_requires=['nltk'],

  package_data={
    '':['*.txt','*.rst','*.csv']
  },

  author="Frederic Boileau, Jimmy Leroux, Nicolas Laliberte"
)
