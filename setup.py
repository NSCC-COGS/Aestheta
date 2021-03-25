from setuptools import setup
from pip.req import parse_requirements

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='Crystal',
    url='https://github.com/NSCC-COGS/Aestheta',
    author='NSCC-COGS',
    author_email='kevin.mcguigan@nscc.ca',
    # Needed to actually package something
    packages=['crystal'],
    # Needed for dependencies
    # install_requires=['numpy'],
    #https://stackoverflow.com/questions/14399534/reference-requirements-txt-for-the-install-requires-kwarg-in-setuptools-setup-py
    install_reqs = parse_requirements('requirements.txt', session='hack'),
    # *strongly* suggested for sharing
    version='1.0',
    # The license can be anything you like
    license='MIT',
    description='Geospatial Machine Learning Toolkit',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)