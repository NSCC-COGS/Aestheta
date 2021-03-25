from setuptools import setup

#https://stackoverflow.com/questions/14399534/reference-requirements-txt-for-the-install-requires-kwarg-in-setuptools-setup-py
'''
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements
    '''

# #testing
# parseTest = parse_requirements('requirements.txt', session='hack')
# print(parseTest)


# https://stackoverflow.com/questions/49689880/proper-way-to-parse-requirements-file-after-pip-upgrade-to-pip-10-x-x
'''
import pkg_resources
import pathlib

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]
'''

# print(install_requires)
# input('ok?')

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='Aestheta',
    url='https://github.com/NSCC-COGS/Aestheta',
    author='NSCC-COGS',
    author_email='kevin.mcguigan@nscc.ca',
    # Needed to actually package something
    packages=['Library'],
    # Needed for dependencies
    install_requires=['numpy','pyshp'],
    
    # install_requires = parse_requirements('requirements.txt', session='hack'), # for try 1
    # install_requires = install_requires, # for try 2

    # *strongly* suggested for sharing
    version='1.0',
    # The license can be anything you like
    license='MIT',
    description='Geospatial Machine Learning Toolkit',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)