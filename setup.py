from setuptools import setup
import os 


def read_requirements_file(filename):
    # source: https://github.com/ikostrikov/jaxrl/blob/main/setup.py#L11 
    req_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]



setup(name='ddpm_pytorch',
      version='0.0.12',
      description='A minimal implementation of denoising diffusion probabilistic models in pytorch.',
      url='https://github.com/rosikand/ddpm-pytorch',
      author='Rohan Sikand',
      author_email='rsikand29@gmail.com',
      license='MIT',
      packages=['ddpm_pytorch'],
      install_requires=read_requirements_file('requirements.txt'),
      zip_safe=False)