import setuptools

setuptools.setup(
    name="texgen",
    version="1.0.0",
    author="SDL Research",
    packages=setuptools.find_packages(),
    install_requires=['numpy>=1.19.1',
                      'torch>=1.6.0',
                      'spacy>=2.3.2',
                      'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz',
                      'tensorflow>=2.2.0',
                      'texar-pytorch @ git+https://github.com/asyml/texar-pytorch.git@6067c8c2a6ead04f02d8bed2b3d60a054fb1ba3b',
                      ],
    include_package_data=True
)
