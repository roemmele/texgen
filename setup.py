import setuptools

setuptools.setup(
    name="texgen",
    version="1.0.0",
    author="SDL Research",
    packages=setuptools.find_packages(),
    install_requires=['numpy>=1.19.1',
                      'torch>=1.6.0',
                      'spacy>=2.3.2',
                      'en_core_web_sm@ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0-py3-none-any.whl',
                      'tensorflow>=2.2.0',
                      'texar-pytorch>=0.1.3'
                      ],
    include_package_data=True
)
