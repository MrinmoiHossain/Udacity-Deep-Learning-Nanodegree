## Anaconda

- an open source distribution for Python designed for large-scale data
- able to simplify package management
- actually a distribution of software that comes with conda, Python, and over 150 scientific packages and their dependencies

### Managing Packages

- to install libraries and other software on your computer
- pip is the default package manager for Python libraries
- 

```bash
conda install numpy
conda install pandas
```

### Environments

- allow you to separate and isolate the packages you are using for different projects

```bash
conda create -n my_env numpy
```

### Conda Command

```bash
conda list
conda upgrade conda
conda upgrade --all
conda install package_name
conda remove package_name
conda create -n env_name list of packages
conda activate my_env
source deactivate
conda env export > file_name.yaml
conda env create -f file_name.yaml
conda env remove -n env_name
```


### Resources

* Anaconda: https://www.anaconda.com/distribution/#download-section
* MKL library: https://docs.continuum.io/mkl-optimizations/
* Official YAML Site: https://yaml.org/