# `iskra` ✨ - Tensor Geometry Processing

![](docs/logo.webp)

This repository contains a lightweight geometry processing library that is meant to be a one-stop-shop for all of your geometric needs. Iskra is:
* modern and Python-first,
* simple by default, powerful when needed,
* fully differentiable (if needed),
* actievely maintained.

## Obtaining Iskra ✨

If you want to pull any of the notebooks in this repository, you will need to have [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage) installed on your system. If not, here are the instructions to help you get set up:
```
# Pick one of the following depending on your distribution:
sudo apt install git-lfs  # on Ubuntu
brew install git-lfs  # on MacOS

# Verify that the installation was successful:
git lfs install
```

Change into the cloned iskra directory and install it to your active environment using:
```
pip install .
```

## Development
Lastly, if you plan on contributing, you will need the development dependencies and to compile the C++ extensions in editable mode.
This can be done by running the following:
```
conda env create -f environment.yaml
conda env update -f environment-dev.yaml
conda activate iskra
pip install --no-build-isolation -Ceditable.rebuild=true -ve .
```

## FAQ
- Why the name? 
    - Iskra means “spark” in Serbo-Croatian: a spark enables using (a) torch. We also expect our system to be the spark that ignites exciting research in geometry. Most of all, I think it sounds cool to say.