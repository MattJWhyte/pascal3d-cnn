# pascal3d-viewpoint-cnn

Repo for training/evaluating models against Pascal3D Imagenet dataset

## Config

In `scripts/config.py`, modify `PASCAL_DIR` to point to the absolute path of Pascal3d dataset on your computer.

## Running

Near the end of `scripts/network.py`, you will find a dictionary of identifiers for network models which can be executed from the command line.
To train a given network, run `python3 main.py [MODEL I.D.] [WIDTH] [HEIGHT]`, ensuring that `WIDTH` and `HEIGHT` correspond to the model's input.
