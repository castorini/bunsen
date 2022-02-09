# Bunsen
A whole flaming mess of utilities for PyTorch.

## Installation

1. Switch to an environment where the `pip` command is associated with a Python 3.8+ interpreter.
   - You're free to use virtualenv or Anaconda, though we recommend Anaconda.
2. Download CMake 3.19+ and g++ 7.5.0+.
3. Clone Bunsen:
```bash
git clone --recursive ssh://git@github.com/castorini/bunsen && cd bunsen
```
4. Build and install Bunsen:
```bash
cmake .  # you need an internet connection to download LibTorch

# Set the BUNSEN_PRODUCTION environment variable to 1 to skip development-time
# routines, like tensor type/size checking.

# This uses `pip install` to install the Bunsen Python library. 
cmake --build . -j 16
```

5. Check if Bunsen is installed:
```bash
python -c "import bunsen; bunsen.hello()"
```
