# HeaveEstimation

This project contains a Python script for estimating heave motion.

## Requirements

The script should be run in an IDE that supports interactive plotting, such as **Spyder** (Anaconda distribution recommended).

### Python Packages

The following Python libraries are required:

- time (standard library)
- threading (standard library)
- numpy
- matplotlib
- scipy
- collections (standard library)

You also need the following **custom classes** located in the same directory:

- `MotionMatreciesClass.py`
- `ShipMotionClass.py`
- `AttitudeClass.py`
- `HeaveClass.py`
- `EkfMatricesClass.py`

Make sure these `.py` files are available in the working directory.

### Additional Notes

- `matplotlib` must be configured to use the `TkAgg` backend. This is set automatically in the script:

  ```python
  matplotlib.use('TkAgg')
