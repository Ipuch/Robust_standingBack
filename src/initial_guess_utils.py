import numpy as np
from scipy.interpolate import interp1d


def interpolate_array(input_array, target_size):
  """
  Interpolates a 2D arrayto (8, target_size).

  Args:
      input_array:
      target_size: The desired width of the output array.

  Returns:
      A NumPy array of shape (8, target_size) with interpolated values.
      Returns None if the input array's shape is not (8,21)
  """

  original_x = np.linspace(0, 1, input_array.shape[1]) # 21 points evenly spaced between 0 and 1
  new_x = np.linspace(0, 1, target_size) # target_size points evenly spaced between 0 and 1

  interpolated_array = np.empty((input_array.shape[0], target_size)) # Pre-allocate array to store output

  for i in range(input_array.shape[0]):  # Iterate over each row
      interp_func = interp1d(original_x, input_array[i], kind='linear', fill_value="extrapolate") # Create an interpolation function for the current row
      interpolated_array[i] = interp_func(new_x) # Interpolate the current row

  return interpolated_array