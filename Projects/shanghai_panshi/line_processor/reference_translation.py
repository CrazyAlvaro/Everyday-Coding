import pandas as pd
import numpy as np

def reference_frame_translation(dataframe, translation_params):
    """
    Translate and rotate points from the original reference frame to the target reference frame.

    Parameters:
        dataframe (pd.DataFrame): A DataFrame containing time series (x, y) in the original reference frame.
                                 The DataFrame should have columns ['x', 'y'] and optionally other columns.
        translation_params (tuple): A tuple (x_translation, y_translation, h) where:
                                    - x_translation, y_translation: Translation components.
                                    - h: Rotation angle in radians (counterclockwise).

    Returns:
        pd.DataFrame: A new DataFrame containing transformed (x, y) in the target reference frame, along with other original columns.
    """
    x_translation, y_translation, h = translation_params

    # Rotation matrix for counterclockwise rotation
    rotation_matrix = np.array([
        [np.cos(h), -np.sin(h)],
        [np.sin(h),  np.cos(h)]
    ])

    # Translate and rotate points
    original_points = dataframe[['x', 'y']].values
    translated_points = original_points - np.array([x_translation, y_translation])
    transformed_points = np.dot(translated_points, rotation_matrix.T)

    # Create a new DataFrame for the transformed points
    transformed_dataframe = dataframe.copy()
    transformed_dataframe[['x', 'y']] = transformed_points

    return transformed_dataframe

# Example usage:
# data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'cluster': [0, 1, 0]})
# translation_params = (1, 2, np.pi / 4)  # Example translation and rotation
# transformed_data = reference_frame_translation(data, translation_params)
# print(transformed_data)
