import os
import pandas as pd


def generate_index_filename_mapping(img_dir, output_csv_file):
    """
    Generate a CSV file mapping from index to image file names in the given directory.

    Args:
        img_dir (str): The directory containing the images.
        output_csv_file (str): The path to the output CSV file.
    """
    filenames = os.listdir(img_dir)
    # Optional: Sort the filenames if needed
    filenames.sort()
    df = pd.DataFrame(filenames, columns=['filename'])
    df.to_csv(output_csv_file, index_label='index')


# Example usage
img_dir = 'data'
output_csv_file = 'annotations.csv'
generate_index_filename_mapping(img_dir, output_csv_file)
