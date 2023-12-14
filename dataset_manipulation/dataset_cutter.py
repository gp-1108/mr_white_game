"""
  This file is meant to be used to reduce the size of a dataset composed of
  sentences in italian.
"""

import sys

def cut_dataset(input_file: str, output_file: str, max_lines: int):
  """
    Given a file containing a dataset, it returns a file with the same name
    but with a suffix "_cut" before the extension. The new file will contain
    at most max_lines lines.

    Args:
      - input_file: string name of the file containing the dataset
      - output_file: string name of the output file
      - max_lines: integer number of lines to be written in the output file

    Returns:
      - None
  """

  with open(input_file, 'r') as f:
    for line in f:
      if max_lines == 0:
        break
      max_lines -= 1
      with open(output_file, 'a') as output:
        output.write(line)

if __name__ == '__main__':
  if len(sys.argv) != 4:
    print('Usage: python3 dataset_cutter.py <input_file> <output_file> <max_lines>')
    sys.exit(1)

  input_file = sys.argv[1]
  output_file = sys.argv[2]
  max_lines = int(sys.argv[3])
  cut_dataset(input_file, output_file, max_lines)
