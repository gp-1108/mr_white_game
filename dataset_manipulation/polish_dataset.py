import re
from Trie import Trie
import os
import sys

"""
  Author: Pietro Girotto
  This file is meant to be used to polish a dataset composed of sentences
  in italian. Sentences will be cleaned from:
  - punctuation
  - numbers
  - articles and prepositions
  - words longer than 20 characters
  - words shorter than 3 characters
  - words with less than 5 occurrences
  - words that start with 'http'

  The output will be a file with the same name of the input file, but with
  the suffix "_polished" before the extension.
"""

def polish_line(line: str):
  """
    Given a line of text, it returns the same line without punctuation,
    numbers, articles and prepositions, words that start with 'http'.

    Args:
      - line: string line of text to polish

    Returns:
      - string line of text polished
  """


  # Remapping accents to lowercase a-z
  line = line.lower()
  line = line.replace('à', 'a')
  line = line.replace('è', 'e')
  line = line.replace('é', 'e')
  line = line.replace('ì', 'i')
  line = line.replace('ò', 'o')
  line = line.replace('ù', 'u')

  # Remapping apostrophes to '
  # apostrophes = ['\u02B9', '\u02BB', '\u02BC', '\u02BD']
  apostrophes = ['’']
  for apostrophe in apostrophes:
    line = line.replace(apostrophe, '\'')

  # Removing articles and prepositions
  art_and_prep = [
    ' il ', ' lo ', ' la ', ' i ', ' gli ', ' le ', ' l\'', ' gl\'',
    ' un ', ' uno ', ' una ', ' un\'',
    ' di ', ' a ', ' da ', ' in ', ' con ', ' su ', ' per ', ' tra ', ' fra ',
    ' del ', ' dello ', ' dell\'', ' della ', ' dei ', ' degli ', ' delle ',
    ' al ', ' allo ', ' all\'', ' alla ', ' ai ', ' agli ', ' alle ',
    ' dal ', ' dallo ', ' dall\'', ' dalla ', ' dai ', ' dagli ', ' dalle ',
    ' nel ', ' nello ', ' nell\'', ' nella ', ' nei ', ' negli ', ' nelle ',
    ' sul ', ' sullo ', ' sull\'', ' sulla ', ' sui ', ' sugli ', ' sulle '
  ]
  for word in art_and_prep:
    line = line.replace(word, ' ')

  # Removing non-alphabetic characters
  pattern = re.compile(r'[^a-z ]')
  line = pattern.sub('', line)

  # Removing words longer than 20 characters
  words = line.split(' ')
  line = ''
  for word in words:
    if len(word) <= 20 and len(word) >= 3:
      line += word + ' '

  # Removing http links
  while (line.find('http') != -1):
    http_index = line.find('http')
    space_index = line.find(' ', http_index)
    line = line[:http_index] + line[space_index + 1:]
  
  return line


def main(file_name: str):
  """
    Main function of the script. It takes a file name as input and
    returns a file with the same name but with the suffix "_polished"
    before the extension.

    Args:
      - file_name: string name of the file to polish (.txt format)
  """

  # Given the file is usually very big, we will read it line by line
  # and write the polished lines to the output file as we go.
  occurrences = Trie()
  count = 0
  with open(file_name, 'r') as file:
    with open(file_name[:-4] + '_polished_tmp' + file_name[-4:], 'w') as output_file:
      # Each text file is composed of many lines of text.
      # a paragraph is ended with an empty line.
      whole_text = ''
      for line in file:
        # If the line is empty, it means we reached the end of a paragraph.
        # We can now write the whole paragraph to the output file.
        count += 1
        if count % 10000 == 0:
          print('Lines read: ' + str(count))

        if line == '\n' and len(whole_text) > 0:
          output_file.write(whole_text + '\n')
          whole_text = ''
          continue


        line = polish_line(line)
        whole_text += line

        if count >= 1E3: # Stopping at 1M lines for testing purposes
          output_file.write(whole_text + '\n')
          whole_text = ''
          break

        for word in line.split():
          occurrences.insert(word)
        
  
  print('Distinct words: ' + str(occurrences.distinct_words))
  
  # Now we have a Trie with all the words and their occurrences.
  # We can now remove the words with less than 5 occurrences.
  with open(file_name[:-4] + '_polished_tmp' + file_name[-4:], 'r') as file:
    with open(file_name[:-4] + '_polished' + file_name[-4:], 'w') as output_file:
      for line in file:
        new_line = ''
        for word in line.split():
          if occurrences.search(word) >= 3:
            new_line += word + ' '
        output_file.write(new_line[:-1] + '\n')
  
  # Removing the temporary file
  # os.remove(file_name[:-4] + '_polished_tmp' + file_name[-4:])


if __name__ == '__main__':
  main(sys.argv[1])