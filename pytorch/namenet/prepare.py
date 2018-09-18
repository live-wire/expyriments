from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)

# print('Data files loaded.', findFiles('data/names/*.txt'))


import unicodedata
import string

# Adding project root to path
import sys
sys.path.append('..')

from utils.preprocess_utils import unicodeToAscii


def fetchData():
    category_lines = {}
    all_categories = []

    # Read a file and split into lines
    def readLines(filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicodeToAscii(line) for line in lines]

    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    return (category_lines, all_categories, n_categories)