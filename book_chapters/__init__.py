# -*- coding: utf-8 -*-
"""
The figures of each chapter of the book, one function per chapter, in a
module per part. Importing this package registers every chapter with
book_figures, which runs them:

    python3 book_figures.py life          the figures of Chapter 10

Each function reads the runs, draws its figures with their captions and
recipes, and records the numbers its chapter quotes. The captions say what
is drawn; the recipes say how to draw it again from the runs by hand.
"""
from book_chapters import brains, cooperation, life, measures, part1, size, society, structure  # noqa: F401  (registers the chapters)
