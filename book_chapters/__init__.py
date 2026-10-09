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
import importlib
import pkgutil

# Every module of the package, so a new one is registered without being listed.
for _module in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module.name}")
