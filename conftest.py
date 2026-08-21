"""
Makes the repository root importable when pytest collects tests/.

pytest inserts the rootdir into sys.path when a conftest.py sits there, which
is all this file is for. The tests also bootstrap their own path so they can
be run directly without pytest at all.
"""
