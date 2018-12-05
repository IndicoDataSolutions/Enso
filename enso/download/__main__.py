"""
Download all available datasets
"""
import glob
import importlib
import os.path
import runpy
import logging

if __name__ == "__main__":
    glob_path = os.path.join(os.path.dirname(__file__), "*.py")
    filenames = [
        os.path.basename(f) for f in glob.glob(glob_path)
        if not os.path.basename(f).startswith("_")
    ]
    for filename in filenames:
        module_name = filename.split('.')[0]
        print("\n$ python -m enso.download.{}".format(module_name))
        runpy.run_module("enso.download.{}".format(module_name), run_name='__main__')
