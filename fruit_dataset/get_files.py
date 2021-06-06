from shutil import copyfile
from glob import glob
import os

for f in glob("test_zip/test/*.jpg"):
    dest = os.path.join('images', f.split('\\')[-1])
    copyfile(f, dest)

for f in glob("train_zip/train/*.jpg"):
    dest = os.path.join('images', f.split('\\')[-1])
    copyfile(f, dest)
