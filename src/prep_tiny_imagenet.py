"""Download Tiny-ImageNet-200 (64x64, 200 classes, 100k train / 10k val) and put the
validation images into class folders so train.py can read it with ImageFolder.
Usage: python prep_tiny_imagenet.py /content/data
"""
import os
import shutil
import sys
import urllib.request
import zipfile

URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
root = sys.argv[1] if len(sys.argv) > 1 else 'data'
os.makedirs(root, exist_ok=True)
zpath = os.path.join(root, 'tiny-imagenet-200.zip')
ds = os.path.join(root, 'tiny-imagenet-200')

if not os.path.isdir(ds):
    if not os.path.exists(zpath):
        print('downloading', URL); urllib.request.urlretrieve(URL, zpath)
    print('extracting'); zipfile.ZipFile(zpath).extractall(root)

# train/<wnid>/images/*.JPEG  ->  train/<wnid>/*.JPEG
for wnid in os.listdir(f'{ds}/train'):
    img_dir = f'{ds}/train/{wnid}/images'
    if os.path.isdir(img_dir):
        for f in os.listdir(img_dir):
            shutil.move(f'{img_dir}/{f}', f'{ds}/train/{wnid}/{f}')
        shutil.rmtree(img_dir)

# val/images/*.JPEG + val_annotations.txt -> val/<wnid>/*.JPEG
ann = f'{ds}/val/val_annotations.txt'
if os.path.exists(ann):
    for line in open(ann):
        fname, wnid = line.split('\t')[:2]
        os.makedirs(f'{ds}/val/{wnid}', exist_ok=True)
        shutil.move(f'{ds}/val/images/{fname}', f'{ds}/val/{wnid}/{fname}')
    shutil.rmtree(f'{ds}/val/images'); os.remove(ann)
print('ready:', ds, '| train classes:', len(os.listdir(f'{ds}/train')), '| val classes:', len(os.listdir(f'{ds}/val')))
