# RetinaFace

A PyTorch implementation of [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641).

## Performance


## Dataset

1. Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.

2. Download annotations (face bounding boxes & five facial landmarks) from baidu cloud or dropbox

### Introduction

MS-Celeb-1M dataset for training, 3,804,846 faces over 85,164 identities.


## Dependencies
- Python 3.6.8
- PyTorch 1.3.0

## Usage

### 
Extract images, scan them, to get bounding boxes and landmarks:
```bash
$ python extract.py
$ python pre_process.py
```
