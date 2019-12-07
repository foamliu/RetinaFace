import zipfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall('data')
    zip_ref.close()


if __name__ == "__main__":
    extract('data/retinaface_gt_v1.zip')