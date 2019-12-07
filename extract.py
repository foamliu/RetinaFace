import zipfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall('data')
    zip_ref.close()


if __name__ == "__main__":
    extract('data/retinaface_gt_v1.1.zip')
    extract('data/WIDER_train.zip')
    extract('data/WIDER_val.zip')
    extract('data/WIDER_test.zip')

