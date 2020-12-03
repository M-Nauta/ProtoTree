import requests
import tarfile

train_url = 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
test_url = 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz'
devkit_url = 'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
test_anno_url = 'http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat'
bb_anno_url = 'http://imagenet.stanford.edu/internal/car196/cars_annos.mat'

target_path_train = './data/cars/cars_train.tgz'
target_path_test = './data/cars/cars_test.tgz'
devkit_path = './data/cars/devkit'

response = requests.get(train_url, stream=True)
if response.status_code == 200:
    with open(target_path_train, 'wb') as f:
        f.write(response.raw.read())
tar = tarfile.open(target_path_train, "r:gz")
tar.extractall(path='./data/cars')
tar.close()

response = requests.get(test_url, stream=True)
if response.status_code == 200:
    with open(target_path_test, 'wb') as f:
        f.write(response.raw.read())
tar = tarfile.open(target_path_test, "r:gz")
tar.extractall(path='./data/cars')
tar.close()

response = requests.get(devkit_url, stream=True)
if response.status_code == 200:
    with open(devkit_path+'.tgz', 'wb') as f:
        f.write(response.raw.read())
tar = tarfile.open(devkit_path+'.tgz', "r:gz")
tar.extractall(path='./data/cars')
tar.close()

response = requests.get(test_anno_url, stream=True)
if response.status_code == 200:
    with open(devkit_path+'/cars_test_annos_withlabels.mat', 'wb') as f:
        f.write(response.raw.read())

response = requests.get(bb_anno_url, stream=True)
if response.status_code == 200:
    with open(devkit_path+'/cars_annos.mat', 'wb') as f:
        f.write(response.raw.read())




