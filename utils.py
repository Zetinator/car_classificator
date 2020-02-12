from scipy.io import loadmat
import os
import shutil
import numpy as np


def create_dataset():

    car_meta = loadmat("./data/devkit/cars_meta.mat")

    idx2car = {}
    for idx, j in enumerate(range(len(car_meta["class_names"][0])), 1):
        idx2car[idx] = car_meta["class_names"][0][j][0]
    car2idx = {v: k for k, v in idx2car.items()}

    car_annos = loadmat("./data/devkit/cars_train_annos.mat")

    caridx2paths = {}
    for car_idx in idx2car.keys():
        caridx2paths[car_idx] = []

    for j in range(len(car_annos["annotations"][0])):
        car_idx = car_annos["annotations"][0][j][-2][0][0]
        caridx2paths[car_idx].append(car_annos["annotations"][0][j][-1][0].split("/")[-1])

    if os.path.exists("./data/data_in_class_folder"): shutil.rmtree("./data/data_in_class_folder")
    os.makedirs("./data/data_in_class_folder", exist_ok=True)
    for car in car2idx.keys():
        print("creating folder {}".format(car))
        if "/" in car:
            os.makedirs("./data/data_in_class_folder/{}".format(car.replace("/", "_")), exist_ok=True)
        else:
            os.makedirs("./data/data_in_class_folder/{}".format(car), exist_ok=True)
        for file in caridx2paths[car2idx[car]]:
            if "/" in car:
                shutil.copy("./data/cars_train/{}".format(file),
                            "./data/data_in_class_folder/{}/{}".format(car.replace("/", "_"), file))
            else:
                shutil.copy("./data/cars_train/{}".format(file), "./data/data_in_class_folder/{}/{}".format(car, file))

