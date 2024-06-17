import os
from os.path import join

import numpy as np

from data.ELD_dataset import metainfo

scene_list = list(range(1, 10 + 1))
basedir = "/mnt/s1/ly/data/ELD/"
cameras = ['CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']
suffixes = ['.CR2', '.CR2', '.nef', '.ARW']
train_ids = [2, 3, 7, 8, 12, 13]
val_ids = [4, 9, 14, 5, 10, 15]
ids_100 = [4, 9, 14]
ids_200 = [5, 10, 15]
gt_ids = [1, 6, 11, 16]
# scene-3/IMG_0014.CR2 scene-3/IMG_0016.CR2 10.0

gt_ids = np.array([1, 6, 11, 16])

# dicts = {"train": train_ids, "val": val_ids, "val_100": ids_100, "val_200": ids_200}
dicts = {"val_100": ids_100, "val_200": ids_200}

for key in dicts:
    ids = dicts[key]
    for camera, suffix in zip(cameras, suffixes):
        with open(f'ELD_{key}_{camera}.txt', 'w') as file:
            for scene_id in scene_list:
                scene = 'scene-{}'.format(scene_id)
                datadir = join(basedir, camera, scene)
                for id in ids:
                    input_path = join(datadir, 'IMG_{:04d}{}'.format(id, suffix))
                    ind = np.argmin(np.abs(id - gt_ids))
                    target_path = join(datadir, 'IMG_{:04d}{}'.format(gt_ids[ind], suffix))

                    iso, expo = metainfo(target_path)
                    target_expo = iso * expo
                    iso, expo = metainfo(input_path)
                    input_expo = iso * expo

                    ratio = target_expo / input_expo
                    input_name = join(scene, os.path.basename(input_path))
                    target_name = join(scene, os.path.basename(target_path))

                    line = input_name + " " + target_name + " " + str(ratio)
                    file.write(line + '\n')


