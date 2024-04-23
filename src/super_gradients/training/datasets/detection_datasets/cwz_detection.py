import os
import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Tuple, Dict, Union, Any, Optional

from super_gradients.common.registry.registry import register_dataset
from super_gradients.common.object_names import Datasets
from super_gradients.common.cache_handler.cache_manager import CacheManager

from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataset
from super_gradients.training.transforms.transforms import DetectionTransform
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat
from super_gradients.training.utils.detection_utils import change_bbox_bounds_for_image_size

MY_CLASSES = ['head', 'body']  # Define your classes

@register_dataset(Datasets.CWZCustomDetectionDataset)
class CWZCustomDetectionDataset(DetectionDataset):
    @staticmethod
    def img2labelfilepath(image_file_path: str) -> str:
        image_dir, image_filename = os.path.split(image_file_path)
        label_dir = image_dir.rsplit("images", 1)[0] + "labels" + image_dir.rsplit("images", 1)[1]
        label_file_path = os.path.join(label_dir, os.path.splitext(image_filename)[0] + '.txt')
        return label_file_path
    
    def __init__(self, dataset_filenames_txts: List[str], input_dim: Tuple[int, int],
                 transforms: List[DetectionTransform], max_num_samples: int = None,
                 class_inclusion_list: Optional[List[str]] = None, force_update_cache_file: Optional[bool] = False, **kwargs):
        self.dataset_filenames_txts = dataset_filenames_txts
        self.image_shapes = {}  # Dictionary to store image dimensions
        self.force_update_cache_file = force_update_cache_file
        super().__init__(data_dir="", input_dim=input_dim,
                         original_target_format=DetectionTargetsFormat.XYXY_LABEL,
                         max_num_samples=max_num_samples,
                         class_inclusion_list=class_inclusion_list, transforms=transforms,
                         all_classes_list=MY_CLASSES, **kwargs)
        

    def _setup_data_source(self) -> int:
        print("CWZCustomDetectionDataset: Setting up data source")
        self.samples_targets_tuples_list = []
        for dataset_filename_txt in self.dataset_filenames_txts:
            print(f"setup data source for {dataset_filename_txt}")
            cacheobj = CacheManager.get_cache_obj(dataset_filename_txt)
            cache_data = {}
            if self.force_update_cache_file:
                cacheobj.remove_cache()
            else:
                cache_data = cacheobj.load_cache_data()

            if cache_data:
                self.samples_targets_tuples_list.extend(cache_data.get('samples_targets', []))
                self.image_shapes.update(cache_data.get('image_shapes', {}))
            else:
                base_dir = os.path.dirname(dataset_filename_txt)
                with open(dataset_filename_txt, 'r') as file:
                    lines = file.read().splitlines()
                new_cache_data = {'samples_targets': [], 'image_shapes': {}}
                for line in tqdm(lines, desc="Processing Labels", unit="line"):
                    image_file_path = os.path.join(base_dir, line.strip())
                    _, ext = os.path.splitext(image_file_path)
                    if ext.lower() not in ['.jpeg', '.jpg', '.png']:
                        print(f"Skipping unsupported file format: {image_file_path} ({ext})")
                        continue  # Skip unsupported file formats
                    label_file_path = CWZCustomDetectionDataset.img2labelfilepath(image_file_path)

                    if os.path.exists(image_file_path) and os.path.exists(label_file_path):
                        self.samples_targets_tuples_list.append((image_file_path, label_file_path))
                        img = cv2.imread(image_file_path)
                        if img is not None:
                            new_cache_data['image_shapes'][len(new_cache_data['samples_targets'])] = img.shape[:2]
                        new_cache_data['samples_targets'].append((image_file_path, label_file_path))
                print(f"Cache {dataset_filename_txt} related file labels to {cacheobj.get_cache_path()}")
                cacheobj.save_cache_data(new_cache_data)

        return len(self.samples_targets_tuples_list)

    def _load_annotation(self, sample_id: int) -> Dict[str, Union[np.ndarray, Any]]:
        sample_path, target_path = self.samples_targets_tuples_list[sample_id]

        if os.path.exists(target_path):
            with open(target_path, 'r') as targets_file:
                lines = targets_file.read().splitlines()
                target = np.array([x.strip().split() for x in lines], dtype=np.float32)
        else:
            target = np.zeros((0, 5))  # If label file doesn't exist, return empty target
        
        img_height, img_width = self.get_image_shape(sample_id)
        
        if len(target) != 0:
            # target_height, target_width = self.input_dim
            res_target = np.zeros_like(target)
            res_target[:, 0] = target[:, 1] * img_width - target[:, 3] * img_width / 2  # X1
            res_target[:, 1] = target[:, 2] * img_height - target[:, 4] * img_height / 2  # Y1
            res_target[:, 2] = target[:, 1] * img_width + target[:, 3] * img_width / 2  # X2
            res_target[:, 3] = target[:, 2] * img_height + target[:, 4] * img_height / 2  # Y2
            res_target[:, 4] = target[:, 0]  # Class label    
        else:
            res_target = np.zeros((0, 5))
            
        labels = res_target[:, 4]
        boxes_xyxy = change_bbox_bounds_for_image_size(res_target[:, 0:4], img_shape=(img_height, img_width), inplace=False)
        mask = np.logical_and(boxes_xyxy[:, 2] >= boxes_xyxy[:, 0], boxes_xyxy[:, 3] >= boxes_xyxy[:, 1])
        boxes_xyxy = boxes_xyxy[mask]
            
        initial_img_shape = (img_height, img_width)
        if self.input_dim is not None:
            scale_factor = min(self.input_dim[0] / img_height, self.input_dim[1] / img_width)
            resized_img_shape = (int(img_height * scale_factor), int(img_width * scale_factor))
        else:
            resized_img_shape = initial_img_shape
            scale_factor = 1
        
        res_target = np.concatenate([boxes_xyxy * scale_factor, labels[:,None]], axis=1).astype(np.float32)
        ### draw debug image
        # img_height, img_width = self.get_image_shape(sample_id)
        # img = cv2.imread(sample_path)
        # for roi in res_target:
        #     x1, y1, x2, y2, cid = roi
        #     color = (0, 0, 255)
        #     if cid==0:
        #         color = (0, 255, 0)
        #     cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        
        annotation = {
            'img_path': sample_path,
            'target': res_target,
            'initial_img_shape': (img_height, img_width),
            'resized_img_shape': resized_img_shape
        }
        return annotation

    def get_image_shape(self, sample_id: int) -> Tuple[int, int]:
        sample_path, _ = self.samples_targets_tuples_list[sample_id]
        if sample_id not in self.image_shapes:
            img = cv2.imread(sample_path)
            if img is not None:
                self.image_shapes[sample_id] = img.shape[:2]
            else:
                self.image_shapes[sample_id] = (0, 0)  # Fallback if image can't be loaded
        return self.image_shapes[sample_id]