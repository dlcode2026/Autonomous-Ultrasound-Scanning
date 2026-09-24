import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import re
import math

def extract_all_numeric_values_from_file(filename):
    numeric_values = []

    with open(filename, 'r') as file:
        lines = file.readlines()

        for line in lines:
            # Find all numeric values
            numbers = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?", line)

            # Filter out any empty strings and convert valid numbers to float
            valid_numbers = [num for num in numbers if num.strip() != '']
            numeric_values.extend([float(num) for num in valid_numbers])

    return numeric_values


def min_max_normalize(value, min_val, max_val, eps=1e-8):
    return (value - min_val) / (max_val - min_val + eps)


def standardize(value, mean, std, eps=1e-8):
    return (value - mean) / (std + eps)


def get_all_dataset_paths(base_image_dir, base_label_dir, exclude_folders=None):
    dataset_pairs = []
    
    if exclude_folders is None:
        exclude_folders = []
    
    # Get all date folders, excluding the ones in exclude_folders
    date_folders = [f for f in os.listdir(base_image_dir) 
                   if os.path.isdir(os.path.join(base_image_dir, f)) and f not in exclude_folders]
    
    for date in date_folders:
        # Ensure the corresponding date folder exists in labels
        date_label_path = os.path.join(base_label_dir, date)
        if not os.path.exists(date_label_path):
            print(f"Warning: No matching label folder for date {date}")
            continue
            
        # Get all view folders within this date
        view_folders = [f for f in os.listdir(os.path.join(base_image_dir, date))
                       if os.path.isdir(os.path.join(base_image_dir, date, f))]
        
        for view in view_folders:
            # Ensure the corresponding view folder exists in labels
            view_label_path = os.path.join(date_label_path, view)
            if not os.path.exists(view_label_path):
                print(f"Warning: No matching label folder for {date}/{view}")
                continue
                
            # Add this valid pair to the list
            image_folder = os.path.join(base_image_dir, date, view)
            label_folder = os.path.join(base_label_dir, date, view)
            
            # Check if there are actually images in this folder
            if len([f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]) > 0:
                dataset_pairs.append((image_folder, label_folder))
                print(f"Found valid dataset pair: {date}/{view}")
    
    return dataset_pairs


class UltrasoundDataset(Dataset):
    
    def __init__(self, image_folder, label_folder, transform=None, 
                normalization_params=None, normalization_method='min_max',
                calculate_params=True):

        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.normalization_method = normalization_method
        
        all_files = os.listdir(image_folder)
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
        self.image_filenames = [f for f in all_files 
                            if os.path.splitext(f)[1].lower() in image_extensions
                            and not f.startswith('.')]
        
        print(f"Found {len(self.image_filenames)} valid image files.")
        
        self.normalization_params = normalization_params
        
        if normalization_method != 'none' and normalization_params is None and calculate_params:
            self.normalization_params = self.calculate_normalization_params()
        
        if self.normalization_params:
            print(f"Using {normalization_method} normalization with the following parameters:")
            for key, value in self.normalization_params.items():
                try:
                    if isinstance(value, (list, np.ndarray)) and len(value) > 6:
                        min_val = np.min(value)
                        max_val = np.max(value)
                        print(f"  {key}: shape={np.shape(value)}, min={min_val:.4f}, max={max_val:.4f}")
                    elif isinstance(value, (list, np.ndarray)):
                        print(f"  {key}: {value}")
                    elif isinstance(value, dict):
                        print(f"  {key}: {type(value)} with keys {list(value.keys())}")
                    else:
                        print(f"  {key}: {type(value)}")
                except Exception as e:
                    print(f"  {key}: {type(value)} (couldn't display: {str(e)})")

    def normalize_values(self, values):
        if self.normalization_method == 'none' or self.normalization_params is None:
            return values
                
        normalized_values = np.zeros_like(values, dtype=np.float32)
        
        if self.normalization_method == 'min_max':
            for i in range(len(values)):
                normalized_values[i] = min_max_normalize(
                    values[i], 
                    self.normalization_params['min'][i], 
                    self.normalization_params['max'][i]
                )
                
        elif self.normalization_method == 'z_score':
            for i in range(len(values)):
                normalized_values[i] = standardize(
                    values[i], 
                    self.normalization_params['mean'][i], 
                    self.normalization_params['std'][i]
                )
                
        elif self.normalization_method == 'robust':
            for i in range(len(values)):
                normalized_values[i] = (values[i] - self.normalization_params['median'][i]) / (self.normalization_params['iqr'][i] + 1e-8)
                
        elif self.normalization_method == 'domain_aware':
            if 'feature_groups' not in self.normalization_params:
                feature_groups = {
                    'forces': list(range(6)),
                    'position': list(range(6, 9)),
                    'rotation': list(range(9, 12))
                }
            else:
                feature_groups = self.normalization_params['feature_groups']
                
            for group_name, indices in feature_groups.items():
                for i in indices:
                    if i >= len(values):
                        continue
                        
                    if group_name == 'forces':
                        # Robust scaling for forces
                        normalized_values[i] = (values[i] - self.normalization_params['median'][i]) / (self.normalization_params['iqr'][i] + 1e-8)
                        # Scale to [0, 1] range
                        normalized_values[i] = (normalized_values[i] + 2) / 4
                        
                    elif group_name == 'position':
                        # Min-max normalization for position
                        normalized_values[i] = min_max_normalize(
                            values[i], 
                            self.normalization_params['min'][i], 
                            self.normalization_params['max'][i]
                        )
                        
                    elif group_name == 'rotation':
                        # Special handling for rotation angles
                        normalized_values[i] = 2 * min_max_normalize(
                            values[i], 
                            self.normalization_params['min'][i], 
                            self.normalization_params['max'][i]
                        ) - 1
        
        return normalized_values

    def calculate_normalization_params(self):
        
        all_combined_values = []
        
        for idx in range(len(self.image_filenames)):
            try:
                image_filename = self.image_filenames[idx]
                image_index = image_filename.split('.')[0].replace('img', '')
                label_filename = f"PA_{image_index}.txt"
                label_path = os.path.join(self.label_folder, label_filename)
                
                numeric_values = extract_all_numeric_values_from_file(label_path)
                numeric_values = numeric_values[1:]  # Skip first value
                numeric_values = numeric_values[:-10]  # Skip last 10 values
                
                # Split the values: first 6 are forces, next 12 are rotation matrix
                force_values = numeric_values[:6]
                matrix_values = numeric_values[6:18]
                
                # Convert matrix to 6DOF
                dof_values = self.matrix_to_6dof(matrix_values)
                
                # Combine force and 6DOF values
                combined_values = np.concatenate([force_values, dof_values])
                all_combined_values.append(combined_values)
                
            except Exception as e:
                print(f"Error processing label {label_filename}: {str(e)}")
                continue
                
        # Convert to numpy array
        all_combined_values = np.array(all_combined_values)
        
        # Calculate statistics for each dimension
        dim_min = np.min(all_combined_values, axis=0)
        dim_max = np.max(all_combined_values, axis=0)
        dim_mean = np.mean(all_combined_values, axis=0)
        dim_std = np.std(all_combined_values, axis=0)
        
        # Calculate median and quartiles
        dim_median = np.median(all_combined_values, axis=0)
        dim_q25 = np.percentile(all_combined_values, 25, axis=0)
        dim_q75 = np.percentile(all_combined_values, 75, axis=0)
        dim_iqr = dim_q75 - dim_q25
        
        # Create feature groups
        feature_groups = {
            'forces': [0, 1, 2, 3, 4, 5],
            'position': [6, 7, 8],
            'rotation': [9, 10, 11]
        }
        
        return {
            'min': dim_min,
            'max': dim_max,
            'mean': dim_mean,
            'std': dim_std,
            'median': dim_median,
            'q25': dim_q25,
            'q75': dim_q75,
            'iqr': dim_iqr,
            'feature_groups': feature_groups,
            'n_samples': len(all_combined_values)
        }

    def rotation_matrix_to_euler_angles(self, R):
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        if sy > 1e-6:
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
            roll = math.atan2(R[2, 1], R[2, 2])
        else:
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(-R[0, 1], R[1, 1])
            roll = 0

        return np.array([roll, pitch, yaw])

    def matrix_to_6dof(self, matrix_values):

        matrix = np.array(matrix_values).reshape(3, 4)
        position = matrix[:, 3]
        rotation_matrix = matrix[:, :3]
        euler_angles = self.rotation_matrix_to_euler_angles(rotation_matrix)
        return np.concatenate([position, euler_angles])

    def denormalize_values(self, normalized_values):
        if self.normalization_method == 'none' or self.normalization_params is None:
            return normalized_values
            
        if isinstance(normalized_values, torch.Tensor):
            values_np = normalized_values.numpy()
        else:
            values_np = normalized_values
            
        denormalized_values = np.zeros_like(values_np, dtype=np.float32)
        
        if self.normalization_method == 'min_max':
            for i in range(len(values_np)):
                min_val = self.normalization_params['min'][i]
                max_val = self.normalization_params['max'][i]
                denormalized_values[i] = values_np[i] * (max_val - min_val) + min_val
                
        elif self.normalization_method == 'z_score':
            for i in range(len(values_np)):
                mean = self.normalization_params['mean'][i]
                std = self.normalization_params['std'][i]
                denormalized_values[i] = values_np[i] * std + mean
                
        elif self.normalization_method == 'robust':
            for i in range(len(values_np)):
                median = self.normalization_params['median'][i]
                iqr = self.normalization_params['iqr'][i]
                denormalized_values[i] = values_np[i] * iqr + median
                
        elif self.normalization_method == 'domain_aware':
            for group_name, indices in self.normalization_params['feature_groups'].items():
                for i in indices:
                    if group_name == 'forces':
                        robust_value = values_np[i] * 4 - 2
                        denormalized_values[i] = robust_value * self.normalization_params['iqr'][i] + self.normalization_params['median'][i]
                        
                    elif group_name == 'position':
                        min_val = self.normalization_params['min'][i]
                        max_val = self.normalization_params['max'][i]
                        denormalized_values[i] = values_np[i] * (max_val - min_val) + min_val
                        
                    elif group_name == 'rotation':
                        min_val = self.normalization_params['min'][i]
                        max_val = self.normalization_params['max'][i]
                        normalized_0_1 = (values_np[i] + 1) / 2
                        denormalized_values[i] = normalized_0_1 * (max_val - min_val) + min_val
        
        return denormalized_values

    def get_raw_item(self, idx):
        try:
            img_name = os.path.join(self.image_folder, self.image_filenames[idx])
            img = Image.open(img_name)

            image_filename = self.image_filenames[idx]
            image_index = image_filename.split('.')[0].replace('img', '')
            label_filename = f"PA_{image_index}.txt"
            label_path = os.path.join(self.label_folder, label_filename)

            numeric_values = extract_all_numeric_values_from_file(label_path)
            numeric_values = numeric_values[1:]
            numeric_values = numeric_values[:-10]

            force_values = numeric_values[:6]
            matrix_values = numeric_values[6:18]
            dof_values = self.matrix_to_6dof(matrix_values)
            combined_values = np.concatenate([force_values, dof_values])

            if self.transform:
                img = self.transform(img)

            return img, combined_values
            
        except Exception as e:
            print(f"Error processing image {self.image_filenames[idx]}: {str(e)}")
            raise e

    def save_normalization_params(self, filepath):
        if self.normalization_params is None:
            print("No normalization parameters to save.")
            return
            
        np.savez(filepath, **self.normalization_params)
        print(f"Saved normalization parameters to {filepath}")

    @staticmethod
    def load_normalization_params(filepath):
        data = np.load(filepath, allow_pickle=True)
        params = {}
        for key in data.files:
            params[key] = data[key]
        return params

    @staticmethod
    def calculate_normalization_params_from_values(all_combined_values):        
        all_combined_values = np.array(all_combined_values)
        
        dim_min = np.min(all_combined_values, axis=0)
        dim_max = np.max(all_combined_values, axis=0)
        dim_mean = np.mean(all_combined_values, axis=0)
        dim_std = np.std(all_combined_values, axis=0)
        
        dim_median = np.median(all_combined_values, axis=0)
        dim_q25 = np.percentile(all_combined_values, 25, axis=0)
        dim_q75 = np.percentile(all_combined_values, 75, axis=0)
        dim_iqr = dim_q75 - dim_q25
        
        feature_groups = {
            'forces': [0, 1, 2, 3, 4, 5],
            'position': [6, 7, 8],
            'rotation': [9, 10, 11]
        }
        
        return {
            'min': dim_min,
            'max': dim_max,
            'mean': dim_mean,
            'std': dim_std,
            'median': dim_median,
            'q25': dim_q25,
            'q75': dim_q75,
            'iqr': dim_iqr,
            'feature_groups': feature_groups,
            'n_samples': len(all_combined_values)
        }

    @staticmethod
    def save_normalization_params_static(filepath, params):
        np.savez(filepath, **params)
        print(f"Saved normalization parameters to {filepath}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.image_folder, self.image_filenames[idx])
            img = Image.open(img_name)

            image_filename = self.image_filenames[idx]
            image_index = image_filename.split('.')[0].replace('img', '')
            label_filename = f"PA_{image_index}.txt"
            label_path = os.path.join(self.label_folder, label_filename)

            numeric_values = extract_all_numeric_values_from_file(label_path)
            numeric_values = numeric_values[1:]
            numeric_values = numeric_values[:-10]

            force_values = numeric_values[:6]
            matrix_values = numeric_values[6:18]
            dof_values = self.matrix_to_6dof(matrix_values)
            combined_values = np.concatenate([force_values, dof_values])

            normalized_values = self.normalize_values(combined_values)
            label_tensor = torch.tensor(normalized_values, dtype=torch.float32)

            if self.transform:
                img = self.transform(img)

            return img, label_tensor
            
        except Exception as e:
            print(f"Error processing image {self.image_filenames[idx]}: {str(e)}")
            raise e


class MultiUltrasoundDataset(Dataset):

    def __init__(self, base_image_dir, base_label_dir, transform=None, 
                normalization_method='domain_aware', calculate_params=True,
                exclude_folders=None):

        self.base_image_dir = base_image_dir
        self.base_label_dir = base_label_dir
        self.transform = transform
        self.normalization_method = normalization_method
        self.exclude_folders = exclude_folders
        
        # Get all valid dataset pairs
        self.dataset_pairs = get_all_dataset_paths(base_image_dir, base_label_dir, exclude_folders)
        
        if len(self.dataset_pairs) == 0:
            raise ValueError("No valid dataset pairs found")
            
        print(f"Found {len(self.dataset_pairs)} valid dataset pairs")
        
        self.datasets = []
        
        if calculate_params and normalization_method != 'none':
            print("Calculating normalization parameters from all datasets")
            all_values = []
            
            for image_folder, label_folder in self.dataset_pairs:
                temp_dataset = UltrasoundDataset(
                    image_folder=image_folder,
                    label_folder=label_folder,
                    transform=transform,
                    normalization_method='none',
                    calculate_params=False
                )
                
                sample_size = min(len(temp_dataset), 100)
                indices = torch.randperm(len(temp_dataset))[:sample_size]
                
                for idx in indices:
                    _, raw_values = temp_dataset.get_raw_item(idx)
                    all_values.append(raw_values)
            
            self.normalization_params = UltrasoundDataset.calculate_normalization_params_from_values(all_values)
        else:
            self.normalization_params = None
        
        # Create the actual datasets with global normalization parameters
        for image_folder, label_folder in self.dataset_pairs:
            dataset = UltrasoundDataset(
                image_folder=image_folder,
                label_folder=label_folder,
                transform=transform,
                normalization_params=self.normalization_params,
                normalization_method=normalization_method,
                calculate_params=False
            )
            self.datasets.append(dataset)
            print(f"Added dataset with {len(dataset)} samples from {image_folder}")
        
        self.dataset_offsets = [0]
        for dataset in self.datasets:
            self.dataset_offsets.append(self.dataset_offsets[-1] + len(dataset))
    
    def __len__(self):
        return self.dataset_offsets[-1]
    
    def __getitem__(self, idx):
        dataset_idx = next(i for i, offset in enumerate(self.dataset_offsets[1:], 1) 
                          if offset > idx) - 1
        local_idx = idx - self.dataset_offsets[dataset_idx]
        return self.datasets[dataset_idx][local_idx]
    
    def get_normalization_params(self):
        return self.normalization_params
    
    def save_normalization_params(self, filepath):
        if self.normalization_params is None:
            print("No normalization parameters to save.")
            return
            
        UltrasoundDataset.save_normalization_params_static(filepath, self.normalization_params)
        print(f"Saved normalization parameters to {filepath}")
