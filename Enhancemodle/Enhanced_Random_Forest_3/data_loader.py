import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

class EnhancedForestDataLoader:
    """Enhanced Aerial forest image data loader for Random Forest"""
    
    def __init__(self, data_path="USA_segmentation"):
        self.data_path = data_path
        self.rgb_path = os.path.join(data_path, "RGB_images")
        self.nrg_path = os.path.join(data_path, "NRG_images")
        self.mask_path = os.path.join(data_path, "masks")
        
        # Get all image files
        self.rgb_files = self._get_image_files(self.rgb_path, "RGB_")
        self.nrg_files = self._get_image_files(self.nrg_path, "NRG_")
        self.mask_files = self._get_image_files(self.mask_path, "mask_")
        
        print(f"Found {len(self.rgb_files)} RGB images")
        print(f"Found {len(self.nrg_files)} NRG images")
        print(f"Found {len(self.mask_files)} mask images")
    
    def _get_image_files(self, path, prefix):
        """Get image files in the specified path"""
        if not os.path.exists(path):
            return []
        
        files = [f for f in os.listdir(path) if f.startswith(prefix) and f.endswith('.png')]
        return sorted(files)
    
    def load_image(self, file_path, grayscale=False):
        """Load image with robust error handling"""
        try:
            # Try using PIL first (better for Unicode paths)
            if grayscale:
                img = Image.open(file_path).convert('L')
                img = np.array(img)
            else:
                img = Image.open(file_path).convert('RGB')
                img = np.array(img)
            return img
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            # Fallback to OpenCV
            try:
                if grayscale:
                    img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(str(file_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
            except Exception as e2:
                print(f"OpenCV also failed for {file_path}: {e2}")
                return None
    
    def load_pair(self, rgb_file, nrg_file, mask_file):
        """Load image pair and mask with error handling"""
        rgb_path = os.path.join(self.rgb_path, rgb_file)
        nrg_path = os.path.join(self.nrg_path, nrg_file)
        mask_path = os.path.join(self.mask_path, mask_file)
        
        rgb_img = self.load_image(rgb_path)
        nrg_img = self.load_image(nrg_path)
        mask_img = self.load_image(mask_path, grayscale=True)
        
        # Check if all images loaded successfully
        if rgb_img is None or nrg_img is None or mask_img is None:
            print(f"Failed to load image pair: {rgb_file}, {nrg_file}, {mask_file}")
            return None, None, None
        
        return rgb_img, nrg_img, mask_img
    
    def extract_enhanced_features(self, rgb_img, nrg_img, mask_img, patch_size=32):
        """Extract enhanced features for Random Forest"""
        features = []
        labels = []
        
        h, w = rgb_img.shape[:2]
        
        # Sliding window to extract features
        for i in range(0, h - patch_size, patch_size // 2):
            for j in range(0, w - patch_size, patch_size // 2):
                # Extract RGB and NRG patches
                rgb_patch = rgb_img[i:i+patch_size, j:j+patch_size]
                nrg_patch = nrg_img[i:i+patch_size, j:j+patch_size]
                mask_patch = mask_img[i:i+patch_size, j:j+patch_size]
                
                # Compute enhanced features
                feature_vector = self._compute_enhanced_features(rgb_patch, nrg_patch)
                
                # Determine label (if there are enough dead tree pixels in the mask, mark as positive)
                label = 1 if np.sum(mask_patch > 0) > (patch_size * patch_size * 0.1) else 0
                
                features.append(feature_vector)
                labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def _compute_enhanced_features(self, rgb_patch, nrg_patch):
        """Compute enhanced features for an image patch"""
        features = []
        
        # 1. Basic RGB features
        rgb_mean = np.mean(rgb_patch, axis=(0, 1))
        rgb_std = np.std(rgb_patch, axis=(0, 1))
        rgb_median = np.median(rgb_patch, axis=(0, 1))
        rgb_min = np.min(rgb_patch, axis=(0, 1))
        rgb_max = np.max(rgb_patch, axis=(0, 1))
        
        # 2. NRG features
        nrg_mean = np.mean(nrg_patch, axis=(0, 1))
        nrg_std = np.std(nrg_patch, axis=(0, 1))
        nrg_median = np.median(nrg_patch, axis=(0, 1))
        
        # 3. Spectral indices
        if nrg_patch.shape[2] >= 4:
            nir = nrg_patch[:, :, 3]
            red = nrg_patch[:, :, 0]
            green = nrg_patch[:, :, 1]
            
            # NDVI (Normalized Difference Vegetation Index)
            ndvi = np.mean((nir - red) / (nir + red + 1e-8))
            
            # GNDVI (Green Normalized Difference Vegetation Index)
            gndvi = np.mean((nir - green) / (nir + green + 1e-8))
            
            # EVI (Enhanced Vegetation Index)
            evi = np.mean(2.5 * (nir - red) / (nir + 6 * red - 7.5 * green + 1 + 1e-8))
        else:
            ndvi = gndvi = evi = 0.0
        
        # 4. Texture features
        gray_patch = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2GRAY)
        
        # GLCM features (simplified)
        glcm_features = self._compute_glcm_features(gray_patch)
        
        # LBP features
        lbp_features = self._compute_lbp_features(gray_patch)
        
        # 5. Edge features
        edge_features = self._compute_edge_features(gray_patch)
        
        # 6. Color histogram features
        hist_features = self._compute_histogram_features(rgb_patch, nrg_patch)
        
        # 7. Statistical moments
        moments = self._compute_statistical_moments(gray_patch)
        
        # Combine all features
        features.extend(rgb_mean)
        features.extend(rgb_std)
        features.extend(rgb_median)
        features.extend(rgb_min)
        features.extend(rgb_max)
        features.extend(nrg_mean)
        features.extend(nrg_std)
        features.extend(nrg_median)
        features.extend([ndvi, gndvi, evi])
        features.extend(glcm_features)
        features.extend(lbp_features)
        features.extend(edge_features)
        features.extend(hist_features)
        features.extend(moments)
        
        return np.array(features)
    
    def _compute_glcm_features(self, gray_patch):
        """Compute simplified GLCM features"""
        # Local variance
        local_var = np.var(gray_patch)
        
        # Local entropy
        hist, _ = np.histogram(gray_patch, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        
        # Contrast
        contrast = np.std(gray_patch)
        
        # Homogeneity (simplified)
        homogeneity = np.mean(1 / (1 + gray_patch**2))
        
        return [local_var, entropy, contrast, homogeneity]
    
    def _compute_lbp_features(self, gray_patch):
        """Compute LBP features"""
        lbp = self._compute_lbp(gray_patch)
        lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        lbp_hist = lbp_hist / np.sum(lbp_hist)
        
        # LBP statistics
        lbp_mean = np.mean(lbp)
        lbp_std = np.std(lbp)
        lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-8))
        
        return np.concatenate([lbp_hist, [lbp_mean, lbp_std, lbp_entropy]])
    
    def _compute_lbp(self, gray_patch):
        """Compute Local Binary Pattern"""
        lbp = np.zeros_like(gray_patch)
        for i in range(1, gray_patch.shape[0] - 1):
            for j in range(1, gray_patch.shape[1] - 1):
                center = gray_patch[i, j]
                code = 0
                code |= (gray_patch[i-1, j-1] > center) << 7
                code |= (gray_patch[i-1, j] > center) << 6
                code |= (gray_patch[i-1, j+1] > center) << 5
                code |= (gray_patch[i, j+1] > center) << 4
                code |= (gray_patch[i+1, j+1] > center) << 3
                code |= (gray_patch[i+1, j] > center) << 2
                code |= (gray_patch[i+1, j-1] > center) << 1
                code |= (gray_patch[i, j-1] > center) << 0
                lbp[i, j] = code
        return lbp
    
    def _compute_edge_features(self, gray_patch):
        """Compute edge features"""
        # Sobel gradients
        grad_x = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_direction = np.arctan2(grad_y, grad_x)
        
        # Canny edges
        edges = cv2.Canny(gray_patch, 50, 150)
        
        # Edge statistics
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        grad_mag_mean = np.mean(grad_magnitude)
        grad_mag_std = np.std(grad_magnitude)
        grad_dir_mean = np.mean(grad_direction)
        grad_dir_std = np.std(grad_direction)
        
        return [edge_density, grad_mag_mean, grad_mag_std, grad_dir_mean, grad_dir_std]
    
    def _compute_histogram_features(self, rgb_patch, nrg_patch):
        """Compute color histogram features"""
        features = []
        
        # RGB histogram
        try:
            rgb_hist = cv2.calcHist([rgb_patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            rgb_hist = rgb_hist.flatten() / (np.sum(rgb_hist) + 1e-8)
        except:
            rgb_hist = np.zeros(512)
        
        # NRG histogram
        try:
            nrg_hist = cv2.calcHist([nrg_patch], [0, 1, 2, 3], None, [6, 6, 6, 6], 
                                   [0, 256, 0, 256, 0, 256, 0, 256])
            nrg_hist = nrg_hist.flatten() / (np.sum(nrg_hist) + 1e-8)
        except:
            nrg_hist = np.zeros(1296)
        
        features.extend(rgb_hist)
        features.extend(nrg_hist)
        
        return features
    
    def _compute_statistical_moments(self, gray_patch):
        """Compute statistical moments"""
        # Central moments
        mean = np.mean(gray_patch)
        variance = np.var(gray_patch)
        skewness = self._compute_skewness(gray_patch, mean)
        kurtosis = self._compute_kurtosis(gray_patch, mean)
        
        return [mean, variance, skewness, kurtosis]
    
    def _compute_skewness(self, data, mean):
        """Compute skewness"""
        n = len(data.flatten())
        std = np.std(data)
        if std == 0:
            return 0
        return np.sum(((data - mean) / std) ** 3) / n
    
    def _compute_kurtosis(self, data, mean):
        """Compute kurtosis"""
        n = len(data.flatten())
        std = np.std(data)
        if std == 0:
            return 0
        return np.sum(((data - mean) / std) ** 4) / n - 3
    
    def prepare_dataset(self, test_size=0.2, random_state=42):
        """Prepare train and test datasets"""
        print("Preparing enhanced dataset...")
        
        all_features = []
        all_labels = []
        
        # Find images with corresponding masks
        available_pairs = []
        for rgb_file in self.rgb_files:
            base_name = rgb_file.replace("RGB_", "")
            nrg_file = f"NRG_{base_name}"
            mask_file = f"mask_{base_name}"
            
            if (nrg_file in self.nrg_files and mask_file in self.mask_files):
                available_pairs.append((rgb_file, nrg_file, mask_file))
        
        print(f"Found {len(available_pairs)} complete image pairs")
        
        # Process each image pair
        for rgb_file, nrg_file, mask_file in tqdm(available_pairs[:10]):  # Limit for speed
            try:
                rgb_img, nrg_img, mask_img = self.load_pair(rgb_file, nrg_file, mask_file)
                features, labels = self.extract_enhanced_features(rgb_img, nrg_img, mask_img)
                
                all_features.extend(features)
                all_labels.extend(labels)
                
            except Exception as e:
                print(f"Error processing {rgb_file}: {e}")
                continue
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels, dtype=np.int32)
        
        print(f"Enhanced feature matrix shape: {X.shape}")
        if len(y) > 0:
            print(f"Label distribution: {np.bincount(y)}")
        else:
            print("Warning: No features extracted.")
            return None, None, None, None
        
        # Split train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def visualize_sample(self, rgb_file, nrg_file, mask_file):
        """Visualize sample images"""
        rgb_img, nrg_img, mask_img = self.load_pair(rgb_file, nrg_file, mask_file)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(rgb_img)
        axes[0].set_title('RGB Image')
        axes[0].axis('off')
        
        axes[1].imshow(nrg_img[:,:,:3])  # Show first 3 channels
        axes[1].set_title('NRG Image (RGB channels)')
        axes[1].axis('off')
        
        axes[2].imshow(mask_img, cmap='gray')
        axes[2].set_title('Mask')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return rgb_img, nrg_img, mask_img 