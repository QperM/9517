import numpy as np
import matplotlib.pyplot as plt
import os

def compute_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def main():
    y_test = np.load('base_model_unet/y_test.npy')
    y_pred = np.load('base_model_unet/y_pred.npy')
    y_pred_bin = (y_pred > 0.5).astype(np.uint8)
    iou = compute_iou(y_test, y_pred_bin)
    print(f'IoU (Jaccard) score: {iou:.4f}')
    # 可视化前5张
    for i in range(min(5, y_test.shape[0])):
        img = None
        # 尝试加载原图
        img_path = None
        # 如果你希望可视化原图，可以在训练脚本里保存 test_imgs 路径列表
        # 这里假设原图已保存为 base_model_unet/test_imgs.txt
        test_imgs_txt = 'base_model_unet/test_imgs.txt'
        if os.path.exists(test_imgs_txt):
            with open(test_imgs_txt, 'r') as f:
                test_imgs = [line.strip() for line in f.readlines()]
            import cv2
            img_path = test_imgs[i]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (y_test.shape[2], y_test.shape[1]))
            img = img.astype(np.float32) / 255.0
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.title('Image')
        if img is not None:
            plt.imshow(img[..., ::-1])
        else:
            plt.text(0.5, 0.5, 'No image', ha='center', va='center')
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.title('Ground Truth')
        plt.imshow(y_test[i], cmap='gray')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.title('Prediction')
        plt.imshow(y_pred_bin[i], cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()