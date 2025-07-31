import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 参数
IMAGE_DIR = 'USA_segmentation/NRG_images'
MASK_DIR = 'USA_segmentation/masks'
SAVE_DIR = 'base_model_unet'
IMG_SIZE = 256
TRAIN_SPLIT = 0.8
os.makedirs(SAVE_DIR, exist_ok=True)

# 数据读取
def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.shape[-1] == 3:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32) / 255.0
    else:
        raise ValueError(f"{path} 不是3通道图像！")

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)

# U-Net结构
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        def CBR(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.center = CBR(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(c), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        out = self.final(d1)
        return out

# 数据集
class SegDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = self.X[idx].transpose(2,0,1)  # HWC->CHW
        mask = self.y[idx][None, ...]       # H,W->1,H,W
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

def main():
    # 1. 读取文件名并划分
    image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png')])
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.png') or f.endswith('.tif') or f.endswith('.tiff')])
    assert len(image_files) == len(mask_files), '图像和mask数量不一致'
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_files, mask_files, train_size=TRAIN_SPLIT, random_state=42)
    with open(os.path.join(SAVE_DIR, 'test_imgs.txt'), 'w') as f:
        for p in test_imgs:
            f.write(p + '\n')
    # 2. 读取数据
    print('加载训练集...')
    X_train = np.stack([read_image(p) for p in train_imgs])
    y_train = np.stack([read_mask(p) for p in train_masks])
    print('加载测试集...')
    X_test = np.stack([read_image(p) for p in test_imgs])
    y_test = np.stack([read_mask(p) for p in test_masks])
    # 3. 训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = SegDataset(X_train, y_train)
    test_ds = SegDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(1, 16):
        model.train()
        epoch_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        print(f'Epoch {epoch} loss: {epoch_loss/len(train_ds):.4f}')
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'unet_model.pth'))
    print('模型已保存到 base_model_unet/unet_model.pth')
    # 4. 预测
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc='Predict'):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_pred = y_pred[:,0]  # (N,1,H,W)->(N,H,W)
    np.save(os.path.join(SAVE_DIR, 'y_pred.npy'), y_pred)
    np.save(os.path.join(SAVE_DIR, 'y_test.npy'), y_test)
    print('预测结果已保存到 base_model_unet/y_pred.npy')

if __name__ == '__main__':
    main()