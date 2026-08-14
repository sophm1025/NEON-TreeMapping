import os
import random
import rasterio
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

c = 0

# ==================== Block-wise resampling and alignment function (BigTIFF + progress bar) ====================
def resample_to_target_blockwise(src_path, target_path, out_path, resampling=Resampling.nearest, block_size=2048):
    with rasterio.open(src_path) as src, rasterio.open(target_path) as tgt:
        profile = tgt.profile.copy()
        profile.update(count=src.count, dtype=src.dtypes[0], BIGTIFF='YES')

        with rasterio.open(out_path, 'w', **profile) as dst:
            total_blocks = ((tgt.height + block_size - 1)//block_size) * ((tgt.width + block_size -1)//block_size)
            pbar = tqdm(total=total_blocks, desc=f"Resampling {os.path.basename(out_path)}")
            for y in range(0, tgt.height, block_size):
                h = min(block_size, tgt.height - y)
                for x in range(0, tgt.width, block_size):
                    w = min(block_size, tgt.width - x)
                    window = Window(x, y, w, h)
                    dest_block = np.zeros((src.count, h, w), dtype=src.dtypes[0])
                    try:
                        reproject(
                            source=rasterio.band(src, list(range(1, src.count+1))),
                            destination=dest_block,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=rasterio.windows.transform(window, tgt.transform),
                            dst_crs=tgt.crs,
                            resampling=resampling
                        )
                        dst.write(dest_block, window=window)
                    except Exception as e:
                        print(f"Warning: failed to resample window {window}: {e}")
                    pbar.update(1)
            pbar.close()


# ==================== Dataset class ====================
class TreePatchDataset(data.Dataset):
    def __init__(self, rgb_path, mask_path, patch_size=1024, stride=768, samples=10000):
        self.rgb_path = rgb_path
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.stride = stride
        self.samples = samples

        self.rgb = rasterio.open(rgb_path)
        self.mask = rasterio.open(mask_path)
        assert self.rgb.width == self.mask.width and self.rgb.height == self.mask.height

        self.H, self.W = self.rgb.height, self.rgb.width
        self.patches = []
        for y in range(0, self.H - patch_size, stride):
            for x in range(0, self.W - patch_size, stride):
                self.patches.append((x, y))

        random.shuffle(self.patches)
        if samples < len(self.patches):
            self.patches = self.patches[:samples]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x, y = self.patches[idx]
        window = Window(x, y, self.patch_size, self.patch_size)
        try:
            rgb = self.rgb.read(window=window)
            mask = self.mask.read(1, window=window)
        except rasterio.errors.RasterioIOError as e:
            # print(f"Warning: cannot read window {window}, skipping.")
            return None

        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = (rgb / 255.).astype(np.float32)
        rgb = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        boxes_list = []
        valid_masks = []
        for obj_id in obj_ids:
            m = (mask == obj_id)
            pos = np.where(m)
            if pos[0].size == 0 or pos[1].size == 0:
                continue
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes_list.append([xmin, ymin, xmax, ymax])
            valid_masks.append(m)

        if len(boxes_list) == 0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            masks = torch.zeros((0,self.patch_size,self.patch_size), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
            masks = torch.as_tensor(np.array(valid_masks), dtype=torch.uint8)
            labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": torch.tensor([idx])}
        return rgb, target


# ==================== collate function ====================
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))


# ==================== Model definition ====================
def get_model(num_classes=2):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


# ==================== Training function ====================
def train(rgb_path, mask_path, workdir, epochs=10, batch_size=2, workers=0, patch=1024, stride=768, samples=10000):
    dataset = TreePatchDataset(rgb_path, mask_path, patch_size=patch, stride=stride, samples=samples)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_skip_none)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    os.makedirs(workdir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            if batch is None:
                continue
            images, targets = batch
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()
        print(f"Epoch {epoch}, Loss: {epoch_loss/len(loader):.4f}")
        torch.save(model.state_dict(), os.path.join(workdir, f"model_epoch{epoch}.pth"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--chm", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patch", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()

    aligned_mask = os.path.join(args.workdir, 'mask_aligned.tif')
    aligned_chm = os.path.join(args.workdir, 'chm_aligned.tif')
    os.makedirs(args.workdir, exist_ok=True)

    # print("Resampling mask to RGB resolution (blockwise, BigTIFF)...")
    # resample_to_target_blockwise(args.mask, args.rgb, aligned_mask, resampling=Resampling.nearest, block_size=2048)
    # print("Resampling CHM to RGB resolution (blockwise, BigTIFF)...")
    # resample_to_target_blockwise(args.chm, args.rgb, aligned_chm, resampling=Resampling.bilinear, block_size=2048)

    train(args.rgb, aligned_mask, args.workdir, args.epochs, args.batch_size, args.workers, args.patch, args.stride, args.samples)
    print("skip:", c)
