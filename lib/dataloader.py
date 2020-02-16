import torchvision.datasets as dset
import torchvision.transforms as transforms
import os

class CycleDataset(dset.VisionDataset):
    def __init__(self, root, loader=dset.folder.default_loader, extensions=('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'),
                 transform=None, target_transform=None, is_valid_file=None):
        super(CycleDataset, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        samples = []
        for file_name in os.listdir(root):
            path = os.path.join(root, file_name)
            if file_name.lower().endswith(extensions):
                samples.append(path)

        self.loader = loader
        self.samples = samples

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample

    def __len__(self):
        return len(self.samples)