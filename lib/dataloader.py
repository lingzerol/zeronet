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
        return sample

    def __len__(self):
        return len(self.samples)


class MultiDomainDataset(dset.VisionDataset):
    def __init__(self, root, loader=dset.folder.default_loader, extensions=('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'),
                 transform=None, target_transform=None, is_valid_file=None):
        super(MultiDomainDataset, self).__init__(root, transform=transform,
                                                 target_transform=target_transform)

        samples = []
        label_to_name = {}
        dirnum = 0
        for i, dirname in enumerate(os.listdir(root)):
            dirpath = os.path.join(root, dirname)
            if not os.path.isdir(dirpath) and not os.path.islink(dirpath):
                continue
            if os.path.islink(dirpath):
                dirpath = os.readlink(dirpath)
            label_to_name[dirname] = dirnum
            for filename in os.listdir(dirpath):
                filepath = os.path.join(dirpath, filename)
                if filename.lower().endswith(extensions):
                    samples.append([filepath, dirnum])
            dirnum += 1

        self.loader = loader
        self.samples = samples
        self.label_to_name = label_to_name

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def target_to_name(self, target=-1):
        if target == -1:
            return self.label_to_name
        return self.label_to_name[target]
