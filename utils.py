
import os, re
import torch
import torchvision
from skimage import io
from torch.utils.data import Dataset

class MILabeled(Dataset):
    def __init__(self, dir_path, transform=None):
        
        self.transform = transform
        self.dir_path  = dir_path
        self.transform = transform
        
        files = os.listdir(dir_path)
        self.labeled_images = list(filter(None, [self._parse_files(f) for f in files]))    
          
    def __getitem__(self, i):
        
        (label, ia) = self.labeled_images[i]
        
        it = self.transform(ia) if self.transform else ia
        
        return (label, it)
    
    def _parse_files(self, fn):
        
        pth = self.dir_path + '/' + fn
        #fn  = os.path.basename(pth)
        m   = re.search(r'_(\d)\.', fn)
        if m:
            label = int(m.group(1))
            f_arr  = io.imread(pth)
            return (label, f_arr)
            
        return None
        
    
    def __len__(self):
        return len(self.labeled_images)
    

class MLUnlabeled(Dataset):
    
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        
        files = os.listdir(dir_path)
        self.images = [ io.imread(dir_path+'/'+f) for f in files if re.search('.jpg$', f)]
    
    def __getitem__(self, i):
        
        ia = self.images[i]
        
        it = self.transform(ia) if self.transform else ia

        
        return it
    
    def __len__(self):
        return len(self.images)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, item):

        if type(item) is tuple:
            img, label = item
        else:
            img = item
            
        img = torch.from_numpy(img.transpose(2,0,1))
        
        return (img, label) if type(item) is tuple else img

