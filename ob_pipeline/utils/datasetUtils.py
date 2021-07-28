
# Copyright 2019 Population Health Sciences and Image Analysis, German Center for Neurodegenerative Diseases(DZNE)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch.utils.data.dataset import Dataset



class testDataset(Dataset):
    """
    Class for loading a img file with augmentations (transforms)
    """
    def __init__(self,img, transforms=None):

        try:
            self.images = img
            self.count = self.images.shape[0]
            self.transforms = transforms

        except Exception as e:
            print("Loading failed: {}".format(e))

    def __getitem__(self, index):

        img = self.images[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return {'image': img}

    def __len__(self):
        return self.count







