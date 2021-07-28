
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

# IMPORTS
import numpy as np
import torch



##
class ToTensorTest(object):
    """
    Convert np.ndarrays in sample to Tensors.
    """

    def __call__(self, img):
        img = img.astype(np.float32)

        # Normalize and clamp between 0 and 1
        img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))

        return torch.from_numpy(img)

