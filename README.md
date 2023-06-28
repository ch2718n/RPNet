# An efficient range-to-point algorithm for 3D semantic segmentation

### Abstract:

We proposed a novel range-to-point framework named RPNet for 3D semantic segmentation. RPNet consists of a range-based efficient Transformer encoder and a point-based lightweight MLP decoder. In addition, a range view completion algorithm was proposed to reduce the impact of null value pixels in the range view on segmentation. Experiments show that the proposed method achieves a mIoU of 64.2% and an inference speed of 40 FPS on the outdoor scene dataset SemanticKITTI.

### Prepare:

Download SemanticKITTI from [official web](http://www.semantic-kitti.org/dataset.html).  Put the [pretrained model](https://drive.google.com/file/d/1sSdiqRsRMhLJCfs0SydF7iKgeQNcXDZj/view?usp=drive_link) into ./models/segformer/ .