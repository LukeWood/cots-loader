# cots-loader

[The Easiest Way to Load the Tensorflow Great Barrier Reef Crown of Thorns Starfish Dataset!](https://www.kaggle.com/c/tensorflow-great-barrier-reef)

`cots-loader` loads the TensorFlow Great Barrier Reef Crown of Thorns (COTS) Starfish Dataset to a `tf.data.Dataset` where
each element is a tuple of Dense Tensors representing images and RaggedTensors representing the
bounding boxes contained in the images.

`cots-loader` supports all of the [bounding box formats available in KerasCV](https://keras.io/api/keras_cv/bounding_box/),
and loads bounding box Tensors as RaggedTensors by default.  This format natively fits the format 
required by the [KerasCV object detection API](https://lukewood.xyz/blog/sneak-peek-object-detection-api).

`cots-loader` requires use of the Kaggle API.

## Quickstart

Getting started with the COTS loader is as easy as:

```python
dataset = cots_loader.load(
  bounding_box_format="xywh", 
  split="train", 
  data_dir='tensorflow-great-barrier-reef',
  batch_size=16
)
```

And fitting a model to the dataset is as easy as:

```python
model = keras_cv.models.RetinaNet(
    classes=2,
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights="imagenet",
    include_rescaling=True,
)
model.backbone.trainable = False

loss = keras_cv.losses.ObjectDetectionLoss(
    classes=20,
    classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction="none"),
    box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
    reduction="auto",
)
model.compile(
    loss=loss,
    optimizer=optimizers.SGD(momentum=0.9, global_clipnorm=10.0)
    metrics=metrics,
)
```

## Setup

TODO(lukewood): write how to download the file and create a data directory.

## API

The cots-loader API supports the following arguments:

- *bounding_box_format*: any KerasCV supported bounding box format, specified in the [Keras documentation](https://keras.io/api/keras_cv/bounding_box/)
- *data_dir*: the directory holding the data.  If you do not have a data directory, please follow the [Setup](#setup) instructions.
- *batch_size*: batch size to use

## Roadmap

- [ ] Kaggle API Integration
- [ ] Guided setup (i.e. if the data directory is missing, give the user commands to run to set it up in Error messages)
- [ ] Guided Kaggle login flow
