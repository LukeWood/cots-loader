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

## API

The cots-loader API works by loading the dataset from a directory specified in 
`data_dir`.  

```
```

## Roadmap

- [ ] Kaggle API Integration
- [ ] Guided Kaggle login flow
