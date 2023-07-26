# Face Age Prediction

This repository contains code for a machine learning project focused on developing a robust age estimation model using facial images. We employed densenet 121 and densenet 169, two advanced CNN architectures, and leveraged transfer learning to fine-tune them on a large dataset of facial images. Our objective was to create an accurate and practical solution for age estimation, applicable to real-world scenarios. We conducted an in-depth analysis of the model's performance on various data distributions, including different ethnicities, genders, and age groups.

### Training the Model using Densenet-121

To train the model using Densenet-121 architecture, run the following command:

```
python train.py --name my_model --feature_extractor dnet121
```

For utilizing pre-trained ImageNet weights, use the `--pretrained` flag. To freeze the feature extractor's weights, include the `--freeze_extractor` flag.

For a more detailed analysis of the models and experiments, please refer to the Experiments.ipynb notebook.

We welcome contributions and feedback to improve our Face Age Prediction model. Feel free to create pull requests or reach out with any suggestions. Happy coding!
