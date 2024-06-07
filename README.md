# Computer Vision - PSL University

## Dataset

The dataset CUB-200-2011 is available at the following link: [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
We use a subset of the dataset, which is available at the following link: Limited at 30 classes

## Requirements

Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

The structure of the project is as follows:
```
.
├── raw-dataset
│   ├── train_images
│   ├── val_images
│   ├── test_images
│   └── classes_indexes.csv
├── dataset
│   ├── train
│   ├── val
│   ├── train_labels.csv
│   ├── val_labels.csv
│   └── test
├── src
│   ├── dataset.py
│   ├── builder.py
│   ├── models.py
│   ├── main.py
│   ├── inference.py
│   └── tools.py
├── figures
├── submissions
├── saved_models
├── README.md
└── requirements.txt
```

Configuration plots are available in the ```tools.py``` file.
```python
plt.style.use('default')
plt.rc('text', usetex=True) # Modify to False if LaTeX is not installed
plt.rc('font', family='sans-serif')
plt.rc('font', size=18)
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('lines', markersize=10)
```

Backbones available for the model are the following:
```python
model_names = [
            'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
            'swin_small_patch4_window7_224.ms_in22k_ft_in1k',
            'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
            'swin_large_patch4_window7_224.ms_in22k_ft_in1k'
        ]
```

To put all images in train, val and test folders, and generate label files, run the following command:
```bash
python3 src/dataset.py --raw_dataset [raw-data-folder] --output_dataset [output-folder] --classes_file [classes-indexes-file]
```

Bird detection and cropping can be done using the following command:
```bash
python3 ./src/cropped.py --input ./dataset/ --output ./cropped/
```

Generate caption images for the train, val and test folders:
```bash
python3 ./src/vlm/retrieval.py --train_dir ./dataset/train/ --val_dir ./dataset/val --test_dir ./dataset/test/
```

To train the model SwinC, run the following command:
```bash
python3 ./src/swex.py --train_csv ./dataset/train.csv --val_csv ./dataset/val.csv --train_dir ./cropped/train --val_dir ./cropped/val --train_text ./dataset/train_prompts.csv --val_text ./dataset/val_prompts.csv --batch_size 32 --num_workers 8 --num_epochs 10 --model_id 1
```

To test the model, run the following command:
```bash
python3 ./src/inference.py --test_dir ./cropped/test --num_classes 30 --model_id 1
```