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

To put all images in train, val and test folders, and generate label files, run the following command:
```bash
python3 src/dataset.py --raw_dataset [raw-data-folder] --output_dataset [output-folder] --classes_file [classes-indexes-file]
```

To train the model, run the following command:
```bash
python3 ./src/main.py --train_csv [train-csv-file] --val_csv [val-csv-file] --train_dir [train-folder] --val_dir [val-folder] --batch_size [batch-size] --num_workers [num-workers] --num_epochs [num-epochs] --model_name [model-name]
```

To test the model, run the following command:
```bash
python3 ./src/inference.py --test_dir [test-folder] --model_name [model-name] --num_classes [num-classes]
```