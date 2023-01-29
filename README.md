# BiComp-DTA
Drug-target binding affinity prediction through complementary biological-related and compression-based featurization approach

## Data
We utilized four DTA datasets including Davis, Kiba, BindingDB, and PDBbind refine set. Davis and Kiba datasets were downloaded from here. BindingDB, and PDBbind datasets were downloaded from here, and here, respectively. It should be noted that you should register and login before downloading data files from the PDBbind repositories.
<br/>
Each dataset folder includes binding affinity (i.e. Y), protein sequences (i.e. proteins.txt), drug SMILES (i.e. ligands_can.txt), and encoded protein sequences (i.e. protVecLZMA1 and protVecSW) files, and a folder includes the train and test folds settings (i.e. folds).

## Requirements
Python <br/>
Tensorflow <br/>
Keras <br/>
Numpy <br/>

## Usage
For training and evaluation of the method, you can run the following script.
```
!python run_experiments.py --num_windows 128 32 \
                          --smi_window_lengths 4 8 16 \
                          --batch_size 704 \
                          --num_epoch 1000 \
                          --max_smi_len 85 \
                          --dataset_path 'data/davis/' \
                          --problem_type 1 \
```

## Cold-start
Under the constraints of cold-start, BiComp-DTA can only predict binding affinity from unseen protein, unseen drug and both of them. <br/>
To train protein cold-start change value of problem_type to 2, to train drug cold-start change value of problem_type to 3 and to train protein-drug cold-start change value of problem_type to 4. For example you can use the following script to train protein cold-start:
```
!python run_experiments.py --num_windows 128 32 \
                          --smi_window_lengths 4 8 16 \
                          --batch_size 704 \
                          --num_epoch 1000 \
                          --max_smi_len 85 \
                          --dataset_path 'data/davis/' \
                          --problem_type 2 \
```
