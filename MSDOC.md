
## Running the tool on multiple scans 

We provided a wrapper for the run_pipeline.py script so it can run consecutively on multiple scans. The wrapper is not a scheduler or runs process in parallel.
For the wrapper to work your input data has to be stored in one directory and organize in a tree structure with a T2 scan inside of each subject folder as illustrated below :
 

 ```
 #Input data Scheme                            
|-- my_dataset                                                                                     
    |-- Subject_1                                
        |-- T2_1.nii.gz                                                                                
    |-- Subject_2                                            
        |-- T2xx.nii.gz                                         
    |-- Subject_3                            
        |-- T2caipi.nii.gz                                      
    ...........                                     
    |-- Subject_xx                                    
        |-- T2.nii.gz  (Note . the T2 scan name can be different for each subject)                    
 ```  


**1- Prepare Participants file (participants.csv)** : the purpose of this file is to configure the participants scans 
that should be process. The file has a two compulsory column  that consist of subject_id and the T2 scan name. Note. the subject_id should match the folder name. For the
the example shown above the csv file is a follows :

subid | image
------------- | -------------
Subject_1 | T2_1.nii.gz
Subject_2 | T2xx.nii.gz
Subject_3 | T2caipi.nii.gz
Subject_xx | T2.nii.gz


**2- Run Wrapper**
The wrapper has the same arguments as the main scripts; the only difference is that it substitutes the in --in_img and --sub_id arguments by :
 
#### Required new arguments
 * `--sublist,-slist`: Subject list where the pipeline is going to be deployed
 * `--data_dir,-indir`: Data directory containing all scans

**Example**
```
# Run paper implementation in multiple scans 
python3 ms_wrapper.py -slits /path/to/participants.csv -out /directory/to/save/output -indir /path/to/my_dataset

# Run the pipeline natively at a different resolution to the default one (0.8 Isotropic) in multiple scans
python3 ms_wrapper.py -slits /path/to/participants.csv -out /directory/to/save/output -indir /path/to/my_dataset -ninter
    
# Run the pipeline at the default resolution (0.8 Isotropic) but mapped segmentation to the native input image resolution in multiple scans
python3 ms_wrapper.py -slits /path/to/participants.csv -out /directory/to/save/output -indir /path/to/my_dataset -ores

# Run paper implementation on cpu in multiple scans
python3 ms_wrapper.py -slits /path/to/participants.csv -out /directory/to/save/output -indir /path/to/my_dataset -ncuda

```


**Output**
 
When using the wrapper, an additional stats file unifying all the subjects stats is created @ */directory/to/save/output/obstats_table.csv*


