# Automated Olfactory Bulb Segmentation on High Resolutional T2-Weighted MRI


This repository contains the tool designed for segmenting the olfactory bulb on high-resolutional (0.7 or 0.8mm Isotropic) whole brain T2-Weighted MRI.

<img src="/images/pipeline.png" class="responsive" alt="Olfactory Bulb Pipeline" style="
	display: block;
	margin-left: auto;
	margin-right: auto;
  width: 100%;
    max-width: 600px;
    height: auto;
">

![](/images/pipeline.png){:class="img-responsive"}

* First publicly available deep learning pipeline to segment the olfactory bulbs (OBs) in sub-millimeter T2-weighted whole-brain MRI.

* Rigorous validation in the Rhineland Study - an ongoing large population-based cohort study - in terms of segmentation accuracy, stability and reliability of volume estimates, as well as sensitivity to replicate known OB volume associations (e.g. age effects).

* Good generalizability to an unseen heterogeneous independent dataset (the Human Connectome Project).

* Robustness even for individuals without apparent OBs, as can be encountered in large cohort studies.


## Tool installation
If the tool is run for the first time run the following steps:

 1. Open Terminal
 2. Change the current working directory to the location where you want the cloned directory 
 3. Type `git-clone https://github.com/Deep-MI/olf-bulb-segmentation.git`  or download .zip file from the github repository 
 4. Download the model weights from the following [link](https://nextcloud.dzne.de/index.php/s/QaYpocJn9HFN7jp) into the same location were the repository was cloned.
 5. Installed the python libraries mention on the requirements file (see [requirements.txt](./requirements.txt))  

**The tool also has a docker version**. Docker images are really easy to run and to make. They eliminate the challenges of setting up your code to work in different environments. Your Docker image should run the same no matter where it is running.
  For more information check the [README.md](./docker/README.md) in the docker directory

## Running the tool

The rotterdam directory contains all the source code and modules needed to run the scripts. 
A list of python libraries used within the code can be found in requirements.txt. 
The main script is called run_prediction.py within which certain options can be selected 
and set via the command line:


#### Required Arguments

#### Optional Arguments




## Reference

If you use this tool please cite:

Estrada, Santiago, et al. "Automated olfactory bulb segmentation on high resolutional T2-weighted MRI." NeuroImage (2021). [https://doi.org/10.1016/j.neuroimage.2021.118464](https://www.sciencedirect.com/science/article/pii/S1053811921007370)
```
@article{estrada2021automated,
  title={Automated Olfactory Bulb Segmentation on High Resolutional T2-Weighted MRI},
  author={Estrada, Santiago and Lu, Ran and Diers, Kersten and Zeng, Weiyi and Ehses, Philipp and St{\"o}cker, Tony and Breteler, Monique MB and Reuter, Martin},
  journal={NeuroImage},
  pages={118464},
  year={2021},
  publisher={Elsevier}
}

```

--------
For any questions and feedback, feel free to contact santiago.estrada(at).dzne.de<br/>

--------