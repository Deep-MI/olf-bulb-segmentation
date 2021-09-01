## Docker Version

We wrap our tool on a docker image, so there is no need to install any library dependencies or drivers, 
the only requirement is to have docker (CPU or GPU)  installed.

Prerequisites:

* Docker (https://docs.docker.com/install/)


## Tool installation 

 If the tool is run for the first time the docker image has to be created.  It can be created by pulling the docker image from [Dockerhub](https://hub.docker.com/r/deepmi/olfsegnet) 
 or building it from the repository. 
 
 **Installation from Dockerhub**
 1. Open terminal
 2. Pull the image from the Dockerhub by typing :
 
 ``` bash
 #GPU version
docker pull deepmi/olfsegnet:latest
```

 ``` bash
 #CPU version
docker pull deepmi/olfsegnet:cpu
```


For checking that the tool image was created correctly type on the terminal<br/>
`docker images`

it should appear a repository with the name **deepmi/olfsegnet** and the tag latest for GPU or cpu for CPU.

**Example** 
``` bash
REPOSITORY            TAG       IMAGE ID      CREATED     SIZE
deepmi/olfsegnet      latest    xxxxxxxx      xxxxxx      xxxx
deepmi/olfsegnet      cpu       xxxxxxxx      xxxxxx      xxxx    
```

**Note:** Both docker images for CPU and GPU can be created on the same machine. If you dont have access to a GPU is recommend to pull the CPU version.

 
**Installation from Repository**

 Run the following steps
 
 1. Open Terminal
 2. Change the current working directory to docker folder of the cloned repository
 3. From the download repository directory run on the terminal: 

* `bash build_docker_cpu.sh` for CPU (No GPU available)<br/> 
* `bash build_docker.sh` for GPU or CPU (GPU available ) <br/>

it should appear a repository with the name **olfsegnet** and the tag gpu for GPU or cpu for CPU.

**Example** 
``` bash
REPOSITORY     TAG       IMAGE ID      CREATED     SIZE
olfsegnet      gpu       xxxxxxxx      xxxxxx      xxxx
olfsegnet      cpu       xxxxxxxx      xxxxxx      xxxx    
```


### Running the tool

For executing tool  is necesary to configure the docker run options as follows :<br/>
```
#For gpu you need to pass the docker gpu flag
docker run --gpus all  [OPTIONS] IMAGE[:TAG] [ARGUMENTS]
#For Cpu
docker run [OPTIONS] IMAGE[:TAG] [ARGUMENTS]
```

After the Docker image is run you will have access to a bash terminal where you can run the tool as explain in the main [ReadMe file](../README.md).<br/>
**Note** A docker container doesnt have access to the system files so volumes has to be mounted using the  `--volume , -v` argument. You can check if the data is correclty mounted by checking the directoy on the bash terminal created by the docker image.
For example assuming the following command

```
docker run --gpus all  -v /my/local/dataset:/docker/data deepmi/olfsegnet:latest [ARGUMENTS]
```
In this case the `/my/local/dataset` directory was mount in the docker image into `/docker/data`. Then the present files can be checked in the docker image with the ls command

```
ls -l ../docker/data
```

if the files were mount correctly you should be able to see the name of the mounted files.


#### Options

We additionally recommend to use the following docker flags:<br/>
 * `--rm` : Automatically clean up the container and remove the file system when the container exits
 * `--user , -u `: Username or UID (format: <name|uid>[:<group|gid>])
 * `--name` : Assign a name to the container
 * `--volume , -v`: Bind mount a volume
 

**Running the tool on GPU in two steps**
    
    1. Run tool GPU image
    docker run --gpus all -it --rm --name ob_gpu -v /my/dataset/:/data/ -u $(id -u) deepmi/olfsegnet:latest

    2. Run pipeline
    python run_pipeline.py -in ../data/t2/image -out ../data/output -sid subject
    or
    python ms_wrapper.py -slist ../data/subjects.csv -indir ../data/ -out ../data/output

The output directory should be mounted to a local directory when running the docker image otherwise the created data will be only accesible inside the docker container and will be lost when the container is closed.
For more information in how to mount data check (https://docs.docker.com/storage/volumes/)
        










