## Start with Docker pytorch base
FROM nvidia/cuda:10.1-base-ubuntu16.04
ARG PYTHON_VERSION=3.6.9

# Install custom libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget git && \
         rm -rf /var/lib/apt/lists/*


# Install miniconda and needed python packages (for FastSurferCNN)
RUN wget -qO ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy scipy matplotlib h5py pandas scikit-image pyyaml==5.4.1 && \
     /opt/conda/bin/conda install -y -c pytorch cudatoolkit=10.1 "pytorch=1.5.0=py3.6_cuda10.1.243_cudnn7.6.3_0" torchvision=0.6.0 && \
     /opt/conda/bin/conda install -y -c conda-forge nibabel=3.1.0 nilearn=0.8.0 && \
     /opt/conda/bin/conda clean -ya
ENV PYTHONUNBUFFERED=0 \
    PATH=/opt/conda/bin:$PATH

ENV MPLCONFIGDIR=/opt/matplotlib
RUN chmod 777 /opt/matplotlib

WORKDIR "/"
# Remove old repository  folder if exists
RUN if [ -d "/olf-bulb-segmentation" ]; then rm -rf '/olf-bulb-segmentation'; fi

RUN git clone https://github.com/Deep-MI/olf-bulb-segmentation.git

WORKDIR "/olf-bulb-segmentation"

RUN git pull

ENTRYPOINT ["bash"]