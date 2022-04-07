# base image is heavy but guaranteed to work well with tensorflow and nvidia GPU
FROM nvcr.io/nvidia/tensorflow:22.03-tf2-py3
# some commands like 'source' assume shell is bash, not sh
SHELL ["/bin/bash", "-c"]

LABEL Description="learned pose estimation environment" Version="0.1"

## install common packages
RUN apt-get update -y && apt-get install -y \
    sudo \
    apt-utils \
    unzip htop \
    tmux \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean

## add user to avoid being root in the container. lpe = learned pose estimation
RUN useradd -ms /bin/bash lpe
RUN adduser lpe sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

#
# ========== User settings and additional programs =========
#
USER lpe
WORKDIR /home/lpe
ENV PATH="$PATH:/home/lpe/.local/bin"

## Install jupyter notebook
RUN pip3 install --user --upgrade \
  jupyter jupyterlab notebook ruamel.yaml \
  matplotlib==3.2  bqplot ipywidgets voila setuptools pyyaml
# Louis: pin matplotlib 3.2 for compatibility with neptune (matplotlib latest version is 3.4 and does not work)

## Install nbextensions for convenient snippets
# the last three lines are a temporary fix to a version mismatch with nbconvert https://github.com/ipython-contrib/jupyter_contrib_nbextensions/issues/1529
RUN pip3 install --user jupyter_contrib_nbextensions \
&& jupyter contrib nbextension install --user \
&& jupyter nbextension enable --user snippets/main \
&& pip3 uninstall jupyter-latex-envs -y \
&& sed -i 's/template_path/template_paths/g' /home/lpe/.local/lib/python3.8/site-packages/jupyter_contrib_nbextensions/nbconvert_support/toc2.py \
&& sed -i 's/template_path/template_paths/g' /home/lpe/.local/lib/python3.8/site-packages/jupyter_contrib_nbextensions/nbconvert_support/exporter_inliner.py

RUN pip3 install --user \
  transforms3d \
  ipympl \
  ipyevents \
  plotly \
  neptune-client neptune-tensorflow-keras neptune-notebooks

RUN jupyter nbextension install --user --py ipympl \
&& jupyter nbextension enable --user --py ipympl \
&& jupyter nbextension enable --user --py ipyevents \
&& jupyter nbextension enable --user --py neptune-notebooks


## KEEP THESE LINES LAST:
# Use these lines for quick install of pip and apt packages during experimentation:
RUN pip3 install --user \
  opencv-python \
  pydot
RUN sudo apt-get update -y && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ffmpeg libsm6 libxext6 \
  python3-rospy python3-geometry-msgs \
  python3-tf python3-tf2-geometry-msgs python3-tf2-ros \
  graphviz \
&& sudo rm -rf /var/lib/apt/lists/* \
&& sudo apt-get clean

# setup entrypoint
RUN echo "echo '~/.bashrc has been executed'" >> ~/.bashrc
COPY ./entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
