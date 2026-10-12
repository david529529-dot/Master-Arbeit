FROM nvcr.io/nvidia/pytorch:24.12-py3

ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME

RUN mkdir /data
# RUN cd llama && pip install -r requirements.txt && pip install -e .

# install tmux (optional)
RUN apt-get update && apt-get install -y tmux

# add /home/vscode/.local/bin to path (e.g. for huggingface-cli)
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/vscode/.bashrc