FROM pytorch/pytorch

LABEL creator="Yaron Winter, yaron.winter@gmail.com"
LABEL task="Anti Spoofing"

RUN apt-get update --fix-missing && apt-get install -y wget python3-pip
RUN mkdir anti_spoofing
RUN mkdir anti_spoofing/data

ENV HOME=/anti_spoofing
ENV SHELL=/bin/bash
VOLUME /anti_spoofing/data
WORKDIR /anti_spoofing
COPY . anti_spoofing
RUN pip --no-cache-dir install --upgrade \
        scikit-learn==1.2.2 \
        torch==2.0.0 \ 
        torchaudio==2.0.1 \
        numpy==1.24.2 \
        soundfile==0.12.1 \
        tqdm==4.65.0


CMD ["/bin/bash"]
