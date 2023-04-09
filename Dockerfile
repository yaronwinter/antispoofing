FROM pytorch/pytorch

LABEL creator="Yaron Winter, yaron.winter@gmail.com"
LABEL task="Anti Spoofing"

RUN apt-get update --fix-missing && apt-get install -y wget python3-pip
RUN mkdir app
RUN mkdir app/data
RUN mkdir app/data/logs
RUN mkdir app/data/models
RUN mkdir app/data/models/cnn
RUN mkdir app/data/models/aasist

ENV HOME=/app
ENV SHELL=/bin/bash
VOLUME /app/data
WORKDIR /app
COPY . /app
RUN pip --no-cache-dir install --upgrade \
        scikit-learn==1.2.2 \
        torchaudio==2.0.1 \
        numpy==1.24.2 \
        soundfile==0.12.1 \
        tqdm==4.65.0


CMD ["/bin/bash"]
