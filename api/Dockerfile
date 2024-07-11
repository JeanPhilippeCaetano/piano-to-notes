FROM python:3.10

WORKDIR /api

RUN apt-get update -qq && apt-get install -qq libfluidsynth3 build-essential libasound2-dev libjack-dev musescore \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-sdk \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch=main https://github.com/magenta/mt3
RUN mv mt3 mt3_tmp; mv mt3_tmp/* .; rm -r mt3_tmp

COPY requirements.txt /api/

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN gsutil -q -m cp -r gs://mt3/checkpoints .

RUN gsutil -q -m cp gs://magentadata/soundfonts/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2 .

RUN pip install FastAPI uvicorn mido

ENV QT_QPA_PLATFORM = offscreen

COPY . /api

EXPOSE 8000

CMD [ "python", "api.py" ]