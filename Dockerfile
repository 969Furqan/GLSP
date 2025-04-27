FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113


COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]