FROM python:3.10
WORKDIR /app

COPY . /app

VOLUME /app/data
RUN pip3 install -r requirements.txt

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN python3 -c "import transformers; from transformers import AutoImageProcessor; from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-base'); AutoImageProcessor.from_pretrained('facebook/dinov2-base')"

RUN chmod +x /app/src/train.py
RUN chmod +x /app/src/make_submission.py

# CMD ["python3","/app/train.py", "/app/dinov2.yaml"]
# CMD ["python3","/app/make_submission.py", "/app/dinov2.yaml"]
