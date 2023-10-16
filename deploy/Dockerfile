FROM python:3.9-slim
COPY ./MovieReviewSentiment_app.py /deploy_net/
COPY ./requirements.txt /deploy_net/
COPY ./sentiment_inference_model_hashed.pkl /deploy_net/
COPY ./vocabulary_hashed.pkl /deploy_net/
WORKDIR /deploy_net/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "MovieReviewSentiment_app.py"]
