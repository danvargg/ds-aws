"""Train a text classifier using Amazon SageMaker BlazingText built-in algorithm."""
import boto3
import sagemaker
import pandas as pd
import numpy as np
import botocore
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split

config = botocore.config.Config(user_agent_extra='dlai-pds/c1/w4')

# low-level service client of the boto3 session
sm = boto3.client(service_name='sagemaker',
                  config=config)

sm_runtime = boto3.client('sagemaker-runtime',
                          config=config)

sess = sagemaker.Session(sagemaker_client=sm,
                         sagemaker_runtime_client=sm_runtime)

bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = sess.boto_region_name

# !aws s3 cp 's3://dlai-practical-data-science/data/balanced/womens_clothing_ecommerce_reviews_balanced.csv' ./
path = './womens_clothing_ecommerce_reviews_balanced.csv'

df = pd.read_csv(path, delimiter=',')
df.head()

nltk.download('punkt')

sentence = "I'm not a fan of this product!"

tokens = nltk.word_tokenize(sentence)
print(tokens)


def tokenize(review):
    # delete commas and quotation marks, apply tokenization and join back into a string separating by spaces
    return ' '.join([str(token) for token in nltk.word_tokenize(str(review).replace(',', '').replace('"', '').lower())])


def prepare_data(df):
    df['sentiment'] = df['sentiment'].map(
        lambda sentiment: '__label__{}'.format(str(sentiment).replace('__label__', '')))
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    df['review_body'] = df['review_body'].map(lambda review: tokenize(review))  # Replace all None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    return df


# create a sample dataframe
df_example = pd.DataFrame({
    'sentiment': [-1, 0, 1],
    'review_body': [
        "I don't like this product!",
        "this product is ok",
        "I do like this product!"]
})

# test the prepare_data function
print(prepare_data(df_example))

df_blazingtext = df[['sentiment', 'review_body']].reset_index(drop=True)
df_blazingtext = prepare_data(df_blazingtext)
df_blazingtext.head()

# Split the dataset into train and validation sets


# Split all data into 90% train and 10% holdout
df_train, df_validation = train_test_split(df_blazingtext,
                                           test_size=0.10,
                                           stratify=df_blazingtext['sentiment'])

labels = ['train', 'validation']
sizes = [len(df_train.index), len(df_validation.index)]
explode = (0.1, 0)

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle.
ax1.axis('equal')

# plt.show()
print(len(df_train))

blazingtext_train_path = './train.csv'
df_train[['sentiment', 'review_body']].to_csv(blazingtext_train_path, index=False, header=False, sep=' ')

blazingtext_validation_path = './validation.csv'
df_validation[['sentiment', 'review_body']].to_csv(blazingtext_validation_path, index=False, header=False, sep=' ')

train_s3_uri = sess.upload_data(bucket=bucket, key_prefix='blazingtext/data', path=blazingtext_train_path)
validation_s3_uri = sess.upload_data(bucket=bucket, key_prefix='blazingtext/data', path=blazingtext_validation_path)

# Train the model
image_uri = sagemaker.image_uris.retrieve(
    region=region,
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    framework='blazingtext'  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
)

estimator = sagemaker.estimator.Estimator(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    image_uri=image_uri,  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    volume_size=30,
    max_run=7200,
    sagemaker_session=sess
)

estimator.set_hyperparameters(mode='supervised',  # supervised (text classification)
                              epochs=10,  # number of complete passes through the dataset: 5 - 15
                              learning_rate=0.01,  # step size for the  numerical optimizer: 0.005 - 0.01
                              min_count=2,  # discard words that appear less than this number: 0 - 100
                              vector_dim=300,  # number of dimensions in vector space: 32-300
                              word_ngrams=3)  # number of words in a word n-gram: 1 - 3

train_data = sagemaker.inputs.TrainingInput(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    train_s3_uri,  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    distribution='FullyReplicated',
    content_type='text/plain',
    s3_data_type='S3Prefix'
)

validation_data = sagemaker.inputs.TrainingInput(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    validation_s3_uri,  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    distribution='FullyReplicated',
    content_type='text/plain',
    s3_data_type='S3Prefix'
)

data_channels = {
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    'train': train_data,  # Replace None
    'validation': validation_data  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
}

estimator.fit(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    inputs=data_channels,  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    wait=False
)

training_job_name = estimator.latest_training_job.name
print('Training Job Name:  {}'.format(training_job_name))

estimator.latest_training_job.wait(logs=False)
estimator.training_job_analytics.dataframe()

# Deploy the model
text_classifier = estimator.deploy(initial_instance_count=1,
                                   instance_type='ml.m5.large',
                                   serializer=sagemaker.serializers.JSONSerializer(),
                                   deserializer=sagemaker.deserializers.JSONDeserializer())

print()
print('Endpoint name:  {}'.format(text_classifier.endpoint_name))

# Test the model
reviews = ['This product is great!',
           'OK, but not great',
           'This is not the right product.']

tokenized_reviews = [' '.join(nltk.word_tokenize(review)) for review in reviews]

payload = {"instances": tokenized_reviews}
print(payload)

predictions = text_classifier.predict(data=payload)
for prediction in predictions:
    print('Predicted class: {}'.format(prediction['label'][0].lstrip('__label__')))
