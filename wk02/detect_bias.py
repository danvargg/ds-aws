"""Detect data bias with Amazon SageMaker Clarify"""
import boto3
import sagemaker
import pandas as pd
import numpy as np
import botocore
import matplotlib.pyplot as plt
import seaborn as sns
from sagemaker import clarify

config = botocore.config.Config(user_agent_extra='dlai-pds/c1/w2')

# low-level service client of the boto3 session
sm = boto3.client(service_name='sagemaker', config=config)

sess = sagemaker.Session(sagemaker_client=sm)

bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = sess.boto_region_name

path = './womens_clothing_ecommerce_reviews_transformed.csv'

df = pd.read_csv(path)
df.head()

sns.countplot(data=df, x='sentiment', hue='product_category')

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Configure a DataConfig for Clarify
bias_report_unbalanced_output_path = 's3://{}/bias/generated_bias_report/unbalanced'.format(bucket)

data_config_unbalanced = clarify.DataConfig(
    s3_data_input_path=data_s3_uri_unbalanced,  # Replace None
    s3_output_path=bias_report_unbalanced_output_path,  # Replace None
    label='sentiment',  # Replace None
    headers=df.columns.to_list(),
    dataset_type='text/csv'
)

# Configure BiasConfig
bias_config_unbalanced = clarify.BiasConfig(
    label_values_or_threshold=[1],  # desired sentiment
    facet_name='product_category'  # sensitive column (facet)
)
# Configure Amazon SageMaker Clarify as a processing job
clarify_processor_unbalanced = clarify.SageMakerClarifyProcessor(
    role=role, instance_count=1, instance_type='ml.m5.large', sagemaker_session=sess
)

# Run the Amazon SageMaker Clarify processing job
clarify_processor_unbalanced.run_pre_training_bias(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    data_config=data_config_unbalanced,  # Replace None
    data_bias_config=bias_config_unbalanced,  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    methods=["CI", "DPL", "KL", "JS", "LP", "TVD", "KS"],
    wait=False,
    logs=False
)

run_unbalanced_bias_processing_job_name = clarify_processor_unbalanced.latest_job.job_name
print(run_unbalanced_bias_processing_job_name)

# Run and review the Amazon SageMaker Clarify processing job on the unbalanced dataset
running_processor = sagemaker.processing.ProcessingJob.from_processing_name(
    processing_job_name=run_unbalanced_bias_processing_job_name, sagemaker_session=sess
)

#  5 - 10 mins
running_processor.wait(logs=False)

# Analyze unbalanced bias report
# !aws s3 ls $bias_report_unbalanced_output_path/
# Balance the dataset by product_category and sentiment
df_grouped_by = df.groupby(['product_category', 'sentiment'])
df_balanced = df_grouped_by.apply(lambda x: x.sample(df_grouped_by.size().min()).reset_index(drop=True))
print(df_balanced.head())

# Visualize the distribution of review sentiment in the balanced dataset
sns.countplot(data=df_balanced, x='sentiment', hue='product_category')

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
