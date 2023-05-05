"""Ingest and transform the customer product reviews dataset."""
import csv
import pandas as pd
import boto3
import sagemaker
import numpy as np
import botocore
import awswrangler as wr
import numpy as np
import seaborn as sns

# Ingest and transform the public dataset¶
# The dataset [Women's Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
# has been chosen as the main dataset.
# It is shared in a public Amazon S3 bucket, and is available as a comma-separated value (CSV) text format:
# s3://dlai-practical-data-science/data/raw/womens_clothing_ecommerce_reviews.csv

# View the list of the files available in the public bucket s3://dlai-practical-data-science/data/raw/.

# !aws s3 ls s3://dlai-practical-data-science/data/raw/

# Copy the data locally to the notebook
# !aws s3 cp s3://dlai-practical-data-science/data/raw/womens_clothing_ecommerce_reviews.csv ./womens_clothing_ecommerce_reviews.csv

# Load and preview the data


df = pd.read_csv('./womens_clothing_ecommerce_reviews.csv',
                 index_col=0)

print(df.shape)

# Transform the data
df_transformed = df.rename(columns={'Review Text': 'review_body',
                                    'Rating': 'star_rating',
                                    'Class Name': 'product_category'})
df_transformed.drop(
    columns=['Clothing ID', 'Age', 'Title', 'Recommended IND', 'Positive Feedback Count', 'Division Name',
             'Department Name'],
    inplace=True)

df_transformed.dropna(inplace=True)

print(df_transformed.shape)


# Now convert the star_rating into the sentiment (positive, neutral, negative), which later on will be for the prediction
def to_sentiment(star_rating):
    if star_rating in {1, 2}:  # negative
        return -1
    if star_rating == 3:  # neutral
        return 0
    if star_rating in {4, 5}:  # positive
        return 1


# transform star_rating into the sentiment
df_transformed['sentiment'] = df_transformed['star_rating'].apply(lambda star_rating:
                                                                  to_sentiment(star_rating=star_rating)
                                                                  )

# drop the star rating column
df_transformed.drop(columns=['star_rating'],
                    inplace=True)

# remove reviews for product_categories with < 10 reviews
df_transformed = df_transformed.groupby('product_category').filter(lambda reviews: len(reviews) > 10)[
    ['sentiment', 'review_body', 'product_category']]

print(df_transformed.shape)
print(df_transformed.head())

# Write the data to a CSV file
df_transformed.to_csv('./womens_clothing_ecommerce_reviews_transformed.csv',
                      index=False)
# Register S3 dataset files as a table for querying
config = botocore.config.Config(user_agent_extra='dlai-pds/c1/w1')

# low-level service client of the boto3 session
sm = boto3.client(service_name='sagemaker',
                  config=config)

sess = sagemaker.Session(sagemaker_client=sm)

bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = sess.boto_region_name
account_id = sess.account_id

print('S3 Bucket: {}'.format(bucket))
print('Region: {}'.format(region))
print('Account ID: {}'.format(account_id))

# Copy the file into the S3 bucket
# !aws s3 cp ./womens_clothing_ecommerce_reviews_transformed.csv s3://$bucket/data/transformed/womens_clothing_ecommerce_reviews_transformed.csv

# Create AWS Glue Catalog database
wr.catalog.create_database(
    name='dsoaws_deep_learning',
    exist_ok=True
)
dbs = wr.catalog.get_databases()

for db in dbs:
    print("Database name: " + db['Name'])

# Register CSV data with AWS Glue Catalog
wr.catalog.create_csv_table(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    database='dsoaws_deep_learning',  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    path='s3://{}/data/transformed/'.format(bucket),
    table="reviews",
    columns_types={
        'sentiment': 'int',
        'review_body': 'string',
        'product_category': 'string'
    },
    mode='overwrite',
    skip_header_line_count=1,
    sep=','
)

# Review the table shape
table = wr.catalog.table(database='dsoaws_deep_learning',
                         table='reviews')
print(table)

# Create default S3 bucket for Amazon Athena
wr.athena.create_athena_bucket()
# EXPECTED OUTPUT
# 's3://aws-athena-query-results-ACCOUNT-REGION/'

# Visualize the data


import matplotlib.pyplot as plt

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

# Set AWS Glue database and table name
database_name = 'dsoaws_deep_learning'
table_name = 'reviews'

# Seaborn params
sns.set_style = 'seaborn-whitegrid'

sns.set(rc={"font.style": "normal",
            "axes.facecolor": "white",
            'grid.color': '.8',
            'grid.linestyle': '-',
            "figure.facecolor": "white",
            "figure.titlesize": 20,
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.labelcolor": "black",
            "axes.grid": True,
            'axes.labelsize': 10,
            'xtick.labelsize': 10,
            'font.size': 10,
            'ytick.labelsize': 10})

# Run SQL queries using Amazon Athena
# How many reviews per sentiment?
statement_count_by_sentiment = """
SELECT sentiment, COUNT(sentiment) AS count_sentiment
FROM reviews
GROUP BY sentiment
ORDER BY sentiment
"""
print(statement_count_by_sentiment)
# Query data in Amazon Athena database cluster using the prepared SQL statement:
df_count_by_sentiment = wr.athena.read_sql_query(
    sql=statement_count_by_sentiment,
    database=database_name
)

print(df_count_by_sentiment)

df_count_by_sentiment.plot(kind='bar', x='sentiment', y='count_sentiment', rot=0)

# Use Amazon Athena query with the standard SQL statement passed as a parameter, to calculate the total number of reviews per product_category in the table reviews
# as a triple quote string into the variable statement_count_by_category. Please use the column sentiment in the COUNT function and give it a new name count_sentiment
# Replace all None
### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
statement_count_by_category = """
SELECT product_category, COUNT(sentiment) AS count_sentiment
FROM reviews
GROUP BY product_category 
ORDER BY count_sentiment DESC
"""
### END SOLUTION - DO NOT delete this comment for grading purposes
print(statement_count_by_category)

# Which product categories are highest rated by average sentiment?
# Set the SQL statement to find the average sentiment per product category, showing the results in the descending order:
statement_avg_by_category = """
SELECT product_category, AVG(sentiment) AS avg_sentiment
FROM {} 
GROUP BY product_category 
ORDER BY avg_sentiment DESC
""".format(table_name)

print(statement_avg_by_category)

# %%time
df_avg_by_category = wr.athena.read_sql_query(
    sql=statement_avg_by_category,
    database=database_name
)


# Visualization
def show_values_barplot(axs, space):
    def _show_on_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() + float(space)
            _y = p.get_y() + p.get_height()
            value = round(float(p.get_width()), 2)
            ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_plot(ax)
    else:
        _show_on_plot(axs)


# Create plot
barplot = sns.barplot(
    data=df_avg_by_category,
    y='product_category',
    x='avg_sentiment',
    color="b",
    saturation=1
)

# Set the size of the figure
sns.set(rc={'figure.figsize': (15.0, 10.0)})

# Set title and x-axis ticks
plt.title('Average sentiment by product category')
# plt.xticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])

# Helper code to show actual values afters bars
show_values_barplot(barplot, 0.1)

plt.xlabel("Average sentiment")
plt.ylabel("Product category")

plt.tight_layout()
# Do not change the figure name - it is used for grading purposes!
plt.savefig('avg_sentiment_per_category.png', dpi=300)

# Show graphic
plt.show(barplot)

# Upload image to S3 bucket
sess.upload_data(path='avg_sentiment_per_category.png', bucket=bucket, key_prefix="images")

# Which product categories have the most reviews
statement_count_by_category_desc = """
SELECT product_category, COUNT(*) AS count_reviews 
FROM {}
GROUP BY product_category 
ORDER BY count_reviews DESC
""".format(table_name)

print(statement_count_by_category_desc)

# %%time
df_count_by_category_desc = wr.athena.read_sql_query(
    sql=statement_count_by_category_desc,
    database=database_name
)
max_sentiment = df_count_by_category_desc['count_reviews'].max()
print(max_sentiment)

# Use barplot function to plot number of reviews per product category
# Create seaborn barplot
barplot = sns.barplot(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    data=df_count_by_category_desc,  # Replace None
    y='count_reviews',  # Replace None
    x='product_category',  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    color="b",
    saturation=1
)

# Set the size of the figure
sns.set(rc={'figure.figsize': (15.0, 10.0)})

# Set title
plt.title("Number of reviews per product category")
plt.xlabel("Number of reviews")
plt.ylabel("Product category")

plt.tight_layout()

# Do not change the figure name - it is used for grading purposes!
plt.savefig('num_reviews_per_category.png', dpi=300)

# Show the barplot
plt.show(barplot)

# Upload image to S3 bucket
sess.upload_data(path='num_reviews_per_category.png', bucket=bucket, key_prefix="images")

# What is the breakdown of sentiments per product category?
statement_count_by_category_and_sentiment = """
SELECT product_category,
         sentiment,
         COUNT(*) AS count_reviews
FROM {}
GROUP BY  product_category, sentiment
ORDER BY  product_category ASC, sentiment DESC, count_reviews
""".format(table_name)

print(statement_count_by_category_and_sentiment)

# %%time
df_count_by_category_and_sentiment = wr.athena.read_sql_query(
    sql=statement_count_by_category_and_sentiment,
    database=database_name
)

# Prepare for stacked percentage horizontal bar plot showing proportion of sentiments per product category
# Create grouped dataframes by category and by sentiment
grouped_category = df_count_by_category_and_sentiment.groupby('product_category')
grouped_star = df_count_by_category_and_sentiment.groupby('sentiment')

# Create sum of sentiments per star sentiment
df_sum = df_count_by_category_and_sentiment.groupby(['sentiment']).sum()

# Calculate total number of sentiments
total = df_sum['count_reviews'].sum()
print('Total number of reviews: {}'.format(total))

# Create dictionary of product categories and array of star rating distribution per category
distribution = {}
count_reviews_per_star = []
i = 0

for category, sentiments in grouped_category:
    count_reviews_per_star = []
    for star in sentiments['sentiment']:
        count_reviews_per_star.append(sentiments.at[i, 'count_reviews'])
        i = i + 1;
    distribution[category] = count_reviews_per_star

# Build array per star across all categories
df_distribution_pct = pd.DataFrame(distribution).transpose().apply(
    lambda num_sentiments: num_sentiments / sum(num_sentiments) * 100, axis=1
)
df_distribution_pct.columns = ['1', '0', '-1']
print(df_distribution_pct)

# Plot the distributions of sentiments per product category
categories = df_distribution_pct.index

# Plot bars
plt.figure(figsize=(10,5))

df_distribution_pct.plot(kind="barh",
                         stacked=True,
                         edgecolor='white',
                         width=1.0,
                         color=['green',
                                'orange',
                                'blue'])

plt.title("Distribution of reviews per sentiment per category",
          fontsize='16')

plt.legend(bbox_to_anchor=(1.04,1),
           loc="upper left",
           labels=['Positive',
                   'Neutral',
                   'Negative'])

plt.xlabel("% Breakdown of sentiments", fontsize='14')
plt.gca().invert_yaxis()
plt.tight_layout()

# Do not change the figure name - it is used for grading purposes!
plt.savefig('distribution_sentiment_per_category.png', dpi=300)
plt.show()

# Upload image to S3 bucket
sess.upload_data(path='distribution_sentiment_per_category.png', bucket=bucket, key_prefix="images")

# Analyze the distribution of review word counts
statement_num_words = """
    SELECT CARDINALITY(SPLIT(review_body, ' ')) as num_words
    FROM {}
""".format(table_name)

print(statement_num_words)

# %%time
df_num_words = wr.athena.read_sql_query(
    sql=statement_num_words,
    database=database_name
)

# Print out and analyse some descriptive statistics:
summary = df_num_words["num_words"].describe(percentiles=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
print(summary)

# Plot the distribution of the words number per review
df_num_words["num_words"].plot.hist(xticks=[0, 16, 32, 64, 128, 256], bins=100, range=[0, 256]).axvline(
    x=summary["100%"], c="red"
)

plt.xlabel("Words number", fontsize='14')
plt.ylabel("Frequency", fontsize='14')
plt.savefig('distribution_num_words_per_review.png', dpi=300)
plt.show()

# Upload image to S3 bucket
sess.upload_data(path='distribution_num_words_per_review.png', bucket=bucket, key_prefix="images")

# !aws s3 cp ./C1_W1_Assignment.ipynb s3://$bucket/C1_W1_Assignment_Learner.ipynb
