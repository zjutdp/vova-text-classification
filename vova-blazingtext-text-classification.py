#!/usr/bin/env python
# coding: utf-8

# In[2]:


bucket = 'zjutdp-mb3-data' 
s3_train_data = 's3://{}/{}'.format(bucket, 'labeled/sgmkbuiltin/sgmk_train.csv')
s3_validation_data = 's3://{}/{}'.format(bucket, 'labeled/sgmkbuiltin/sgmk_validation.csv')
s3_output_location = 's3://{}/{}/output'.format(bucket, 'labeled/sgmkbuiltin/sgmk-output/')


# In[3]:


region_name = boto3.Session().region_name

container = sagemaker.amazon.amazon_estimator.get_image_uri(region_name, "blazingtext", "latest")
print('Using SageMaker BlazingText container: {} ({})'.format(container, region_name))

bt_model = sagemaker.estimator.Estimator(container,
                                         role, 
                                         train_instance_count=1, 
                                         train_instance_type='ml.c4.xlarge',
                                         train_volume_size = 30,
                                         train_max_run = 360000,
                                         input_mode= 'File',
                                         output_path=s3_output_location,
                                         sagemaker_session=sess)


bt_model.set_hyperparameters(mode="supervised",
                            epochs=20,
                            min_count=20,
                            learning_rate=0.1,
                            vector_dim=150,
                            early_stopping=True,
                            patience=4,
                            min_epochs=5,
                            word_ngrams=2)


# In[4]:


train_data = sagemaker.session.s3_input(s3_train_data, distribution='FullyReplicated', 
                        content_type='text/plain', s3_data_type='S3Prefix')
validation_data = sagemaker.session.s3_input(s3_validation_data, distribution='FullyReplicated', 
                             content_type='text/plain', s3_data_type='S3Prefix')
data_channels = {'train': train_data, 'validation': validation_data}


# In[5]:


bt_model.fit(inputs=data_channels, logs=True)


# In[6]:


text_classifier = bt_model.deploy(initial_instance_count = 1,instance_type = 'ml.m4.xlarge')


# In[16]:


df = pandas.read_csv('s3://{}/{}'.format('zjutdp-mb3-data', 'labeled/sgmkbuiltin/sgmk_test.csv'), header = None)
df.columns
df.head
#df[0][:5].values.tolist()


# In[17]:


import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')

print("Origin: {}".format(len(stop_words)))
ps=pandas.read_csv('product_stopwords.txt', header=None)
stop_words = stop_words + ps[0].values.tolist()
print("New: {}".format(len(stop_words)))

def remove_stop_word(text):
    words = [item.strip() for item in text.split(' ') if item not in stop_words and len(item.strip())>0]
    return ' '.join(words).strip()


# In[25]:


stop_word_lambda = lambda x: remove_stop_word(x)
df = pandas.DataFrame(df[0].apply(stop_word_lambda))

sentences = df[0][:200].values.tolist()
payload = {"instances" : sentences}
response = text_classifier.predict(json.dumps(payload))
predictions = json.loads(response)
#print(json.dumps(predictions, indent=2))

total = 0
for i, s in enumerate(sentences):
    #if predictions[i]["label"][0] != "__label__SEXUAL_WELLNESS":
    if predictions[i]["label"][0] == "__label__SEXUAL_WELLNESS" and predictions[i]["prob"][0] < 0.80:
        print("{}. {}".format(i, s))
        print(predictions[i])
        total = total + 1
        
print(total)


# In[29]:


sentences = df[0][200:].values.tolist()
payload = {"instances" : sentences}
response = text_classifier.predict(json.dumps(payload))
predictions = json.loads(response)
#print(json.dumps(predictions, indent=2))

total = 0
for i, s in enumerate(sentences):
    if predictions[i]["label"][0] == "__label__NON_SEXUAL_WELLNESS" and predictions[i]["prob"][0] < 0.60:
        print("{}. {}".format(i, s))
        print(predictions[i])
        total = total + 1
print(total)


# In[31]:


sentences = df[0].values.tolist()
payload = {"instances" : sentences}
response = text_classifier.predict(json.dumps(payload))

client = boto3.client('s3')
client.put_object(Body=response, Bucket='zjutdp-mb3-data', Key='sgmk-output/batch.json')





