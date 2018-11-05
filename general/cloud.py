# Cloud utilities


### Google Cloud Storage upload
from google.cloud import storage

def upload_to_google_cloud(source_file_name, bucket_name="jacobshack"):
   """Uploads a file to the bucket."""
   storage_client = storage.Client()
   bucket = storage_client.get_bucket(bucket_name)
   blob = bucket.blob(source_file_name)
   blob.upload_from_filename(source_file_name)
   return "https://storage.googleapis.com/" + bucket_name + "/" + source_file_name


### Algolia sample
from algoliasearch import algoliasearch

client = algoliasearch.Client(os.environ['ALGOLIA_APPLICATION'], os.environ['ALGOLIA_ADMIN'])
index = client.init_index('whodat')
index.set_settings({
  'searchableAttributes': [
    'caption'
  ]
})

index.addObject({'url':'blah', 'caption': 'blah caption'})
