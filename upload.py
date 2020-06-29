"""Running this uploads all Python modules in this folder to an S3 bucket, to
access when training on Colab or another cloud service.
"""

import os
import code

import boto3
from google.cloud import storage

from cfg import get_config; CFG = get_config()

S3_BUCKET_NAME = CFG['s3_bucket']
GCS_BUCKET_NAME = CFG['gcs_bucket']
code_path = os.path.dirname(os.path.abspath(__file__))
# S3
s3_bucket = boto3.resource('s3').Bucket(S3_BUCKET_NAME)
s3_client = boto3.client('s3')
# GCS
if CFG['use_gcs']:
  gcs_client = storage.Client()
  gcs_bucket = gcs_client.bucket(GCS_BUCKET_NAME)


def s3_upload_data(s3_path, body):
  """Upload data to S3 bucket at a particular path
  """
  s3_bucket.put_object(Key=s3_path, Body=body)


def s3_upload_file(local_path, s3_path):
  """Upload a locally-saved file to S3
  """
  s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_path)


def gcs_upload_blob(source_file_name, destination_blob_name):
    """Upload a file to Google Storage bucket
    """
    blob = gcs_bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(
        "File {} uploaded to {}".format(
            source_file_name, destination_blob_name
        )
    )


def gcs_upload_blob_from_memory(source_file, destination_blob_name):
  blob = gcs_bucket.blob(destination_blob_name)
  blob.upload_from_file(source_file)
  print(f"File uploaded to {destination_blob_name}")


def gcs_upload_blob_from_string(source_string, destination_blob_name, print_str=False):
  blob = gcs_bucket.blob(destination_blob_name)
  blob.upload_from_string(source_string, content_type="application/json")
  if print_str:
    print(source_string)
  print(f"File uploaded to {destination_blob_name}")


def gcs_download_blob_as_string(blob_name):
  blob = gcs_bucket.blob(blob_name)
  blob_str = blob.download_as_string()
  return blob_str


def gcs_folder_exists(file_name):
  blobs = list(gcs_client.list_blobs(GCS_BUCKET_NAME, prefix=file_name))
  if len(blobs) > 0:
    return True
  else:
    return False


if __name__ == "__main__":
  code_only = True
  if code_only:
    exts = ['.py', '.yaml']
  else:
    exts = ['.py', '.json', '.yaml', '.h5']

  # base path
  for file in os.listdir(code_path):
    if any([file.endswith(ext) for ext in exts]):
      print(f"Uploading {file}")
      s3_upload_file(os.path.join(code_path, file), f"cg/{file}")

  # relative paths
  for sub in ['bit', 'models']:
    sub_path = os.path.join(code_path, sub)
    for file in os.listdir(sub_path):
      if any([file.endswith(ext) for ext in exts]):
        print(f"Uploading {sub}/{file}")
        s3_upload_file(os.path.join(code_path, f"{sub}/{file}"), f"cg/{sub}/{file}")
