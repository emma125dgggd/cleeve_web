import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

firebase_storage = storage.bucket()

blob = firebase_storage.blob('best (3).pt')
blob.download_to_filename('best.pt')
