import firebase_admin
from firebase_admin import credentials, auth, storage, db



cred = credentials.Certificate('cleeve-api-firebase-adminsdk.json')

try:

    firebase_admin.get_app()

except ValueError:
    firebase_admin.initialize_app(cred, {
    'storageBucket': "cleeve-api.appspot.com" 
})
