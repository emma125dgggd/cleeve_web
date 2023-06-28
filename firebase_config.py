import firebase_admin
from firebase_admin import credentials, auth, storage, db



cred = credentials.Certificate('firebase-adminsdk.json')

try:

    firebase_admin.get_app()

except ValueError:
    firebase_admin.initialize_app(cred, {
    'databaseURL': "https://cleeve-database-default-rtdb.europe-west1.firebasedatabase.app/",
    'storageBucket': "cleeve-database.appspot.com" 
})
