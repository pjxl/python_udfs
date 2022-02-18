from google.colab import auth

def bq_auth():
  return auth.authenticate_user()
