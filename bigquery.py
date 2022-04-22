try:
    from google.cloud import bigquery
    from google.colab import auth
except:
    pass



def bq_auth():
    return auth.authenticate_user()


class BQClient():
    def __init__(self, project='etsy-bigquery-adhoc-prod'):
        self.client = bigquery.Client(project=project)
        
        
    def query(self, sql):
        return self.client.query(sql).to_dataframe()
