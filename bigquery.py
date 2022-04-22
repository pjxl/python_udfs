from google.cloud import bigquery
from google.colab import auth


def bq_auth():
    return auth.authenticate_user()


class BQClient(project='etsy-bigquery-adhoc-prod'):
    def __init__(self, project):
        self.client = bigquery.Client(project=project)
        
        
    def query(self, sql):
        return self.client.query(sql).to_dataframe()
