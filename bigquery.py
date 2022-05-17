try:
    from google.cloud import bigquery
    from google.colab import auth
except:
    pass



def bq_auth():
    return auth.authenticate_user()


class BigQueryClient():
    def __init__(self, project='etsy-bigquery-adhoc-prod', params=dict():
        self.params = params
        try:
            self.client = bigquery.Client(project=project)
        
        try:
            bq_auth()        
          
    def query(self, sql):
        return self.client.query(sql).to_dataframe()
