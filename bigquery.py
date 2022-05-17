try:
    from google.cloud import bigquery
    from google.colab import auth
except:
    pass



def bq_auth():
    return auth.authenticate_user()


class BigQueryClient():
    def __init__(self, project='etsy-bigquery-adhoc-prod', params=dict()):
        self.params = params
        
        try:
            self.client = bigquery.Client(project=project)
        except ModuleNotFoundError as error:
            print('Warning:', error)
        
        try:
            bq_auth()
        except ModuleNotFoundError as error:
            print('Warning:', error)
        
        
    def query(self, sql):
        return self.client.query(sql).to_dataframe()
    
    
    def set_params(self, params: dict, verbose=False):
        self.params.update(params)
        
        if verbose:
            print(self.params)
        else:
            pass
