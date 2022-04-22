import sys

try:
    from google.cloud import bigquery
    from google.colab import auth
except:
    pass
# except ModuleNotFoundError as moduleErr:
#     print("[Error]: Failed to import (Module Not Found) {}.".format(moduleErr.args[0]))
#     sys.exit(1)
# except ImportError as impErr:
#     print("[Error]: Failed to import (Import Error) {}.".format(impErr.args[0]))
#     sys.exit(1)



def bq_auth():
    return auth.authenticate_user()


class BQClient():
    def __init__(self, project='etsy-bigquery-adhoc-prod'):
        self.client = bigquery.Client(project=project)
        
        
    def query(self, sql):
        return self.client.query(sql).to_dataframe()
