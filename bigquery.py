try:
    from google.cloud import bigquery
    from google.colab import auth
except:
    pass



def _parse_table_string(table):
    project_id, schema_id, table_id = table.split('.')
    return project_id, schema_id, table_id


def bq_auth():
    return auth.authenticate_user()


class BigQueryClient():
    def __init__(self, project='etsy-bigquery-adhoc-prod', params=dict()):
        self.project = project
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


    def audit_table(self, table, pk=None):
        project_id, schema_id, table_id = _parse_table_string(table)

        print('Column count:', self.query(
            f"""
            select format("%'d", sum(1)) from {'.'.join([project_id, schema_id])}.INFORMATION_SCHEMA.COLUMNS where table_name = '{table_id}'
            """
            ).squeeze())

        print('Row count:', self.query(
            f"""select format("%'d", sum(1)) from {table}"""
            ).squeeze())

        if pk:
            print('Rows with null PK:', str(self.query(
                f"""
                select coalesce(sum(1),0) from {table} where {pk} is null
                """
                ).squeeze()).upper())

            print('PK is unique:', str(self.query(
                f"""
                select sum(1) = count(distinct {pk}) from {table}
                """
                ).squeeze()).upper())


    def preview_table(self, table, nrows=5, order_by='rand()', ascending=True):
        direction = 'asc' if ascending else 'desc'
        return self.query(f'select * from {table} order by {order_by} {direction} limit {nrows}')

    
    def view_table_columns(self, table):
        project_id, schema_id, table_id = _parse_table_string(table)

        return self.query(
            f"""
            select column_name, is_nullable, data_type 
            from {'.'.join([project_id, schema_id])}.INFORMATION_SCHEMA.COLUMNS 
            where table_name = '{table_id}'
            order by ordinal_position
            """
            )


    def grant_table(self, table, users=None, groups=None, role='viewer'):  
        grantees = []
        if users:
            grantees.extend([f"'user:{i}'" for i in users])
        if groups:            
            grantees.extend([f"'group:{i}'" for i in groups])
        
        # GRANT TABLE appears to require backticks around the project ID, 
        # To avoid double-backticking, strip out any pre-existing backticks in the `table` string
        return self.query(f"""grant `roles/bigquery.data{role.capitalize()}` on table `{table.replace('`', '')}` to {','.join(grantees)}""")


    def set_params(self, params: dict, verbose=False):
        self.params.update(params)
        
        if verbose:
            print(self.params)
        else:
            pass
