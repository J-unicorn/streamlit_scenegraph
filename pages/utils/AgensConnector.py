#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
#-*- coding: utf-8 -*-
#####################################################
# Program        : AgensConnector.py
# Main function  : Connect AgensGraph and Pandas Dataframe
# Creator        : Doohee Jung (Bitnine Graph Science)
# Created date   : 2022.03.13
# Comment        :
#####################################################

import pandas as pd
import psycopg2 as pg2
from datetime import datetime as dt
from sqlalchemy import create_engine

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]    


class AgensConnector(metaclass=SingletonMeta):
 
    def __init__(self, host, port, database, user, password, graph_path = None, autocommit=True):
        self.db_params = self.agensConfig(host, port, database, user, password)
        self.graph_path = graph_path
        self.conn = pg2.connect(**self.db_params)
        self.cur = self.conn.cursor()
        self.conn.autocommit = autocommit
        self.set_graph(graph_path)
        print('Successful connection to agensgraph!')
        print('connection_info')
        print(self.db_params)

    def __del__(self):

        self.close_agens()
        print('close the agensgraph connection!')

    def conn_info(self):
        db_config = self.db_params
        return db_config    
        
    def agensConfig(self, host, port, database, user, password):
        db_config = {"host":host, "port":port, "database":database, "user":user, "password":password}
        return db_config

    def set_graph(self,graph):


        graph_path = self.graph_path
        self.graph = graph       

        if self.graph:
            graph_path = self.graph

        if not graph_path:
            return

        return self.cur.execute("SET graph_path="+graph_path) 


    def close_agens(self):
        if not self.conn:
            return

        if not self.conn.closed:
            self.conn.close()
        
    def commit_agens(self):
        if not self.conn:
            return
        self.conn.commit()          
    
    # transform pandas from agensquery

    def query_one(self,query):
        try:
            self.query = query
            conn = self.conn
            cur = self.cur
            cur.execute(self.query)
            result = cur.fetchone

        except (Exception, pg2.DatabaseError) as error:
            print('[%s][ERR|query_pandas] %s' % (str(dt.now()), error))
            print(query)
            conn.rollback()    


    def query_pandas(self,query,table =None,graph=None, header=True):
        try:
            self.query = query
            conn = self.conn
            cur = self.cur

            if table:
                self.query = f'SELECT * FROM {query}' 

            if graph:
                set_graph = self.set_graph(graph)
            
            cur.execute(self.query)
            result = cur.fetchall()
            
            if not result:
                return pd.DataFrame()
            
            result = pd.DataFrame(result)
            
            
            if header:

                if 'limit' in self.query:
                    self.query = query[0:query.find('limit')]

                self.query_column = self.query + ' LIMIT 0'
                cur.execute(self.query_column)
                result.columns = [desc[0] for desc in cur.description]
            
            return result

        except (Exception, pg2.DatabaseError) as error:
            print('[%s][ERR|query_pandas] %s' % (str(dt.now()), error))
            print(query)
            conn.rollback()

    # execute query into agensgraph (add iterable)
    def query_exe(self,query,graph=None,iterable = None,autocommit=True):
        try:    
            self.query = query
            self.iterable=iterable
            conn = self.conn
            cur = self.cur
            conn.autocommit = autocommit

            if graph:
                set_graph = self.set_graph(graph)
            
            if iterable: 
                cur.executemany(self.query,self.iterable)
            else:
                cur.execute(self.query)
            
            return

        except (Exception, pg2.DatabaseError) as error:
            print('[%s][ERR|query_exe] %s' % (str(dt.now()), error))
            print(query)
            conn.rollback()

    def pandas_exe(self, df, table,schema ='public', exists = 'append'):

        db_params=self.db_params
        self.df = df
        self.table = table
        self.schema = 'public'
        self.exists = exists

        engine_info = db_params['user']+':'+ db_params['password']+'@'+ db_params['host']+':'+ db_params['port']+'/'+ db_params['database']
        engine = create_engine('postgresql://' + engine_info)

        df.to_sql(  
                    name = self.table,
                    con = engine,
                    schema = self.schema,
                    if_exists = self.exists,
                    index = False
          ) 

        return

    def commit(self):
        if ( self.conn is None  ):
            return
        self.conn.commit()

    def rollback(self):
        if ( self.conn is None  ):
            return
        self.conn.rollback()

    def load_vertex(self,df,graph,vlabel,key):
        
        init_graph = f'CREATE GRAPH IF NOT EXISTS {graph} ; SET GRAPH_PATH = {graph}; '  
        init_vlabel = f'CREATE VLABEL IF NOT EXISTS {vlabel}'
        self.query_exe(init_graph+init_vlabel)
        table_temp = 'temp_load_table'
        self.pandas_exe(df,table_temp,exists='replace')
        filter_temp = f'''
        
        
                        create temp table vt1 as 
                        select *,row_to_json(t) as row_json from (
                        select distinct * from {table_temp}
                        where not exists (
                        match(a:{vlabel}) where a->>'{key}' = {key} return a)
                        )t;
                    '''
        loader_vertex = f'load from vt1 as tb create(a:{vlabel}) set properties(a) = tb.row_json; '
     
        drop_table = f'drop table temp_load_table; ' +   'drop table vt1;'
       
        self.query_exe(filter_temp) 
        self.query_exe(loader_vertex) 
        self.query_exe(drop_table)
        
        return 

    def load_graph(self,dfv1,dfv2,dfe1,graph,vlabel1,vlabel2,elabel,key1,key2):

        # load_vertex + load edge

        #####################################    

        # preference setting setting example
        
        # please do divide your dataframe into vertex and edge dataframe 

        # param_dict = {}
        # param_dict['dfv1']=df1                vertex1 data frame
        # param_dict['dfv2']=dfv2               vertex2 data frame
        # param_dict['dfe1']=df3                edge data frame
        # param_dict['graph']='vio_test'        graph_path
        # param_dict['vlabel1']='vt_vio'        vertex label1
        # param_dict['vlabel2']='vt_vio'        vertex label2
        # param_dict['elabel']='edg_viocng'     edge label1
        # param_dict['key1']='vio_nm'           vertex1 unique key
        # param_dict['key2']='vio_nm'           vertex2 unique key    

        # execute  : load_graph(**param_dict) 


    
        # set graph
        chk_graph = f'CREATE GRAPH IF NOT EXISTS {graph}'
        set_graph = f'SET GRAPH_PATH = {graph};'  
        self.query_exe(chk_graph)
        self.query_exe(set_graph)
        
        # vertex/edge keys check
        v1_key,v2_key,e1_key = dfv1.columns.tolist(),dfv2.columns.tolist(),dfe1.columns.tolist()
        
        vlabel_lst=self.query_pandas(f'select array_agg(distinct label(a)) as label from (match(a) return a) t').fillna('').loc[:,'label'].tolist()[0]
        elabel_lst=self.query_pandas(f'select array_agg(distinct label(r)) as label from (match()-[r]-() return r) t',graph='vio_graph').fillna('').loc[:,'label'].tolist()[0]
        
        if vlabel1 in vlabel_lst:
            v1_key = self.query_pandas(f'match(a:{vlabel1}) return keys(a) limit 1').loc[:,'keys'].tolist()[0]
        
        if vlabel2 in vlabel_lst:
            v2_key = self.query_pandas(f'match(a:{vlabel2}) return keys(a) limit 1').loc[:,'keys'].tolist()[0]
        
        if elabel in elabel_lst:
            e1_key = self.query_pandas(f'match()-[{elabel}]-() return keys(a) limit 1').loc[:,'keys'].tolist()[0]
        
        if not (v1_key == dfv1.columns.tolist() and v2_key == dfv2.columns.tolist() and e1_key == dfe1.columns.tolist()) :
            print('should your vertex/edge property equal to dataframe key!')
            return 
        

        # vertex load
        
        self.load_vertex(dfv1,graph,vlabel1,key1)
        self.load_vertex(dfv2,graph,vlabel2,key2)
        
        # set edge dataframe
        
        edge_df=pd.concat([dfv1[key1],dfv2[key2],dfe1],axis=1)
        e1_param =','.join(e1_key)
        edge_df.columns =  ['v1','v2'] + e1_key 
        
        # chk_elabel 
        chk_label = f'CREATE ELABEL IF NOT EXISTS {elabel};'
        self.query_exe(chk_label)
        
        
        table_temp = 'temp_load_table'
        self.pandas_exe(edge_df,table_temp,exists='replace')   
        
        filter_temp = f'''       
                        CREATE TEMP TABLE tb1  AS
                        SELECT v1,v2,
                            (SELECT row_to_json(_)
                                FROM (
                                    SELECT 
                                        {e1_param}
                                    ) 
                            as _) as row_json 
                        FROM {table_temp};
                
                    '''
        # load edge
        
        load_edge = f'''
                        load from tb1 as tb 
                        match(a:{vlabel1}),(b:{vlabel2}) 
                        where a.{key1} = tb.v1 and b.{key2} = tb.v2    
                        create(a)-[r:{elabel}]->(b) 
                        set properties(r) = tb.row_json; 
                    '''
        drop_table = f'drop table {table_temp}; ' +  'drop table tb1;'
        self.query_exe(filter_temp)
        self.query_exe(load_edge)
        self.query_exe(drop_table)
        
        return 