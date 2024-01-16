import sqlite3
import pandas as pd

# for quick insertion of data into database tables
def insert_from_dict(con, cur, table, params):
    """
        insert into table from params

        Parameters
        ----------
        con : sqlite3 connection object
            connection object
        cur : sqlite3 cursor object
            cursor object
        table : str
            table name
        params : dict
            dictionary of parameters (keys and values) wheren keys has to match the column names of the table!
    """
    try:
        # the sql command does not specify the values but only placeholders with ?
        sql=f"INSERT INTO simulations ({','.join(params.keys())}) VALUES ({','.join(['?']*len(params.keys()))})"
        # values are passed when executing the command
        cur.execute(sql, list(params.values()))
        # commit the changes
        con.commit()
    except Exception as e:
        print(e)