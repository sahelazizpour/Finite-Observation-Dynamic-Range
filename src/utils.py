import sqlite3
import pandas as pd

# quick fetch of data from database tables
def fetch_from_database(con, cur, table, params):
    """
        fetch from table

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
    cur.execute(f"SELECT * FROM {table} WHERE {' AND '.join([f'{key}=?' for key in params.keys()])}", list(params.values()))
    return cur.fetchall()

# for quick insertion of data into database tables
def insert_into_database(con, cur, table, params):
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
        sql=f"INSERT INTO {table} ({','.join(params.keys())}) VALUES ({','.join(['?']*len(params.keys()))})"
        # values are passed when executing the command
        cur.execute(sql, list(params.values()))
    except Exception as e:
        print(e)

# naive check if entry exists; not sure about performance
def exists_in_database(con, cur, table, params):
    """
        check if entry exists in table

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
    cur.execute(f"SELECT * FROM {table} WHERE {' AND '.join([f'{key}=?' for key in params.keys()])}", list(params.values()))
    return len(cur.fetchall()) > 0