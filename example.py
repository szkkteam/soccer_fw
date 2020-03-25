import os
import soccer_fw as fw
import db_conn as conn
import pandas as pd

def get_raw_data():
    matches = [
        8243388
    ]

    configuration = {
        'ssh_address_or_host': (os.environ.get('SSH_ADDRESS'), 22),
        'ssh_username': os.environ.get('SSH_USERNAME'),
        'ssh_password': os.environ.get('SSH_PASSWORD'),
        'remote_bind_address': ('127.0.0.1', 5432),
        'local_bind_address': ('127.0.0.1', 8080),  # could be any available port
    }
    tunnel = conn.utils.Tunnel(config=configuration)
    configuration = {
        'db_username': os.environ.get('DB_USERNAME'),
        'db_password': os.environ.get('DB_PASSWORD'),
        'db_address': ('127.0.0.1', 8080),
        'db_name': os.environ.get('DB_NAME')
    }
    db = conn.psql.Connection(config=configuration, tunnel=tunnel)

    df = conn.query.sc_soccer.get_combined_data(db, include_lineups=True, match_ids=matches)
    df.to_csv("simple_test_raw.csv")

    db.terminate()

def preprocess():
    df = pd.read_csv("simple_test_raw.csv")
    subs = df.filter(regex=(".*substitute.*"))
    print(subs)


def main():
    get_raw_data()












if __name__ == '__main__':
    main()