import sys
import time
from glob_inc.client_fl import *
from glob_inc.add_config import client_config

if __name__ == "__main__":
    if len(sys.argv) == '':
        print("Usage: python client.py [client_id]")
        sys.exit(1)

    client_id = "client_" + sys.argv[1]
    print(client_id)
    time.sleep(5)
    fl_client = FLClient(client_id, client_config["host"])
    fl_client.start()
