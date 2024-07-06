import json
import paho.mqtt.client as client
import logging
import datetime
from glob_inc.utils import *
from glob_inc.server_fl import *
from glob_inc.add_config import server_config

LOG_DIR = 'model_api/src/logs'
LOG_FILE = f"model_api/src/logs/app-{datetime.today().strftime('%Y-%m-%d')}.log"

    
if __name__ == "__main__":
    broker_name = server_config['host']
    port_mqtt = server_config['port_mqtt']

    server = Server(broker_name, port_mqtt, server_config['ID'])
    server.connect(broker_name, port=port_mqtt, keepalive=3600)
    server.on_connect
    server.on_disconnect
    server.on_message
    server.on_subscribe
    server.loop_start()
    server.subscribe(topic = "dynamicFL/join")
    print_log(f"server sub to dynamicFL/join of {broker_name}")
    print_log("server is waiting for clients to join the topic ...")

    while (server.NUM_DEVICE > len(server.client_dict)):
       time.sleep(1)

    server.start_round()
    server._thread.join()
    time.sleep(10)
    print_log("server exits")
 