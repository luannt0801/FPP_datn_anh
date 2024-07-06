import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import paho.mqtt.client as mqtt
from paho.mqtt.client import Client as MqttClient
import time
import threading
import json
import logging
import torch
from sklearn.cluster import KMeans
import pandas as pd
import torch
from collections import OrderedDict
from glob_inc.add_config import server_config
from glob_inc.utils import *
from model_api.src.lenet_api import *

logger = logging.getLogger()

class Server(MqttClient):
    def __init__(self, client_fl_id, clean_session=True, userdata=None, protocol=mqtt.MQTTv311):
        super().__init__(client_fl_id, clean_session, userdata, protocol)
        
        # Set callbacks
        self.on_connect = self.on_connect_callback
        self.on_message = self.on_message_callback
        self.on_disconnect = self.on_disconnect_callback
        self.on_subscribe = self.on_subscribe_callback

        self.client_dict = {}
        self.client_dict_wait = {}

        self.client_trainres_dict = {}
        self.client_train_dict_wait = {}
        self.client_trainres_protos = {} # FPP
        self.penalty_lambda = {} # FPP

        self.client_training_times = {}  # Dict để lưu trữ thời gian training của từng client

        self.list_drop = []

        self.NUM_ROUND = server_config['NUM_ROUND']
        self.NUM_DEVICE = server_config['NUM_DEVICE']
        self.time_between_two_round = server_config['time_between_two_round']
        self.round_state = server_config['round_state']
        self.n_round = server_config['n_round']

        self.round_counter = 0
        self.round_start_time = None
        self.round_end_time = None
        self.round_durations = []

        self.server_prototypes = {}
    
    # add new
    def _remove_client(self, drop_client_id):
        if drop_client_id in self.client_trainres_dict:
            self.client_dict_wait[drop_client_id] = self.client_dict.pop(drop_client_id)
            self.client_train_dict_wait[drop_client_id] = self.client_trainres_dict.pop(drop_client_id)

    def _add_client(self, client_id):
        if client_id in self.client_dict_wait:
            self.client_dict[client_id] = self.client_dict_wait.pop(client_id)
            self.client_trainres_dict[client_id] = self.client_train_dict_wait.pop(client_id)

    # check connect to broker return result code
    def on_connect_callback(self, client, userdata, flags, rc):
        print("do on_connect_callback")
        print_log("Connected with result code "+str(rc))

    def on_disconnect_callback(self, client, userdata, rc):
        print("do disconnect call back")
        print_log("Disconnected with result code "+str(rc))
        self.reconnect()

    # handle message receive from client
    def on_message_callback(self, client, userdata, msg):
        print("do message call back")
        print(f"received msg from {msg.topic}")
        topic = msg.topic
        if topic == "dynamicFL/join": # topic is join --> handle_join
            self.handle_join(self, userdata, msg)
        elif "dynamicFL/res" in topic:
            tmp = topic.split("/")
            this_client_id = tmp[2]
            self.handle_res(this_client_id, msg)
        elif "dynamicFL/wait" in topic:
            tmp = topic.split("/")
            this_client_id = tmp[2]
            self.handle_wait(this_client_id, msg)
        # elif "$SYS/#":
        #     print("Received message:", str(msg.payload.decode("utf-8")))

    def on_subscribe_callback(self, mosq, obj, mid, granted_qos):
        print("do on_subcribe_callback")
        print_log("Subscribed: " + str(mid) + " " + str(granted_qos))

    def send_task(self, task_name, client, this_client_id):
        print("do send_task")
        print(this_client_id)
        print(task_name)
        print_log("publish to " + "dynamicFL/req/"+this_client_id)
        if task_name == "TRAIN":
            self.client_training_times[this_client_id] = time.time()  # Ghi nhận thời gian bắt đầu
        self.publish(topic="dynamicFL/req/"+this_client_id, payload=task_name)

    def send_model(self, path, client, this_client_id):
        print("do send_model")
        f = open(path, "rb")
        data = f.read()
        f.close()
        self.publish(topic="dynamicFL/model/all_client", payload=data)

    def handle_res(self, this_client_id, msg):
        print("do handle_res")
        data = json.loads(msg.payload)
        cmd = data["task"]
        if cmd == "EVA_CONN":
            print_log(f"{this_client_id} complete task EVA_CONN")
            self.handle_pingres(this_client_id, msg)
        elif cmd == "TRAIN":
            print_log(f"{this_client_id} complete task TRAIN")
            print(self.client_training_times)
            if self.client_training_times[this_client_id]:
                self.client_training_times[this_client_id] = time.time() - self.client_training_times[this_client_id]  # Tính thời gian training
            self.handle_trainres(this_client_id, msg)
        elif cmd == "WRITE_MODEL":
            print_log(f"{this_client_id} complete task WRITE_MODEL")
            self.handle_update_writemodel(this_client_id, msg)

    def handle_join(self, client, userdata, msg):
        print("handle join")
        this_client_id = msg.payload.decode("utf-8")
        print("joined from"+" "+this_client_id)
        self.client_dict[this_client_id] = {
            "state": "joined"
        }
        self.subscribe(topic="dynamicFL/res/"+this_client_id)
        
    def handle_wait(self):
        self.recall_clients_from_wait()
 
    def handle_pingres(self, this_client_id, msg):
        print("do handle_pingres")
        print(msg.topic+" "+str(msg.payload.decode()))
        ping_res = json.loads(msg.payload)
        this_client_id = ping_res["client_id"]
        if ping_res["packet_loss"] == 0.0:
            print_log(f"{this_client_id} is a good client")
            state = self.client_dict[this_client_id]["state"]
            print_log(f"state {this_client_id}: {state}, round: {self.n_round}")
            if state == "joined" or state == "trained":
                self.client_dict[this_client_id]["state"] = "eva_conn_ok"
                #send_model("saved_model/FashionMnist.pt", server, this_client_id)
                #print(client_dict)
                count_eva_conn_ok = sum(1 for client_info in self.client_dict.values() if client_info["state"] == "eva_conn_ok")
                if(count_eva_conn_ok == self.NUM_DEVICE):
                    print_log("publish to " + "dynamicFL/model/all_client")
                    self.send_model("saved_model/LENETModel.pt", "s", this_client_id) # hmm

    def handle_trainres(self, this_client_id, msg):
        payload = json.loads(msg.payload.decode())
        
        self.client_trainres_dict[this_client_id] = payload["weight"]
        self.client_trainres_protos[this_client_id] = payload["protos"]
        state = self.client_dict[this_client_id]["state"]
        if state == "model_recv":
            self.client_dict[this_client_id]["state"] = "trained"
        print("done train!")
        
    def handle_update_writemodel(self, this_client_id, msg):
        print("do handle_update_writemodel")
        state = self.client_dict[this_client_id]["state"]
        if state == "eva_conn_ok":
            self.client_dict[this_client_id]["state"] = "model_recv"
            self.send_task("TRAIN", self, this_client_id) # hmm
            count_model_recv = sum(1 for client_info in self.client_dict.values() if client_info["state"] == "model_recv")
            if(count_model_recv == self.NUM_DEVICE):
                print_log(f"Waiting for training round {self.n_round} from client...")
                logger.info(f"Start training round {self.n_round} ...")


    def start_round(self):
        self.round_start_time = time.time()
        print("do start_round")
        self.n_round
        self.n_round = self.n_round + 1
        print_log(f"server start round {self.n_round}")
        logger.info(f"server start round {self.n_round}")
        self.round_state = "started"
        print(self.client_dict)
        for client_i in self.client_dict:
            self.send_task("EVA_CONN", self, client_i) # hmm
        while (len(self.client_trainres_dict) != self.NUM_DEVICE):
            time.sleep(1)
        time.sleep(1)
        self.end_round()

    def do_aggregate(self):
        print("do do_aggregate")
        print_log("Do aggregate ...")
        logger.info("start aggregate ...")
        self.aggregated_models()
        logger.info("end aggregate!")

    def do_aggregate_with_proto(self):
        print("do do_aggregate with proto penalty")
        print_log("Do aggregate with proto penalty ...")
        logger.info("start aggregate with proto penalty ...")
        self.aggregated_models_with_Protos()
        logger.info("end aggregate with proto penalty!")

        
    def handle_next_round_duration(self):
        print("do handle_next_round_duration")
        #if len(client_trainres_dict) < len(client_dict):
            #time_between_two_round = time_between_two_round + 10
        while (len(self.client_trainres_dict) < self.NUM_DEVICE):
            time.sleep(1)

    def end_round(self):
        print("do end_round")
        logger.info(f"server end round {self.n_round}")
        print_log(f"server end round {self.n_round}")
        self.round_end_time = time.time()
        round_duration = self.round_end_time - self.round_start_time
        self.round_durations.append(round_duration)
        round_state = "finished"

        if server_config['mode'] == 'FedAvg':
            if self.n_round < self.NUM_ROUND:
                self.handle_next_round_duration()
                self.do_aggregate()
                t = threading.Timer(self.time_between_two_round, self.start_round)
                t.start()
            else: 
                self.do_aggregate()
                for c in self.client_dict:
                    self.send_task("STOP", self, c)
                    print_log(f"send task STOP {c}")
                for c in self.list_drop:
                    self.send_task("STOP", self, c)
                    print_log(f"send task STOP {c}")
                logger.info(f"Stop all!")
                self.loop_stop()

        elif server_config['mode'] == 'FPP':
            train_loader, test_loader, prototype_loader, train_dataset = get_dataset()
            if self.n_round == 1:
                self.do_aggregate()
                self.server_prototypes = calculate_server_prototypes(prototype_loader)
                t = threading.Timer(self.time_between_two_round, self.start_round)
                t.start()
            else:
                # server proto in new round
                self.server_prototypes = calculate_server_prototypes(prototype_loader)
                print("\n server_prototypes: ", self.server_prototypes)
                print("\n client_trainres_protos: ", self.client_trainres_protos)
                for key, value in self.client_trainres_protos.items():
                    print("\n" + key)
                distance_dict = calculate_prototype_distance(self.client_trainres_protos, self.n_round, self.server_prototypes)
                print("\n distance_dict: ", distance_dict)
                self.server_prototypes.clear()

                self.penalty_lambda = calculate_penalty(distance_dict)
                print("\n penalty_lambda", self.penalty_lambda)
                
                if self.n_round < self.NUM_ROUND:
                    self.handle_next_round_duration()
                    self.do_aggregate_with_proto()
                    t = threading.Timer(self.time_between_two_round, self.start_round)
                    t.start()
                else:
                    self.do_aggregate_with_proto()
                    for c in self.client_dict:
                        self.send_task("STOP", self, c)
                        print_log(f"send task STOP {c}")
                    logger.info(f"Stop all!")
                    self.loop_stop()
        else:
            raise ValueError ("Mode is incorrect")

    def aggregated_models(self):
        # Khởi tạo một OrderedDict để lưu trữ tổng của các tham số của mỗi layer
        sum_state_dict = OrderedDict()

        # Lặp qua các giá trị của dict chính và cộng giá trị của từng tham số vào sum_state_dict
        for client_id, state_dict in self.client_trainres_dict.items():
            for key, value in state_dict.items():
                if key in sum_state_dict:
                    sum_state_dict[key] = sum_state_dict[key] + torch.tensor(value, dtype=torch.float32)
                else:
                    sum_state_dict[key] = torch.tensor(value, dtype=torch.float32)

        # Tính trung bình của các tham số
        num_models = len(self.client_trainres_dict)
        avg_state_dict = OrderedDict((key, value / num_models) for key, value in sum_state_dict.items())
        torch.save(avg_state_dict, f'model_round/model_server_round_{self.n_round}.pt')
        torch.save(avg_state_dict, "saved_model/LENETModel.pt")
        #delete parameter in client_trainres to start new round
        self.client_trainres_dict.clear()

        # Trainning time clients
        with open(f'training_times/round_times.txt', 'a') as f:
            f.write(f'round {self.n_round}: {self.round_durations[self.n_round-1]}\n')
            f.write(f'Remove to wait: {self.list_drop}\n')
            for client_id, training_time in self.client_training_times.items():
                f.write(f'{client_id}: {training_time} seconds\n')

        # clear trainning time round
        self.client_training_times.clear()

    
    def aggregated_models_with_Protos(self):
        # Khởi tạo một OrderedDict để lưu trữ tổng của các tham số của mỗi layer

        sum_state_dict = OrderedDict()
        print_log("######### Check penalty lamda ############")
        print(self.penalty_lambda)
        weight_sum = 0

        for client_id, state_dict in self.client_trainres_dict.items():
            int_client_id = int(client_id.split('_')[1])
            client_penalty = self.penalty_lambda.get(int_client_id, {})
            print(f"Client: {client_id}, Penalty: {client_penalty}")
            for key, value in state_dict.items():
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.float32)
                full_key = key.split('.')[-1]
                try:
                    label = int(full_key)
                except ValueError:
                    continue
                weight = client_penalty.get(label, 1)
                if key in sum_state_dict:
                    sum_state_dict[key] += value * weight
                else:
                    sum_state_dict[key] = value * weight
            weight_sum += sum(client_penalty.values())
        if weight_sum == 0:
            raise ValueError("Sum of weights is zero. Cannot normalize by zero.")
        
        avg_state_dict = OrderedDict((key, value / weight_sum) for key, value in sum_state_dict.items())
        torch.save(avg_state_dict, f'model_round_{self.n_round}.pt')
        torch.save(avg_state_dict, "saved_model/LENETModel.pt")
        self.client_trainres_dict.clear()
        self.penalty_lambda.clear()
