from glob_inc.utils import *
from glob_inc.add_config import client_config
from model_api.src.lenet_api import start_trainning_mnist
from model_api.src.lenet_api import *
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import paho.mqtt.client as mqtt
import json
import torch
from collections  import Counter
class FLClient():
    def __init__(self, client_id, broker_name):
        self.client_id = client_id
        self.broker_name = broker_name
        self.start_line = client_config["start_line"]
        self.start_benign = client_config["start_benign"]
        self.start_main_dga = client_config["start_main_dga"]
        self.count = client_config["count"]
        self.alpha = client_config["alpha"]
        self.arr_num_line = client_config["arr_num_line"]
        # self.num_line = self.arr_num_line[self.count]
        self.num_line = 200

        self.waiting = False

        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe

    def on_connect(self, client, userdata, flags, rc):
        print("do on_connect")
        print_log(f"Connected with result code {rc}")
        if not self.waiting:
            self.join_dFL_topic()

    def on_disconnect(self, client, userdata, rc):
        print("do on_disconnect")
        print_log(f"Disconnected with result code {rc}")
        if not self.waiting:
            self.client.reconnect()

    def on_message(self, client, userdata, msg):
        print("do on_message")
        print_log(f"on_message {client._client_id.decode()}")
        print_log(f"RECEIVED msg from {msg.topic}")
        # if self.waiting and msg.topic != "dynamicFL/wait/" + self.client_id:
        #     return
        topic = msg.topic
        print(topic)
        if topic == "dynamicFL/req/"+self.client_id:
            self.handle_cmd(msg)
        elif topic == "dynamicFL/model/all_client":
            self.handle_model(client, userdata, msg)
        elif self.waiting:
            if "dynamicFL/wait/" in topic:
                self.handle_recall(msg)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("do on_subscribe")
        print_log(f"Subscribed: {mid} {granted_qos}")

    def do_evaluate_connection(self):
        print("do do_evalate_connection")
        print_log("doing ping")
        result = ping_host(self.broker_name)
        result["client_id"] = self.client_id
        result["task"] = "EVA_CONN"
        self.client.publish(topic="dynamicFL/res/"+self.client_id, payload=json.dumps(result))
        print_log(f"Published to topic dynamicFL/res/{self.client_id}")
        return result

    def do_train(self):
        print("do_train")
        print_log(f"start training")
        client_id = self.client_id

        _,_,test_loader,train_dataset = get_dataset()


        dict_data_users = sample_mnist_data(dataset=train_dataset, num_devices=server_config['NUM_DEVICE'])
        print(dict_data_users)
        for i, user_data in dict_data_users.items():
            print(f"Thiết bị {i}: {len(user_data)} mẫu")
            print("-----------------------------------")
            print(user_data)

            
        client_data_loader = get_dataloader_for_client(client_id=self.client_id, train_dataset= train_dataset, dict_users=dict_data_users)
        result, protos = train_mnist(client_data_loader, test_loader)

        torch.save(result, f'model_client/model_client_{client_id}.pt')

        # Convert tensors to numpy arrays
        result_np = {key: value.cpu().numpy().tolist() for key, value in result.items()}
        payload = {
            "task": "TRAIN",
            "weight": result_np,
            "protos": protos
        }
        self.client.publish(topic="dynamicFL/res/" + client_id, payload=json.dumps(payload))

    def do_evaluate_data(self):
        print("do_evaluate_data")
        pass

    def do_test(self):
        print("do_test")
        pass

    def do_update_model(self):
        print("do_update_model")
        pass

    def do_stop_client(self):
        print("do_stop_client")
        print_log("stop client")
        self.client.loop_stop()

    def handle_task(self, msg):
        print("handle_task")
        task_name = msg.payload.decode("utf-8")
        print(task_name)
        if task_name == "EVA_CONN":
            self.do_evaluate_connection()
        elif task_name == "EVA_DATA":
            self.do_evaluate_data()
        elif task_name == "TRAIN":
            self.do_train()
        elif task_name == "TEST":
            self.do_test()
        elif task_name == "UPDATE":
            self.do_update_model()
        elif task_name == "REJECTED":
            self.do_add_errors()
        elif task_name == "STOP":
            self.do_stop_client()
        elif task_name == "MOVE_WAIT":
            self.do_wait()
        else:
            print_log(f"Command {task_name} is not supported")

    def join_dFL_topic(self):
        print("do join_dFL_topic")
        self.client.publish(topic="dynamicFL/join", payload=self.client_id)
        print_log(f"{self.client_id} joined dynamicFL/join of {self.broker_name}")

    def do_add_errors(self):
        print("do_add_errors")
        publish.single(topic="dynamicFL/errors", payload=self.client_id, hostname=self.broker_name, client_id=self.client_id)

    def wait_for_model(self):
        print("do wait_for_model")
        msg = subscribe.simple("dynamicFL/model", hostname=self.broker_name)
        with open("mymodel.pt", "wb") as fo:
            fo.write(msg.payload)
        print_log(f"{self.client_id} write model to mymodel.pt")

    def handle_cmd(self, msg):
        print("do handle_cmd")
        print_log("wait for cmd")
        self.handle_task(msg)
 
    def handle_model(self, client, userdata, msg):
        print("do handle_model")
        print_log("receive model")
        with open("newmode.pt", "wb") as f:
            f.write(msg.payload)
        print_log("done write model")
        result = {
            "client_id": self.client_id,
            "task": "WRITE_MODEL" 
        }
        self.client.publish(topic="dynamicFL/res/"+self.client_id, payload=json.dumps(result))

    def handle_recall(self, msg):
        print("do handle_recall")
        task_name = msg.payload.decode("utf-8")
        if task_name == "RECALL":
            self.do_recall()

    def start(self):
        print("do start")
        self.client.connect(self.broker_name, port=1883, keepalive=3600)
        self.client.message_callback_add("dynamicFL/model/all_client", self.handle_model)
        self.client.loop_start()
        print(self.waiting)
        if not self.waiting:
            self.client.subscribe(topic="dynamicFL/model/all_client")
            self.client.subscribe(topic="dynamicFL/req/" + self.client_id)
            self.client.subscribe(topic="dynamicFL/wait/" + self.client_id)
            self.client.publish(topic="dynamicFL/join", payload=self.client_id)
            print_log(f"{self.client_id} joined dynamicFL/join of {self.broker_name}")

        self.client._thread.join()
        print_log("client exits")
    