import base64
import json
import os
import threading
import time
from typing import Any, Dict, List, Literal, Tuple
import xml.etree.ElementTree as xml

import grpc
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2
import requests

from .base import SimEngineBase
from .res.cqsim import engine_pb2
from .res.cqsim import engine_pb2_grpc


class CQSim(SimEngineBase):

    def __init__(
        self,
        *,
        ctrl_addr: str,
        res_addr: str,
        x_token: str,
        proxy_id: str,
    ):
        """Init CQSim engine.

        Args:
            ctrl_addr: Address of CQSim engine controller.
            res_addr: Address of CQSim resource service.
        """
        super().__init__()

        self.ctrl_addr = ctrl_addr
        self.res_addr = res_addr
        self.x_token = x_token
        self.proxy_id = proxy_id

        self.channel = grpc.insecure_channel(ctrl_addr)
        self.engine = engine_pb2_grpc.SimControllerStub(channel=self.channel)

        self.sim_params = None

        self.current_repeat_time = 1

        self.data_thread = None
        self.data_responses = None
        self.data_lock = threading.Lock()
        self.data_cache = {}

        self.logs_thread = None
        self.logs_responses = None
        self.logs_lock = threading.Lock()
        self.logs_cache = []

        self.cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res/cqsim/tmp')
        os.makedirs(self.cwd, exist_ok=True)

    def __del__(self):
        """Close CQSim engine."""
        self.join_threads()
        self.channel.close()

    def control(
        self,
        cmd: Literal['init', 'start', 'pause', 'step', 'resume', 'stop', 'episode', 'param'],
        params: Dict[str, Any] = {},
    ) -> bool:
        """Control CQSim engine.

        Args:
            cmd: Control command. `episode` means ending current episode, `param` means setting simulation parameters.
            params: Control parameters.

        Returns:
            True if success.
        """
        if cmd == 'init':
            self.sim_params = params
            self.join_threads()
            self.init_threads()
            self.fine_params()
            self.set_configs(self.renew_configs(self.reset_configs(self.get_configs())))
            self.engine.SetHttpInfo(engine_pb2.HttpInfo(token=self.x_token))
            p = self.sim_params['task']
            if 'exp_design_id' in p:
                sample = engine_pb2.InitInfo.MultiSample(exp_design_id=p['exp_design_id'])
                self.engine.Init(engine_pb2.InitInfo(multi_sample_config=sample))
            else:
                sample = engine_pb2.InitInfo.OneSample(task_id=p['scenario_id'])
                self.engine.Init(engine_pb2.InitInfo(one_sample_config=sample))
            sim_start_time = timestamp_pb2.Timestamp()
            sim_start_time.FromSeconds(p['sim_start_time'])
            self.engine.Control(engine_pb2.ControlCmd(sim_start_time=sim_start_time))
            sim_duration = duration_pb2.Duration()
            sim_duration.FromSeconds(p['sim_duration'])
            self.engine.Control(engine_pb2.ControlCmd(sim_duration=sim_duration))
            self.control('param', p)
            self.state = 'stopped'
            return True
        elif cmd == 'start':
            self.current_repeat_time = 1
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.START))
            self.state = 'running'
            return True
        elif cmd == 'pause':
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.SUSPEND))
            self.state = 'suspended'
            return True
        elif cmd == 'step':
            return False
        elif cmd == 'resume':
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.CONTINUE))
            self.state = 'running'
            return True
        elif cmd == 'stop':
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.STOP))
            self.state = 'stopped'
            return True
        elif cmd == 'episode':
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.STOP_CURRENT_SAMPLE))
            self.state = 'running'
            return True
        elif cmd == 'param':
            if 'time_step' in params:
                self.sim_params['task']['time_step'] = params['time_step']
                self.engine.Control(engine_pb2.ControlCmd(time_step=params['time_step']))
            if 'speed_ratio' in params:
                self.sim_params['task']['speed_ratio'] = params['speed_ratio']
                self.engine.Control(engine_pb2.ControlCmd(speed_ratio=params['speed_ratio']))
            return True
        else:
            return False

    def monitor(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Monitor CQSim engine.

        Returns:
            Data of simulation process.
            Logs of CQSim engine.
        """
        with self.data_lock:
            data = self.data_cache.copy()
        with self.logs_lock:
            logs = self.logs_cache.copy()
            self.logs_cache.clear()
        return data, logs

    def call(self, str_data: str, bin_data: bytes) -> Tuple[str, bytes]:
        """Any method can be called.

        Args:
            str_data: String data. `reset-proxy` means resetting proxy enviroment.
            bin_data: Binary data.

        Returns:
            String data and binary data.
        """
        if str_data == 'reset-proxy':
            self.set_configs(self.reset_configs(self.get_configs()))
        return '', b''

    def data_callback(self):
        self.data_responses = self.engine.GetSysInfo(engine_pb2.CommonRequest())
        try:
            STOPPED = engine_pb2.EngineNodeState.State.STOPPED
            prev_state, now_state = None, None
            for response in self.data_responses:
                with self.data_lock:
                    self.data_cache['sim_current_time'] = response.sim_current_time.ToSeconds()
                    self.data_cache['sim_duration'] = response.sim_duration.ToSeconds()
                    self.data_cache['real_duration'] = response.real_duration.ToSeconds()
                    self.data_cache['sim_time_step'] = response.sim_time_step
                    self.data_cache['speed_ratio'] = response.speed_ratio
                    self.data_cache['real_speed_ratio'] = response.real_speed_ratio
                    self.data_cache['current_sample_id'] = response.current_sample_id
                    self.data_cache['current_repeat_time'] = self.current_repeat_time

                p = self.sim_params['task']
                if self.state == 'running':
                    now_state = response.node_state[0].state
                    if now_state == STOPPED and now_state != prev_state and prev_state is not None:
                        if 'exp_design_id' not in p or response.current_sample_id == p['exp_sample_num'] - 1:
                            if self.current_repeat_time < p['repeat_times']:
                                time.sleep(0.5)
                                self.control('start')
                                self.current_repeat_time = self.data_cache['current_repeat_time']
                                self.current_repeat_time += 1
                                now_state = None
                    prev_state = now_state
        except grpc.RpcError:
            with self.data_lock:
                self.data_cache.clear()

    def logs_callback(self):
        self.logs_responses = self.engine.GetErrorMsg(engine_pb2.CommonRequest())
        try:
            for response in self.logs_responses:
                with self.logs_lock:
                    self.logs_cache.append(response.msg)
        except grpc.RpcError:
            with self.logs_lock:
                self.logs_cache.clear()

    def join_threads(self):
        if self.data_thread is not None:
            self.data_responses.cancel()
            self.data_thread.join(1)
            self.data_responses = None
            self.data_thread = None
            with self.data_lock:
                self.data_cache.clear()
        if self.logs_thread is not None:
            self.logs_responses.cancel()
            self.logs_thread.join(1)
            self.logs_responses = None
            self.logs_thread = None
            with self.logs_lock:
                self.logs_cache.clear()

    def init_threads(self):
        self.data_thread = threading.Thread(name='data_thread', target=self.data_callback)
        self.data_thread.daemon = True
        self.data_thread.start()
        self.logs_thread = threading.Thread(name='logs_thread', target=self.logs_callback)
        self.logs_thread.daemon = True
        self.logs_thread.start()

    def fine_params(self):
        structs = {}

        r = requests.post(
            f'http://{self.res_addr}/api/scenario/unpack',
            headers={'x-token': self.x_token},
            json={
                'id': self.sim_params['task']['scenario_id'],
                'types': [8],
            },
        )
        msg = r.json()
        typedefine_b64 = msg['data']['typeDefineFile']
        typedefine_str = base64.b64decode(typedefine_b64).decode('utf-8')
        typedefine_xml = xml.fromstring(typedefine_str)

        def extract_structs(type_name):
            type_name = type_name.replace('[]', '')
            if type_name not in structs:
                struct = typedefine_xml.find(f'./Type[@name="{type_name}"]')
                if struct is not None:
                    structs[type_name] = {}
                    for field in struct[0].findall('./Param'):
                        structs[type_name][field.attrib['name']] = field.attrib['type']
                        extract_structs(field.attrib['type'])

        data = self.sim_params['proxy']['data']
        ids = [v['id'] for _, v in data.items()]
        r = requests.post(
            f'http://{self.res_addr}/api/model/unpack',
            headers={'x-token': self.x_token},
            json={
                'ids': ids,
                'types': [1],
            },
        )
        msg = r.json()
        for model in msg['data']:
            model_name = model['name']
            proxy_b64 = model['configFile']
            proxy_str = base64.b64decode(proxy_b64).decode('utf-8')
            proxy_xml = xml.fromstring(proxy_str)

            proxy_inputs = {}
            inputs = data[model_name]['inputs']
            if isinstance(inputs, list):
                for input_name in inputs:
                    input_param = proxy_xml[0].find(f'./Parameter[@name="{input_name}"]')
                    input_type = input_param.attrib['type']
                    proxy_inputs[input_name] = input_type
                    extract_structs(input_type)
            elif isinstance(inputs, dict):
                for input_name, input_value in inputs.items():
                    input_param = proxy_xml[0].find(f'./Parameter[@name="{input_name}"]')
                    input_type = input_param.attrib['type']
                    proxy_inputs[input_name] = {
                        'type': input_type,
                        'value': input_value,
                    }
                    extract_structs(input_type)
            self.sim_params['proxy']['data'][model_name]['inputs'] = proxy_inputs

            proxy_outputs = {}
            outputs = data[model_name]['outputs']
            for output_name in outputs:
                output_param = proxy_xml[0].find(f'./Parameter[@name="{output_name}"]')
                output_type = output_param.attrib['type']
                proxy_outputs[output_name] = output_type
                extract_structs(output_type)
            self.sim_params['proxy']['data'][model_name]['outputs'] = proxy_outputs

        self.sim_params['proxy']['types'] = structs

    def get_configs(self):
        r = requests.post(
            f'http://{self.res_addr}/api/model/unpack',
            headers={'x-token': self.x_token},
            json={
                'ids': [self.proxy_id],
                'types': [1]
            },
        )
        msg = r.json()
        proxy_b64 = msg['data'][0]['configFile']
        proxy_str = base64.b64decode(proxy_b64).decode('utf-8')
        proxy_xml = xml.fromstring(proxy_str)

        r = requests.post(
            f'http://{self.res_addr}/api/scenario/unpack',
            headers={'x-token': self.x_token},
            json={
                'id': self.sim_params['task']['scenario_id'],
                'types': [1, 2],
            },
        )
        msg = r.json()
        scenario_b64 = msg['data']['scenarioFile']
        scenario_str = base64.b64decode(scenario_b64).decode('utf-8')
        scenario_xml = xml.fromstring(scenario_str)
        interaction_b64 = msg['data']['interactionFile']
        interaction_str = base64.b64decode(interaction_b64).decode('utf-8')
        interaction_xml = xml.fromstring(interaction_str)

        return {
            'proxy_xml': proxy_xml,
            'scenario_xml': scenario_xml,
            'interaction_xml': interaction_xml,
        }

    def set_configs(self, configs):
        proxy_bin = xml.tostring(configs['proxy_xml'], encoding='UTF-8', xml_declaration=True)
        proxy_b64 = base64.b64encode(proxy_bin).decode('utf-8')
        with open(f'{self.cwd}/../zlib.dll', 'rb') as f1, \
                open(f'{self.cwd}/../build.bat', 'rb') as f2, \
                open(f'{self.cwd}/configs.json', 'rb') as f3:
            requests.put(
                f'http://{self.res_addr}/api/model/{self.proxy_id}',
                headers={'x-token': self.x_token},
                files=[
                    ('configFile', (None, proxy_b64)),
                    ('dependencyFile', ('zlib.dll', f1)),
                    ('dependencyFile', ('build.bat', f2)),
                    ('dependencyFile', ('configs.json', f3)),
                ],
            )

        scenario_bin = xml.tostring(configs['scenario_xml'], encoding='UTF-8', xml_declaration=True)
        scenario_b64 = base64.b64encode(scenario_bin).decode('utf-8')
        interaction_bin = xml.tostring(configs['interaction_xml'], encoding='UTF-8', xml_declaration=True)
        interaction_b64 = base64.b64encode(interaction_bin).decode('utf-8')
        requests.put(
            f'http://{self.res_addr}/api/scenario/{self.sim_params["task"]["scenario_id"]}',
            headers={'x-token': self.x_token},
            json={
                'scenarioFile': scenario_b64,
                'interactionFile': interaction_b64,
            },
        )

    def reset_configs(self, configs):
        proxy_xml = configs['proxy_xml']
        for el in proxy_xml[0].findall('./Parameter[@unit="proxy"]'):
            proxy_xml[0].remove(el)

        with open(f'{self.cwd}/../configs.json', 'rb') as src, open(f'{self.cwd}/configs.json', 'wb') as tgt:
            tgt.write(src.read())

        scenario_xml = configs['scenario_xml']
        proxy_side = scenario_xml[2].find('./ForceSide[@id="80"]')
        if proxy_side is not None:
            scenario_xml[2].remove(proxy_side)
        proxy_entity = scenario_xml[3].find('./Entity[@id="8080"]')
        if proxy_entity is not None:
            scenario_xml[3].remove(proxy_entity)

        interaction_xml = configs['interaction_xml']
        for node in interaction_xml[0:2]:
            for topic in node.findall('*'):
                name = topic.get('name')
                if name.find('Proxy') != -1:
                    node.remove(topic)
        proxy_pubsub = interaction_xml[2].find(f'./ModelPubSubInfo[@modelID="{self.proxy_id}"]')
        if proxy_pubsub is not None:
            interaction_xml[2].remove(proxy_pubsub)
        for pub_sub in interaction_xml[2]:
            for node in pub_sub:
                for param in node.findall('*'):
                    name = param.get('topicName')
                    if name.find('Proxy') != -1:
                        node.remove(param)

        return {
            'proxy_xml': proxy_xml,
            'scenario_xml': scenario_xml,
            'interaction_xml': interaction_xml,
        }

    def renew_configs(self, configs):
        proxy_xml = configs['proxy_xml']
        proxy_name = proxy_xml.get('displayName', '代理')
        proxy_xml[0].find('./Parameter[@name="InstanceName"]').set('value', proxy_name)
        proxy_xml[0].find('./Parameter[@name="ForceSideID"]').set('value', '80')
        proxy_xml[0].find('./Parameter[@name="ID"]').set('value', '8080')
        for model_name, model_config in self.sim_params['proxy']['data'].items():
            for input_name, input_config in model_config['inputs'].items():
                param_name = f'{model_name}_input_{input_name}'
                param_type = input_config['type'] if isinstance(input_config, dict) else input_config
                xml.SubElement(
                    proxy_xml[0],
                    'Parameter',
                    attrib={
                        'name': param_name,
                        'type': param_type,
                        'displayName': param_name,
                        'usage': 'output',
                        'value': '',
                        'unit': 'proxy',
                    },
                )
            for output_name, output_config in model_config['outputs'].items():
                param_name = f'{model_name}_output_{output_name}'
                param_type = output_config['type'] if isinstance(output_config, dict) else output_config
                xml.SubElement(
                    proxy_xml[0],
                    'Parameter',
                    attrib={
                        'name': param_name,
                        'type': param_type,
                        'displayName': param_name,
                        'usage': 'input',
                        'value': '',
                        'unit': 'proxy',
                    },
                )

        with open(f'{self.cwd}/configs.json', 'w') as f:
            json.dump(self.sim_params, f)

        scenario_xml = configs['scenario_xml']
        proxy_side = xml.fromstring('''
        <ForceSide id="80" name="代理" color="#FFD700">
            <Units>
                <Unit id="8080"/>
            </Units>
        </ForceSide>''')
        scenario_xml[2].append(proxy_side)
        proxy_entity = xml.SubElement(
            scenario_xml[3],
            'Entity',
            attrib={
                'id': '8080',
                'modelID': self.proxy_id,
                'entityName': proxy_name,
                'modelDisplayName': proxy_name,
            },
        )
        proxy_entity.append(proxy_xml[0])

        interaction_xml = configs['interaction_xml']
        proxy_pubsub = xml.SubElement(interaction_xml[2], 'ModelPubSubInfo', attrib={'modelID': self.proxy_id})
        xml.SubElement(proxy_pubsub, 'PublishParams')
        xml.SubElement(proxy_pubsub, 'SubscribeParams')
        for model_name, model_config in self.sim_params['proxy']['data'].items():
            pub_sub = interaction_xml[2].find(f'./ModelPubSubInfo[@modelID="{model_config["id"]}"]')
            if pub_sub is None:
                pub_sub = xml.SubElement(interaction_xml[2], 'ModelPubSubInfo', attrib={'modelID': model_config['id']})
                xml.SubElement(pub_sub, 'PublishParams')
                xml.SubElement(pub_sub, 'SubscribeParams')

            if len(model_config['inputs']) > 0:
                topic_name = f'Proxy_{model_name}_Input'
                topic_type = xml.SubElement(
                    interaction_xml[0],
                    'TopicType',
                    attrib={
                        'name': topic_name,
                        'modelID': model_config['id'],
                        'isTaskFlow': 'false',
                    },
                )
                topic_params = xml.SubElement(topic_type, 'Params')
                xml.SubElement(interaction_xml[1], 'Topic', attrib={'name': topic_name, 'type': topic_name})
                for input_name, input_config in model_config['inputs'].items():
                    xml.SubElement(
                        topic_params,
                        'Param',
                        attrib={
                            'name': input_name,
                            'type': input_config['type'] if isinstance(input_config, dict) else input_config,
                        },
                    )
                    xml.SubElement(
                        proxy_pubsub[0],
                        'PublishParam',
                        attrib={
                            'topicName': topic_name,
                            'topicParamName': input_name,
                            'modelParamName': f'{model_name}_input_{input_name}',
                        },
                    )
                    xml.SubElement(
                        pub_sub[1],
                        'SubscribeParam',
                        attrib={
                            'topicName': topic_name,
                            'topicParamName': input_name,
                            'modelParamName': input_name,
                        },
                    )

            if len(model_config['outputs']) > 0:
                topic_name = f'Proxy_{model_name}_Output'
                topic_type = xml.SubElement(
                    interaction_xml[0],
                    'TopicType',
                    attrib={
                        'name': topic_name,
                        'modelID': model_config['id'],
                        'isTaskFlow': 'false',
                    },
                )
                topic_params = xml.SubElement(topic_type, 'Params')
                xml.SubElement(interaction_xml[1], 'Topic', attrib={'name': topic_name, 'type': topic_name})
                for output_name, output_config in model_config['outputs'].items():
                    xml.SubElement(
                        topic_params,
                        'Param',
                        attrib={
                            'name': output_name,
                            'type': output_config['type'] if isinstance(output_config, dict) else output_config,
                        },
                    )
                    xml.SubElement(
                        pub_sub[0],
                        'PublishParam',
                        attrib={
                            'topicName': topic_name,
                            'topicParamName': output_name,
                            'modelParamName': output_name,
                        },
                    )
                    xml.SubElement(
                        proxy_pubsub[1],
                        'SubscribeParam',
                        attrib={
                            'topicName': topic_name,
                            'topicParamName': output_name,
                            'modelParamName': f'{model_name}_output_{output_name}',
                        },
                    )

        return {
            'proxy_xml': proxy_xml,
            'scenario_xml': scenario_xml,
            'interaction_xml': interaction_xml,
        }
