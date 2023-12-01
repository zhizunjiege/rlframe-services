import base64
import json
import logging
import os
import subprocess
import threading
import time
from typing import Any, Dict, List, Tuple, Union
import xml.etree.ElementTree as xml

import grpc
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2
import requests

from ..base import SimEngineBase, AnyDict, CommandType, EngineState
from .engine import engine_pb2
from .engine import engine_pb2_grpc


class CQSIM(SimEngineBase):

    def __init__(
        self,
        *,
        ctrl_addr='localhost:50041',
        res_addr='localhost:8001',
        x_token='Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRhc2NvcGUiOiIiLCJleHAiOjQ4MTAxOTcxNTQsImlkZW50aXR5 \
            IjoxLCJuaWNlIjoiYWRtaW4iLCJvcmlnX2lhdCI6MTY1NjU2MTE1NCwicm9sZWlkIjoxLCJyb2xla2V5IjoiYWRtaW4iLCJyb2xlbmFtZ \
            SI6Iuezu-e7n-euoeeQhuWRmCJ9.BvjGw26L1vbWHwl0n8Y1_yTF-fiFNZNmIw20iYe7ToU',
        proxy_id='',
        scenario_id=0,
        exp_design_id=0,
        repeat_times=1,
        sim_start_time=0,
        sim_duration=1,
        time_step=50,
        speed_ratio=1,
        data: Union[Dict[str, Any], List[Dict[str, Any]]] = {},
        routes: Union[Dict[str, List[str]], Dict[str, Any]] = {},
        simenv_addr='localhost:10001',
        sim_step_ratio=1,
        sim_term_func='',
    ):
        """Init CQSIM engine.

        Args:
            ctrl_addr: control server address.
            res_addr: resource server address.
            x_token: token for resource server.
            proxy_id: proxy model ID.
            scenario_id: scenario ID.
            exp_design_id: experimental design ID.
            repeat_times: times to repeat the scenario or experiment.
            sim_start_time: start time of scenario in timestamp.
            sim_duration: simulation duration in seconds.
            time_step: time step in milliseconds.
            speed_ratio: speed ratio.
            data: input and output data needed for interaction.
            routes: routes for engine.
            simenv_addr: simenv service address.
            sim_step_ratio: number of steps to take once request for decision.
            sim_term_func: termination function written in c++.
        """
        super().__init__()

        self.ctrl_addr = ctrl_addr
        self.res_addr = res_addr
        self.x_token = x_token
        self.proxy_id = proxy_id

        self.scenario_id = scenario_id
        self.exp_design_id = exp_design_id
        self.exp_sample_num = 0
        self.repeat_times = repeat_times
        self.sim_start_time = sim_start_time
        self.sim_duration = sim_duration
        self.time_step = time_step
        self.speed_ratio = speed_ratio

        self.types = {}
        self.data = data if isinstance(data, dict) else {item['name']: item for item in data}
        self.routes = routes if isinstance(routes, dict) else {item['addr']: item['models'] for item in routes}
        self.simenv_addr = simenv_addr
        self.sim_step_ratio = sim_step_ratio
        self.sim_term_func = sim_term_func

        self.channel = grpc.insecure_channel(self.ctrl_addr)
        self.engine = engine_pb2_grpc.SimControllerStub(channel=self.channel)

        self.cache = {}
        self.check_args()

        self.current_repeat_time = 1

        self.data_thread = None
        self.data_responses = None
        self.data_lock = threading.Lock()
        self.data_cache = {}

        self.logs_thread = None
        self.logs_responses = None
        self.logs_lock = threading.Lock()
        self.logs_cache = []

        self.cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
        os.makedirs(self.cwd, exist_ok=True)
        self.logger = logging.getLogger('simenv.cqsim')

        self.set_configs(self.renew_configs(self.reset_configs(self.get_configs())))

    def check_args(self):
        # check scenario
        if self.exp_design_id > 0:
            r = requests.get(
                f'http://{self.res_addr}/api/design/{self.exp_design_id}',
                headers={'x-token': self.x_token},
            )
            msg = r.json()
            if not msg['data']:
                raise ValueError('Invalid exp_design_id')
            self.scenario_id = msg['data']['scenarioId']
            self.exp_sample_num = msg['data']['sampleSize']
        if self.scenario_id <= 0:
            raise ValueError('Missing scenario_id or exp_design_id')
        r = requests.post(
            f'http://{self.res_addr}/api/scenario/unpack',
            headers={'x-token': self.x_token},
            json={
                'id': self.scenario_id,
                'types': [1, 2, 8],
            },
        )
        msg = r.json()
        if not msg['data']:
            raise ValueError('Invalid scenario_id')

        # cache scenario related definitions
        scenario_b64 = msg['data']['scenarioFile']
        scenario_str = base64.b64decode(scenario_b64).decode('utf-8')
        scenario_xml = xml.fromstring(scenario_str)
        interaction_b64 = msg['data']['interactionFile']
        interaction_str = base64.b64decode(interaction_b64).decode('utf-8')
        interaction_xml = xml.fromstring(interaction_str)
        typedefine_b64 = msg['data']['typeDefineFile']
        typedefine_str = base64.b64decode(typedefine_b64).decode('utf-8')
        typedefine_xml = xml.fromstring(typedefine_str)
        self.cache['scenario_xml'] = scenario_xml
        self.cache['interaction_xml'] = interaction_xml
        self.cache['typedefine_xml'] = typedefine_xml

        # check models
        model_ids = [self.proxy_id] + [v['modelid'] for _, v in self.data.items()]
        r = requests.post(
            f'http://{self.res_addr}/api/model/unpack',
            headers={'x-token': self.x_token},
            json={
                'ids': model_ids,
                'types': [1],
            },
        )
        msg = r.json()
        if len(msg['data']) != len(model_ids):
            invalid_ids = set(model_ids) - set([v['id'] for v in msg['data']])
            raise ValueError(f'Invalid modelid: {invalid_ids}')

        # check model inputs and outputs
        structs = {}
        for model in msg['data']:
            if model['id'] == self.proxy_id:
                proxy_b64 = model['configFile']
                proxy_str = base64.b64decode(proxy_b64).decode('utf-8')
                proxy_xml = xml.fromstring(proxy_str)
                self.cache['proxy_xml'] = proxy_xml
                continue

            model_b64 = model['configFile']
            model_str = base64.b64decode(model_b64).decode('utf-8')
            model_xml = xml.fromstring(model_str)

            model_config = self.data[model['name']]

            model_inputs = {}
            init = isinstance(model_config['inputs'], dict)
            for input_name in model_config['inputs']:
                input_param = model_xml[0].find(f'./Parameter[@name="{input_name}"]')
                if input_param is None or input_param.attrib['usage'].find('input') == -1:
                    raise ValueError(f'Invalid input name {input_name} for modelid {model["id"]}')
                input_type = input_param.attrib['type']
                self.extract_struct(structs, typedefine_xml, input_type)
                model_inputs[input_name] = {
                    'type': input_type,
                    'value': model_config['inputs'][input_name],
                } if init else input_type
            model_config['inputs'] = model_inputs

            model_outputs = {}
            for output_name in model_config['outputs']:
                output_param = model_xml[0].find(f'./Parameter[@name="{output_name}"]')
                if output_param is None or output_param.attrib['usage'].find('output') == -1:
                    raise ValueError(f'Invalid output name {output_name} for modelid {model["id"]}')
                output_type = output_param.attrib['type']
                self.extract_struct(structs, typedefine_xml, output_type)
                model_outputs[output_name] = output_type
            model_config['outputs'] = model_outputs
        self.types = structs

    def extract_struct(self, structs, type_define, type_name):
        type_name = type_name.replace('[]', '')
        if type_name not in structs:
            type_node = type_define.find(f'./Type[@name="{type_name}"]')
            if type_node is not None:
                structs[type_name] = {}
                for field in type_node[0].findall('./Param'):
                    param_name, param_type = field.attrib['name'], field.attrib['type']
                    structs[type_name][param_name] = param_type
                    self.extract_struct(structs, type_define, param_type)

    def control(self, type: CommandType, params: AnyDict = {}) -> bool:
        """Control CQSIM engine.

        Args:
            type: Command type.
            params: Command params.

        Returns:
            True if success.
        """
        if type == CommandType.INIT:
            self.init_threads()
            self.engine.SetHttpInfo(engine_pb2.HttpInfo(token=self.x_token))
            if self.exp_design_id > 0:
                sample = engine_pb2.InitInfo.MultiSample(exp_design_id=self.exp_design_id)
                self.engine.Init(engine_pb2.InitInfo(multi_sample_config=sample))
            else:
                sample = engine_pb2.InitInfo.OneSample(task_id=self.scenario_id)
                self.engine.Init(engine_pb2.InitInfo(one_sample_config=sample))
            sim_start_time = timestamp_pb2.Timestamp()
            sim_start_time.FromMilliseconds(self.sim_start_time)
            self.engine.Control(engine_pb2.ControlCmd(sim_start_time=sim_start_time))
            sim_duration = duration_pb2.Duration()
            sim_duration.FromSeconds(self.sim_duration)
            self.engine.Control(engine_pb2.ControlCmd(sim_duration=sim_duration))
            self.control(CommandType.PARAM, {'time_step': self.time_step, 'speed_ratio': self.speed_ratio})
            self.state = EngineState.STOPPED
            self.logger.info('CQSIM engine inited.')
            return True
        elif type == CommandType.START:
            self.current_repeat_time = 1
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.START))
            self.state = EngineState.RUNNING
            self.logger.info('CQSIM engine started.')
            return True
        elif type == CommandType.PAUSE:
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.SUSPEND))
            self.state = EngineState.SUSPENDED
            self.logger.info('CQSIM engine paused.')
            return True
        elif type == CommandType.STEP:
            self.logger.warning('CQSIM engine does not support step.')
            return False
        elif type == CommandType.RESUME:
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.CONTINUE))
            self.state = EngineState.RUNNING
            self.logger.info('CQSIM engine resumed.')
            return True
        elif type == CommandType.STOP:
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.STOP))
            self.join_threads()
            self.state = EngineState.STOPPED
            self.logger.info('CQSIM engine stopped.')
            return True
        elif type == CommandType.EPISODE:
            self.engine.Control(engine_pb2.ControlCmd(run_cmd=engine_pb2.ControlCmd.RunCmdType.STOP_CURRENT_SAMPLE))
            self.state = EngineState.RUNNING
            self.logger.info('CQSIM engine stopped current episode.')
            return True
        elif type == CommandType.PARAM:
            if 'time_step' in params:
                self.time_step = params['time_step']
                self.engine.Control(engine_pb2.ControlCmd(time_step=params['time_step']))
                self.logger.info(f'CQSIM engine time step set to {params["time_step"]}.')
            if 'speed_ratio' in params:
                self.speed_ratio = params['speed_ratio']
                self.engine.Control(engine_pb2.ControlCmd(speed_ratio=params['speed_ratio']))
                self.logger.info(f'CQSIM engine speed ratio set to {params["speed_ratio"]}.')
            return True
        else:
            self.logger.warning(f'Unknown command type {type}.')
            return False

    def monitor(self) -> Tuple[List[AnyDict], List[str]]:
        """Monitor CQSIM engine.

        Returns:
            Data of CQSIM engine.
            Logs of CQSIM engine.
        """
        with self.data_lock:
            data = self.data_cache.copy()
        with self.logs_lock:
            logs = self.logs_cache.copy()
            self.logs_cache.clear()
        return data, logs

    def call(self, name: str, dstr='', dbin=b'') -> Tuple[str, str, bytes]:
        """Any method can be called.

        Args:
            name: Name of method. `reset-proxy` means resetting proxy enviroment.
            dstr: String data.
            dbin: Binary data.

        Returns:
            Name of method, string data and binary data.
        """
        if name == 'reset-proxy':
            self.set_configs(self.reset_configs(self.get_configs()))
            self.logger.info('CQSIM engine proxy reseted.')
        else:
            self.logger.warning(f'Unknown method {name}.')
        return name, '', b''

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

                if self.state == EngineState.RUNNING:
                    now_state = response.node_state[0].state
                    if now_state == STOPPED and now_state != prev_state and prev_state is not None:
                        if self.exp_design_id <= 0 or response.current_sample_id == self.exp_sample_num - 1:
                            if self.current_repeat_time < self.repeat_times:
                                self.control(CommandType.STOP)
                                time.sleep(1)
                                self.control(CommandType.START)
                                self.current_repeat_time = self.data_cache['current_repeat_time'] + 1
                                now_state = None
                                self.logger.debug(f'CQSIM engine repeat {self.current_repeat_time} times.')
                    prev_state = now_state
        except grpc.FutureCancelledError:
            self.logger.debug('CQSIM engine data callback cancelled.')
        except grpc.RpcError as e:
            self.logger.error(f'CQSIM engine data callback error: {e.args}')

    def logs_callback(self):
        self.logs_responses = self.engine.GetErrorMsg(engine_pb2.CommonRequest())
        try:
            for response in self.logs_responses:
                with self.logs_lock:
                    self.logs_cache.append(response.msg)
        except grpc.FutureCancelledError:
            self.logger.debug('CQSIM engine logs callback cancelled.')
        except grpc.RpcError as e:
            self.logger.error(f'CQSIM engine logs callback error: {e.args}')

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

    def get_configs(self):
        return self.cache

    def set_configs(self, configs):
        proxy_bin = xml.tostring(configs['proxy_xml'], encoding='UTF-8', xml_declaration=True)
        proxy_b64 = base64.b64encode(proxy_bin).decode('utf-8')
        with open(f'{self.cwd}/../zlib.dll', 'rb') as f1, \
             open(f'{self.cwd}/configs.json', 'rb') as f2, \
             open(f'{self.cwd}/sim_term_func.dll', 'rb') as f3:
            requests.put(
                f'http://{self.res_addr}/api/model/{self.proxy_id}',
                headers={'x-token': self.x_token},
                files=[
                    ('configFile', (None, proxy_b64)),
                    ('dependencyFile', ('zlib.dll', f1)),
                    ('dependencyFile', ('configs.json', f2)),
                    ('dependencyFile', ('sim_term_func.dll', f3)),
                ],
            )

        scenario_bin = xml.tostring(configs['scenario_xml'], encoding='UTF-8', xml_declaration=True)
        scenario_b64 = base64.b64encode(scenario_bin).decode('utf-8')
        interaction_bin = xml.tostring(configs['interaction_xml'], encoding='UTF-8', xml_declaration=True)
        interaction_b64 = base64.b64encode(interaction_bin).decode('utf-8')
        requests.put(
            f'http://{self.res_addr}/api/scenario/{self.scenario_id}',
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
        for model_name, model_config in self.data.items():
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

        with open(f'{self.cwd}/configs.json', 'w') as f1, \
             open(f'{self.cwd}/../proxy/serialize.hpp', 'r') as f2, \
             open(f'{self.cwd}/serialize.hpp', 'w') as f3, \
             open(f'{self.cwd}/../proxy/interface.cpp', 'r') as f4, \
             open(f'{self.cwd}/interface.cpp', 'w') as f5, \
             open(f'{self.cwd}/sim_term_func.cpp', 'w') as f6:
            proxy = {
                'types': self.types,
                'data': self.data,
                'routes': self.routes,
                'sim_duration': self.sim_duration,
                'sim_step_ratio': self.sim_step_ratio,
                'simenv_addr': self.simenv_addr,
            }
            json.dump(proxy, f1)
            f3.write(f2.read())
            f5.write(f4.read())
            f6.write(self.sim_term_func)
        cmd = 'x86_64-w64-mingw32-g++ -O1 -static -shared -o sim_term_func.dll -std=c++17 interface.cpp sim_term_func.cpp'
        subprocess.run(cmd, cwd=self.cwd, timeout=10, shell=True, capture_output=True)

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
        for model_name, model_config in self.data.items():
            pub_sub = interaction_xml[2].find(f'./ModelPubSubInfo[@modelID="{model_config["modelid"]}"]')
            if pub_sub is None:
                pub_sub = xml.SubElement(interaction_xml[2], 'ModelPubSubInfo', attrib={'modelID': model_config['modelid']})
                xml.SubElement(pub_sub, 'PublishParams')
                xml.SubElement(pub_sub, 'SubscribeParams')

            if len(model_config['inputs']) > 0:
                topic_name = f'Proxy_{model_name}_Input'
                topic_type = xml.SubElement(
                    interaction_xml[0],
                    'TopicType',
                    attrib={
                        'name': topic_name,
                        'modelID': model_config['modelid'],
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
                            'isTaskFlow': 'false',
                        },
                    )

            if len(model_config['outputs']) > 0:
                topic_name = f'Proxy_{model_name}_Output'
                topic_type = xml.SubElement(
                    interaction_xml[0],
                    'TopicType',
                    attrib={
                        'name': topic_name,
                        'modelID': model_config['modelid'],
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
                            'isTaskFlow': 'false',
                        },
                    )

        return {
            'proxy_xml': proxy_xml,
            'scenario_xml': scenario_xml,
            'interaction_xml': interaction_xml,
        }

    def __del__(self):
        """Close CQSIM engine."""
        self.join_threads()
        self.channel.close()
        super().__del__()
