import base64
import json
import os
import subprocess
import threading
from typing import Any, Dict, List, Literal, Tuple
import xml.etree.ElementTree as xml
import zipfile

import grpc
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2
import requests

from .base import SimEngineBase
from .cqsim_pb import engine_pb2
from .cqsim_pb import engine_pb2_grpc


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
        self.need_repeat = False

        self.data_thread = None
        self.data_responses = None
        self.data_lock = threading.Lock()
        self.data_cache = {}

        self.logs_thread = None
        self.logs_responses = None
        self.logs_lock = threading.Lock()
        self.logs_cache = []

    def control(
        self,
        cmd: Literal['init', 'start', 'pause', 'step', 'resume', 'stop', 'episode', 'param'],
        params: Dict[str, Any],
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

            if self.need_repeat:
                self.current_repeat_time += 1
                self.need_repeat = False
            else:
                self.current_repeat_time = 1
                if params['changed']:
                    self.update_resource()
                self.join_threads()
                self.init_threads()

            p = params['task']
            if 'exp_design_id' in p:
                sample = engine_pb2.InitInfo.MultiSample(exp_design_id=p['exp_design_id'])
                self.engine.Init(engine_pb2.InitInfo(multi_sample_config=sample))
            else:
                sample = engine_pb2.InitInfo.OneSample(task_id=p['task_id'])
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
            data['current_repeat_time'] = self.current_repeat_time
        with self.logs_lock:
            logs = self.logs_cache.copy()
            self.logs_cache.clear()
        return data, logs

    def close(self):
        """Close CQSim engine.

        Returns:
            True if success.
        """
        self.join_threads()
        self.channel.close()
        return True

    def data_callback(self):
        self.data_responses = self.engine.GetSysInfo(engine_pb2.CommonRequest())
        try:
            for response in self.data_responses:
                with self.data_lock:
                    self.data_cache['sim_current_time'] = response.sim_current_time.ToSeconds()
                    self.data_cache['sim_duration'] = response.sim_duration.ToSeconds()
                    self.data_cache['real_duration'] = response.real_duration.ToSeconds()
                    self.data_cache['sim_time_step'] = response.sim_time_step
                    self.data_cache['speed_ratio'] = response.speed_ratio
                    self.data_cache['real_speed_ratio'] = response.real_speed_ratio
                    self.data_cache['current_sample_id'] = response.current_sample_id

                p = self.sim_params['task']
                if not ('exp_design_id' in p and response.current_sample_id != p['exp_sample_num'] - 1) and \
                        response.node_state[0].state == engine_pb2.EngineNodeState.State.STOPPED and \
                        self.current_repeat_time < p['repeat_times']:
                    self.need_repeat = True
                    self.control('stop', {})
                    self.control('init', p)
                    self.control('start', {})
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

    def init_threads(self):
        self.data_thread = threading.Thread(name='data_thread', target=self.data_callback)
        self.data_thread.daemon = True
        self.data_thread.start()
        self.logs_thread = threading.Thread(name='logs_thread', target=self.logs_callback)
        self.logs_thread.daemon = True
        self.logs_thread.start()

    def join_threads(self):
        if self.data_thread is not None:
            self.data_responses.cancel()
            self.data_thread.join(1)
        if self.logs_thread is not None:
            self.logs_responses.cancel()
            self.logs_thread.join(1)

    def update_resource(self):
        cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp/cqsim')
        os.makedirs(cwd, exist_ok=True)

        r = requests.post(
            f'http://{self.res_addr}/api/model/unpack',
            headers={'x-token': self.x_token},
            data={
                'ids': [self.proxy_id],
                'types': [1]
            },
        )
        msg = r.json()

        cfg_b64 = msg['data'][0]['configFile']
        cfg_bin = base64.b64decode(cfg_b64.encode('utf-8'))
        cfg_str = cfg_bin.decode('utf-8')
        cfg_xml = xml.fromstring(cfg_str)
        for el in cfg_xml[0].findall('./Parameter[@proxy="true"]'):
            cfg_xml[0].remove(el)
        proxy_name = cfg_xml.get('displayName', '代理')
        cfg_xml[0].find('./Parameter[@name="InstanceName"]').set('value', proxy_name)
        cfg_xml[0].find('./Parameter[@name="ForceSideID"]').set('value', '80')
        cfg_xml[0].find('./Parameter[@name="ID"]').set('value', '8080')
        for model_name, model_config in self.sim_params['proxy']['data'].items():
            for input_name, input_config in model_config['inputs'].items():
                param_name = f'{model_name}_{input_name}'
                xml.SubElement(
                    cfg_xml[0],
                    'Parameter',
                    attrib={
                        'name': param_name,
                        'type': input_config['type'],
                        'displayName': param_name,
                        'usage': 'input',
                        'value': json.dumps(input_config['value']),
                        'unit': '',
                        'proxy': 'true',
                    },
                )
            for output_name, output_config in model_config['outputs'].items():
                xml.SubElement(
                    cfg_xml[0],
                    'Parameter',
                    attrib={
                        'name': param_name,
                        'type': output_config['type'],
                        'displayName': param_name,
                        'usage': 'output',
                        'value': '',
                        'unit': '',
                        'proxy': 'true',
                    },
                )
        cfg_str = xml.tostring(cfg_xml, encoding='unicode')
        cfg_bin = cfg_str.encode('utf-8')
        cfg_b64 = base64.b64encode(cfg_bin).decode('utf-8')

        with open(f'{cwd}/configs.json', 'w') as f1, open(f'{cwd}/sim_term_func.cc', 'w') as f2:
            json.dump(self.sim_params, f1)
            f2.write(self.sim_params['proxy']['sim_term_func'])
        cmd = 'cl /LD /std:c++17 sim_term_func.cc'
        subprocess.run(cmd, cwd=cwd, timeout=10, shell=True, capture_output=True)
        with zipfile.ZipFile(f'{cwd}/dependency.zip', 'w') as f:
            f.write(f'{cwd}/configs.json', arcname='configs.json')
            f.write(f'{cwd}/sim_term_func.dll', arcname='sim_term_func.dll')

        requests.put(
            f'http://{self.res_addr}/api/model/{self.proxy_id}',
            headers={'x-token': self.x_token},
            data={'configFile': cfg_b64},
            files={'dependencyFile': open(f'{cwd}/dependency.zip', 'rb')},
        )

        scenario_id = self.sim_params['task']['task_id']
        r = requests.post(
            f'http://{self.res_addr}/api/scenario/unpack',
            headers={'x-token': self.x_token},
            data={
                'id': scenario_id,
                'types': [1, 2],
            },
        )
        msg = r.json()

        scenario_b64 = msg['data']['scenarioFile']
        scenario_bin = base64.b64decode(scenario_b64.encode('utf-8'))
        scenario_str = scenario_bin.decode('utf-8')
        scenario_xml = xml.fromstring(scenario_str)
        proxy_side = scenario_xml[2].find('./ForceSide[@proxy="true"]')
        if proxy_side is None:
            proxy_side = xml.fromstring('''
            <ForceSide id="80" name="代理" color="#FFD700" proxy="true">
                <Units>
                    <Unit id="8080"/>
                </Units>
            </ForceSide>''')
            scenario_xml[2].append(proxy_side)
        proxy_entity = scenario_xml[3].find('./Entity[@proxy="true"]')
        if proxy_entity is None:
            proxy_entity = xml.SubElement(
                scenario_xml[3],
                'Entity',
                attrib={
                    'id': '8080',
                    'modelID': self.proxy_id,
                    'entityName': proxy_name,
                    'modelDisplayName': proxy_name,
                    'proxy': 'true',
                },
            )
        else:
            proxy_entity.clear()
        proxy_entity.append(cfg_xml[0])
        scenario_str = xml.tostring(scenario_xml, encoding='unicode')
        scenario_bin = scenario_str.encode('utf-8')
        scenario_b64 = base64.b64encode(scenario_bin).decode('utf-8')

        interaction_b64 = msg['data']['interactionFile']
        interaction_bin = base64.b64decode(interaction_b64.encode('utf-8'))
        interaction_str = interaction_bin.decode('utf-8')
        interaction_xml = xml.fromstring(interaction_str)
        for el in interaction_xml[0].findall('./TopicType[@proxy="true"]'):
            interaction_xml[0].remove(el)
        for el in interaction_xml[1].findall('./Topic[@proxy="true"]'):
            interaction_xml[1].remove(el)
        for pub_sub in interaction_xml[3]:
            for el in pub_sub[0].findall('./PublishParam[@proxy="true"]'):
                pub_sub[0].remove(el)
            for el in pub_sub[1].findall('./SubscribeParam[@proxy="true"]'):
                pub_sub[1].remove(el)
        proxy_pubsub = interaction_xml[3].find(f'./ModelPubSubInfo[@modelID="{self.proxy_id}"]')
        if proxy_pubsub is None:
            proxy_pubsub = xml.SubElement(interaction_xml[3], 'ModelPubSubInfo', attrib={'modelID': self.proxy_id})
            xml.SubElement(proxy_pubsub, 'PublishParams')
            xml.SubElement(proxy_pubsub, 'SubscribeParams')
        for model_name, model_config in self.sim_params['proxy']['data'].items():
            pub_sub = interaction_xml[3].find(f'./ModelPubSubInfo[@modelID="{model_config["model_id"]}"]')
            if pub_sub is None:
                pub_sub = xml.SubElement(interaction_xml[3], 'ModelPubSubInfo', attrib={'modelID': model_config['model_id']})
                xml.SubElement(pub_sub, 'PublishParams')
                xml.SubElement(pub_sub, 'SubscribeParams')

            topic_name = f'ProxyInput_{model_name}'
            topic_type = xml.SubElement(
                interaction_xml[0],
                'TopicType',
                attrib={
                    'name': topic_name,
                    'modelID': model_config['model_id'],
                    'isTaskFlow': 'false',
                    'proxy': 'true',
                },
            )
            topic_params = xml.SubElement(topic_type, 'Params')
            for input_name, input_config in model_config['inputs'].items():
                xml.SubElement(topic_params, 'Param', attrib={'name': input_name, 'type': input_config['type']})
                xml.SubElement(
                    proxy_pubsub[0],
                    'PublishParam',
                    attrib={
                        'topicName': topic_name,
                        'topicParamName': input_name,
                        'modelParamName': f'{model_name}_{input_name}',
                        'proxy': 'true',
                    },
                )
                xml.SubElement(
                    pub_sub[1],
                    'SubscribeParam',
                    attrib={
                        'topicName': topic_name,
                        'topicParamName': input_name,
                        'modelParamName': input_name,
                        'proxy': 'true',
                    },
                )
            xml.SubElement(interaction_xml[1], 'Topic', attrib={'name': topic_name, 'type': topic_name})

            topic_name = f'ProxyOutput_{model_name}'
            topic_type = xml.SubElement(
                interaction_xml[0],
                'TopicType',
                attrib={
                    'name': topic_name,
                    'modelID': model_config['model_id'],
                    'isTaskFlow': 'false',
                    'proxy': 'true',
                },
            )
            topic_params = xml.SubElement(topic_type, 'Params')
            for output_name, output_config in model_config['outputs'].items():
                xml.SubElement(topic_params, 'Param', attrib={'name': output_name, 'type': output_config['type']})
                xml.SubElement(
                    proxy_pubsub[1],
                    'SubscribeParam',
                    attrib={
                        'topicName': topic_name,
                        'topicParamName': output_name,
                        'modelParamName': f'{model_name}_{output_name}',
                        'proxy': 'true',
                    },
                )
                xml.SubElement(
                    pub_sub[0],
                    'PublishParams',
                    attrib={
                        'topicName': topic_name,
                        'topicParamName': output_name,
                        'modelParamName': output_name,
                        'proxy': 'true',
                    },
                )
            xml.SubElement(interaction_xml[1], 'Topic', attrib={'name': topic_name, 'type': topic_name})
        interaction_str = xml.tostring(interaction_xml, encoding='unicode')
        interaction_bin = interaction_str.encode('utf-8')
        interaction_b64 = base64.b64encode(interaction_bin).decode('utf-8')

        requests.put(
            f'http://{self.res_addr}/api/scenario/{scenario_id}',
            headers={'x-token': self.x_token},
            data={
                'scenarioFile': scenario_b64,
                'interactionFile': interaction_b64,
            },
        )
