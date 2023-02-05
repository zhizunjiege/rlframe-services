import base64
import json
import os
import subprocess
import threading
import time
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
                self.need_repeat = False
            else:
                self.current_repeat_time = 1
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

                now_state = response.node_state[0].state
                p = self.sim_params['task']
                if not ('exp_design_id' in p and response.current_sample_id != p['exp_sample_num'] - 1) and \
                        self.current_repeat_time < p['repeat_times'] and \
                        now_state == engine_pb2.EngineNodeState.State.STOPPED and \
                        now_state != prev_state and prev_state is not None:
                    self.need_repeat = True
                    self.control('stop', {})
                    time.sleep(0.5)
                    self.control('init', self.sim_params)
                    time.sleep(0.5)
                    self.control('start', {})
                    self.current_repeat_time += 1
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
            json={
                'ids': [self.proxy_id],
                'types': [1]
            },
        )
        msg = r.json()

        cfg_b64 = msg['data'][0]['configFile']
        cfg_str = base64.b64decode(cfg_b64.encode('utf-8')).decode('utf-8')
        cfg_xml = xml.fromstring(cfg_str)
        for el in cfg_xml[0].findall('./Parameter[@unit="proxy"]'):
            cfg_xml[0].remove(el)
        proxy_name = cfg_xml.get('displayName', '代理')
        cfg_xml[0].find('./Parameter[@name="InstanceName"]').set('value', proxy_name)
        cfg_xml[0].find('./Parameter[@name="ForceSideID"]').set('value', '80')
        cfg_xml[0].find('./Parameter[@name="ID"]').set('value', '8080')
        for model_name, model_config in self.sim_params['proxy']['data'].items():
            for inouts_tuple in [
                (model_config['inputs'], 'input', 'output'),
                (model_config['outputs'], 'output', 'input'),
            ]:
                for inouts_name, inouts_config in inouts_tuple[0].items():
                    param_name = f'{model_name}_{inouts_tuple[1]}_{inouts_name}'
                    param_type = inouts_config['type'] if isinstance(inouts_config, dict) else inouts_config
                    xml.SubElement(
                        cfg_xml[0],
                        'Parameter',
                        attrib={
                            'name': param_name,
                            'type': param_type,
                            'displayName': param_name,
                            'usage': inouts_tuple[2],
                            'value': '',
                            'unit': 'proxy',
                        },
                    )
        cfg_bin = xml.tostring(cfg_xml, encoding='UTF-8', xml_declaration=True)
        cfg_b64 = base64.b64encode(cfg_bin).decode('utf-8')

        with open(f'{cwd}/configs.json', 'w') as f1, open(f'{cwd}/sim_term_func.cc', 'w') as f2:
            json.dump(self.sim_params, f1)
            f2.write(self.sim_params['proxy']['sim_term_func'])
        cmd = 'g++ -shared -o sim_term_func.dll -std=c++17 sim_term_func.cc'
        subprocess.run(cmd, cwd=cwd, timeout=10, shell=True, capture_output=True)
        with zipfile.ZipFile(f'{cwd}/dependency.zip', 'w') as f:
            f.write(f'{cwd}/configs.json', arcname='configs.json')
            f.write(f'{cwd}/sim_term_func.dll', arcname='sim_term_func.dll')

        with open(f'{cwd}/dependency.zip', 'rb') as fb:
            requests.put(
                f'http://{self.res_addr}/api/model/{self.proxy_id}',
                headers={'x-token': self.x_token},
                files={
                    'configFile': (None, cfg_b64),
                    'dependencyFile': ('dependency.zip', fb),
                },
            )

        scenario_id = self.sim_params['task']['task_id']
        r = requests.post(
            f'http://{self.res_addr}/api/scenario/unpack',
            headers={'x-token': self.x_token},
            json={
                'id': scenario_id,
                'types': [1, 2],
            },
        )
        msg = r.json()

        scenario_b64 = msg['data']['scenarioFile']
        scenario_str = base64.b64decode(scenario_b64.encode('utf-8')).decode('utf-8')
        scenario_xml = xml.fromstring(scenario_str)
        proxy_side = scenario_xml[2].find('./ForceSide[@id="80"]')
        if proxy_side is not None:
            scenario_xml[2].remove(proxy_side)
        proxy_side = xml.fromstring('''
        <ForceSide id="80" name="代理" color="#FFD700">
            <Units>
                <Unit id="8080"/>
            </Units>
        </ForceSide>''')
        scenario_xml[2].append(proxy_side)
        proxy_entity = scenario_xml[3].find('./Entity[@id="8080"]')
        if proxy_entity is not None:
            scenario_xml[3].remove(proxy_entity)
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
        proxy_entity.append(cfg_xml[0])
        scenario_bin = xml.tostring(scenario_xml, encoding='UTF-8', xml_declaration=True)
        scenario_b64 = base64.b64encode(scenario_bin).decode('utf-8')

        interaction_b64 = msg['data']['interactionFile']
        interaction_str = base64.b64decode(interaction_b64.encode('utf-8')).decode('utf-8')
        interaction_xml = xml.fromstring(interaction_str)

        for node in interaction_xml[0:2]:
            for topic in node.findall('*'):
                name = topic.get('name')
                if name.find('Proxy') != -1:
                    node.remove(topic)
        for pub_sub in interaction_xml[2]:
            for node in pub_sub:
                for param in node.findall('*'):
                    name = param.get('topicName')
                    if name.find('Proxy') != -1:
                        node.remove(param)
        proxy_pubsub = interaction_xml[2].find(f'./ModelPubSubInfo[@modelID="{self.proxy_id}"]')
        if proxy_pubsub is None:
            proxy_pubsub = xml.SubElement(interaction_xml[2], 'ModelPubSubInfo', attrib={'modelID': self.proxy_id})
            xml.SubElement(proxy_pubsub, 'PublishParams')
            xml.SubElement(proxy_pubsub, 'SubscribeParams')
        for model_name, model_config in self.sim_params['proxy']['data'].items():
            pub_sub = interaction_xml[2].find(f'./ModelPubSubInfo[@modelID="{model_config["id"]}"]')
            if pub_sub is None:
                pub_sub = xml.SubElement(interaction_xml[2], 'ModelPubSubInfo', attrib={'modelID': model_config['id']})
                xml.SubElement(pub_sub, 'PublishParams')
                xml.SubElement(pub_sub, 'SubscribeParams')

            for inouts_tuple in [
                (model_config['inputs'], 'input', 0, 'PublishParam', 1, 'SubscribeParam'),
                (model_config['outputs'], 'output', 1, 'SubscribeParam', 0, 'PublishParam'),
            ]:
                if len(inouts_tuple[0]) > 0:
                    topic_name = f'Proxy_{model_name}_{inouts_tuple[1].capitalize()}'
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
                    for inout_name, inout_config in inouts_tuple[0].items():
                        xml.SubElement(
                            topic_params,
                            'Param',
                            attrib={
                                'name': inout_name,
                                'type': inout_config['type'] if isinstance(inout_config, dict) else inout_config,
                            },
                        )
                        for sub_tuple in [
                            (proxy_pubsub[inouts_tuple[2]], inouts_tuple[3], f'{model_name}_{inouts_tuple[1]}_{inout_name}'),
                            (pub_sub[inouts_tuple[4]], inouts_tuple[5], inout_name),
                        ]:
                            xml.SubElement(
                                sub_tuple[0],
                                sub_tuple[1],
                                attrib={
                                    'topicName': topic_name,
                                    'topicParamName': inout_name,
                                    'modelParamName': sub_tuple[2],
                                },
                            )
        interaction_bin = xml.tostring(interaction_xml, encoding='UTF-8', xml_declaration=True)
        interaction_b64 = base64.b64encode(interaction_bin).decode('utf-8')

        requests.put(
            f'http://{self.res_addr}/api/scenario/{scenario_id}',
            headers={'x-token': self.x_token},
            json={
                'scenarioFile': scenario_b64,
                'interactionFile': interaction_b64,
            },
        )
