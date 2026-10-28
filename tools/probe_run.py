import sys,os
sys.path.insert(0,os.getcwd())
from src.validator import BaseValidator
v=BaseValidator(myagent_id='0',myagent_module_path='agents/Probe',opponent_id='1',opponent_module_path='Agents/BenchMark',common_dir='common',youth=False)
v.validate(result_dir='./results',result_path='probe.json',replay=0,num_validation=1,time_out=3.0,memory_limit=7.5,random_test=0,control_init='random',make_log=0,visualize=0,color='Red',max_size=0.08)
