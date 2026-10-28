import random
import os
import time
import importlib
import copy
from collections import defaultdict
import traceback
import numpy as np
from timeout_decorator import timeout, TimeoutError

from ASRCAISim1.core import Factory, nljson, Fighter, Entity # type: ignore
from ASRCAISim1.common import addPythonClass
from ASRCAISim1.GymManager import GymManager
from ASRCAISim1.plugins.rayUtility.utility.TransparentRemoteObject import unwrap_if_needed # type: ignore
from ASRCAISim1.plugins.MatchMaker.TwoTeamCombatMatchMaker import ( # type: ignore
    getFromDesiredInitialState,
    getRandomizedInitialState,
    invert,
)
from r8_contest_models.R8ContestTwoTeamCombatMatchMaker import (
    getForEscort,
    R8ContestTwoTeamCombatMatchMaker
)

class Fight():
    def __init__(self, match_config: dict[str, dict]) -> None:
        seed=match_config.get("seed",None)
        if(seed is None):
            import numpy as np
            seed = np.random.randint(2**31)
        logName="_vs_".join([info.get("logName",info["userModelID"]) for info in match_config["teams"].values()])

        # ユーザーモジュールの読み込み
        importedUserModules={}
        for team,info in match_config["teams"].items():
            userModelID=info["userModelID"]
            userModuleID=info["userModuleID"]
            args=info.get("args",{})
            args["userModelID"]=userModelID
            info["args"]=args
            if(not userModuleID in importedUserModules):
                try:
                    module = importlib.import_module(userModuleID)
                    assert hasattr(module, "getUserAgentClass")
                    assert hasattr(module, "getUserAgentModelConfig")
                    assert hasattr(module, "isUserAgentSingleAsset")
                    assert hasattr(module, "getUserPolicy")
                except Exception as e:
                    raise e
                importedUserModules[userModuleID]=module
            module=importedUserModules[userModuleID]
            agentClass = module.getUserAgentClass(args)
            addPythonClass("Agent", "Agent_"+userModelID, agentClass)
            Factory.addDefaultModel(
                "Agent",
                "Agent_"+userModelID,
                {
                    "class": "Agent_"+userModelID,
                    "config": module.getUserAgentModelConfig(args)
                }
            )
        self.importedUserModules = importedUserModules

        # コンフィグの生成
        loggerConfig = {}
        if "log_dir" in match_config:
            # loggerConfig = {
            #     "MultiEpisodeLogger":{
            #         "class":"MultiEpisodeLogger",
            #         "config":{
            #             "prefix":os.path.join(match_config["log_dir"],logName), # type: ignore
            #             "episodeInterval":1,
            #             "ratingDenominator":100
            #         }
            #     }
            # }
            if match_config["replay"]:
                loggerConfig["GUIStateLogger"]={
                    "class":"GUIStateLogger",
                    "config":{
                        "prefix":os.path.join(match_config["log_dir"],logName), # type: ignore
                        "episodeInterval":1,
                        "innerInterval":1
                    }
                }

        division="youth" if match_config["youth"] else "open"
        configs = [
            os.path.join(match_config["common_dir"], "R8_contest_"+division+"_mission_config.json"), # type: ignore
            os.path.join(match_config["common_dir"], "R8_contest_"+division+"_asset_placeholder.json"), # type: ignore
            {
                "Manager": {
                    "Rewards": [],
                    "seed":seed,
                    "ViewerType":"R8Contest"+("Youth" if match_config["youth"] else "Open")+"God" if match_config["visualize"] else "None",
                    "Loggers": loggerConfig,
                    "AgentConfigDispatcher": {}
                }
            }
        ]
        context = {
            "config": configs,
            "worker_index": 0,
            "vector_index": 0
        }

        self.context = context

        # 環境の生成
        self.env = wrapEnvForTwoTeamCombatMatchMaking(GymManager)(context)

        # MatchMakerの生成
        teams = []
        for team,info in match_config["teams"].items():
            teams.append(team)
            info["isSingle"]=importedUserModules[info["userModuleID"]].isUserAgentSingleAsset(info["args"])

        match_config["asset_slots"] = ["Fighter1", "Fighter2", "Fighter3", "Fighter4", "Escorted1"]
        match_config["initial_states"] = [
            {
                "number": 4,
                "type": "normal",
                "asset_slots": ["Fighter1", "Fighter2", "Fighter3", "Fighter4"],
                "symmetric": True,
                "lower": {
                    "pos": [-20000.0,95000.0,-12000.],
                    "vel": 260.0,
                    "heading": 225.0
                },
                "upper": {
                    "pos": [20000.0,105000.0,-6000.0],
                    "vel": 280.0,
                    "heading": 315.0
                }
            },
            {
                "number": 1,
                "type": "escorted",
                "asset_slots": ["Escorted1"],
                "symmetric": False,
                "p1": [100000.0,120000.0,-10000.0],
                "p2": [-100000.0,120000.0,-10000.0],
                "radius": 5000.0,
                "vel": 250.0 # 護衛対象機Agentの目標速度で上書きされるので指定無効
            }
        ]
        if not match_config["youth"]:
            # オープン部門は性能をランダム化
            match_config["asset_spec_randomization"] = {
                "fighter": {
                    "number": 4,
                    "type": "Fighter",
                    "asset_slots": ["Fighter1", "Fighter2", "Fighter3", "Fighter4"],
                    "symmetric": True,
                    "heterogeneous": True,
                    "shuffle": True,
                    "randomized_asset_spec": {
                        # ランダムにしない値はコメントアウトすること。
                        "rcs_scale": [0.4,1.0], # 探知距離にして約0.8〜1.0倍
                        "radar_range": [0.8,1.2],
                        "radar_coverage": [60.0,120.0],
                        "ir_radiation_intensity_scale": [0.64,1.0], # 探知距離にして約0.8〜1.0倍
                        "irst_range": [0.8,1.2],
                        "irst_coverage": [60.0,120.0],
                        "maximum_speed": [0.8,1.2],
                        "num_missiles": [4,8],
                        "missile_thrust": [0.8,1.2],
                        "shot_approval_delay": [1,5]
                    }
                },
                "escorted": {
                    "number": 1,
                    "type": "Escorted",
                    "asset_slots": ["Escorted1"],
                    "symmetric": True,
                    "heterogeneous": True,
                    "shuffle": True,
                    "randomized_asset_spec": {
                        # "rcs_scale": [0.4,1.0], # 探知距離にして約1.6〜2.0倍
                        # "ir_radiation_intensity_scale": [0.64,1.0], # 探知距離にして約1.6〜2.0倍
                        # "maximum_speed": [0.9,1.1],
                        # "velocity": 250.0
                    }
                },
                "escorted_agent": {
                    "number": 1,
                    "type": "EscortedAgent",
                    "asset_slots": ["Escorted1"], # escortedのasset_slotsと一致させること。
                    "symmetric": False,
                    "heterogeneous": True,
                    "shuffle": True,
                    "randomized_asset_spec": {
                        "velocity": [225.0, 275.0], #目標速度 = 初期速度として設定
                        "isClockwise": { # 周回方向をそれぞれの確率で指定
                            "right": 0.5,
                            "left": 0.5
                        }
                    }
                }
            }
        self.match_config = match_config
        self.matchMaker = R8ContestMatchMakerForEvaluation({"match_config":match_config})
        self.reverse = False
        self.reverse_map = {
            teams[0]: teams[1],
            teams[1]: teams[0]
        }


    def make_matchInfo(self) -> None:
        self.matchInfo=self.matchMaker.makeNextMatch("",0)


    def swap_condition(self) -> None:
        print('swap condition')
        teams=list(self.matchInfo["teams"].keys())
        tmp=self.matchInfo["teams"][teams[0]]["Agent"]
        self.matchInfo["teams"][teams[0]]["Agent"]=self.matchInfo["teams"][teams[1]]["Agent"]
        self.matchInfo["teams"][teams[1]]["Agent"]=tmp

        self.reverse = True


    def set_match(self) -> None:
        self.env.setMatch(self.matchInfo)


    def get_ticks(self, num_assets):
        ticks = {}
        red_fighter_names = [a().getFullName() for a in self.env.manager.getAgents() if 'Red' in a().getFullName()]
        blue_fighter_names = [a().getFullName() for a in self.env.manager.getAgents() if 'Blue' in a().getFullName()]
        ticks['Red'] = self.get_stepcount(red_fighter_names, num_assets)
        ticks['Blue'] = self.get_stepcount(blue_fighter_names, num_assets)

        return ticks


    def get_stepcount(self, fighter_names, num_assets):
        tick = {}
        if len(fighter_names)>1:
            for fighter_name in fighter_names:
                agentName, modelName, policyName = fighter_name.split(":")
                team = agentName.split('_')[0]
                port = agentName.split('_')[-1]
                tick[int(port)] = self.env.manager.getAgentStepCount(fighter_name)
        elif len(fighter_names) == 1:
            for i in range(num_assets):
                tick[i+1] = self.env.manager.getAgentStepCount(fighter_names[0])
        return tick


    def get_data(self, data, fighter_names):
        ptime = self.env.manager.getElapsedTime() #self.env.manager.getTime().__str__()
        data_f = {}
        i = 1
        for k, v in data.items():
            if k in fighter_names:
                data_f[i] = v
                i+=1
        out = {'ptime':ptime, 'data':data_f}
        n_data = len(data_f)
        return out, n_data


    def update_logs(self, red_fighter_names, blue_fighter_names, logs, obs, infos, red_actions, blue_actions):
        def conv(src):
            return map_r(src, lambda d: d.tolist() if isinstance(d, np.ndarray) else d)

        a, n_data = self.get_data(conv(red_actions), red_fighter_names)
        if n_data:
            logs['Red']['actions'].append(a)
        o, n_data = self.get_data(conv(obs), red_fighter_names)
        if n_data:
            logs['Red']['observations'].append(o)
        info, n_data = self.get_data(infos, red_fighter_names)
        if n_data:
            logs['Red']['infos'].append(info)

        a, n_data = self.get_data(conv(blue_actions), blue_fighter_names)
        if n_data:
            logs['Blue']['actions'].append(a)
        o, n_data = self.get_data(conv(obs), blue_fighter_names)
        if n_data:
            logs['Blue']['observations'].append(o)
        info, n_data = self.get_data(infos, blue_fighter_names)
        if n_data:
            logs['Blue']['infos'].append(info)


    def run(self, random_test: bool = False) -> tuple:
        if random_test:
            return random.choice(['Red', 'Blue', 'Draw']), {}

        # StandalonePolicyの生成
        policies = {
            "Policy_"+info["userModelID"]: self.importedUserModules[info["userModuleID"]].getUserPolicy(info["args"])
            for team,info in self.match_config["teams"].items()
        }

        # 生成状況の確認
        obs, info = self.env.reset() # 確認前にresetでエピソードを初期化しておく必要あり
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        print("=====Agent classes=====")
        for team,info in self.match_config["teams"].items():
            print("Agent_"+info["userModelID"], " = ", self.importedUserModules[info["userModuleID"]].getUserAgentClass(info["args"]))
        print("=====Policies=====")
        for name, policy in policies.items():
            print(name, " = ", type(policy))
        print("=====Policy Map (at reset)=====")
        for fullName in action_space:
            print(fullName, " -> ", policyMapper(fullName))
        print("=====Agent to Asset map=====")
        for agent in [a() for a in self.env.manager.getAgents()]:
            print(agent.getFullName(), " -> ", "{")
            for port, parent in agent.parents.items():
                print("  ", port, " : ", parent.getFullName())
            print("}")

        rewards = {k: 0.0 for k in obs.keys()}
        terminateds = {k: False for k in obs.keys()}
        truncateds = {k: False for k in obs.keys()}
        infos = {k: None for k in obs.keys()}
        for p in policies.values():
            p.reset()

        # 対戦スコア計算用の変数
        userModelIDs = {
            team: teamInfo["userModelID"]
            for team, teamInfo in self.match_config["teams"].items()
            }
        teams = ["Blue", "Red"]
        opponents = {teams[0]: teams[1], teams[1]: teams[0]}

        # 各戦闘（エピソード）の勝利時に、以下の要領で戦闘スコアを計算する。
        # 所定のエピソード数が終了した時点の戦闘スコアの合計値を対戦スコアとし、対戦スコアの大きい方を対戦の勝者として扱う。
        # 
        # - 失格による決着の場合、失格となっていない陣営を勝者としてに護衛対象機撃破時と同じ得点計算を行う。双方失格の場合は引き分けとして+0.5点。
        # - 護衛対象機の撃破（撃墜又は相手の自滅）による決着の場合、先に撃破した陣営を勝者として0.6 + 0.3 * (命中弾数/射撃弾数) + 0.1 * min(1, max(0, 2-経過秒数/900))点を与え、敗者の得点は1-(勝者の得点)とする。同時撃破の場合は引き分けとして双方に+0.5点。

        # 各陣営の初期弾数、初期機数の計算
        initialNumMissiles = defaultdict(lambda: 0)

        for team in teams:
            for wFighter in self.env.manager.getAssets(lambda a:a.getTeam()==team and isinstance(a,Fighter)):
                # 戦闘機も護衛対象機もFighterクラス
                fighter = wFighter() # weakref(weak_ptr)から強参照に変更
                initialNumMissiles[team] += fighter.numMsls

        # 各陣営の命中数のカウント
        hitCounts = defaultdict(lambda: 0)

        events = []
        godlog = {'missiles': {}, 'uid2name': {}}
        def onHit(args: nljson):
            # 命中数を数えるためのイベントハンドラ
            # argsは{"wpn":wpn,"tgt":tgt}
            wpn = Entity.from_json(args["wpn"])
            hitCounts[wpn.getTeam()] += 1
            try:
                tgt = Entity.from_json(args["tgt"])
                events.append((self.env.manager.getElapsedTime(), 'hit', wpn.getFullName(), tgt.getFullName()))
            except Exception:
                pass
        def onCrash(args):
            try:
                a = Entity.from_json(args)
                events.append((self.env.manager.getElapsedTime(), 'crash', '', a.getFullName()))
            except Exception:
                events.append((self.env.manager.getElapsedTime(), 'crash', '', str(args)[:80]))

        # イベントハンドラはSimulationManagerAccessorForCallbackからしか追加できないのでRuler経由で取得して追加する。
        managerAccessorForCB=self.env.manager.getRuler()().manager
        managerAccessorForCB.addEventHandler("Hit",onHit)
        try:
            managerAccessorForCB.addEventHandler("Crash",onCrash)
        except Exception:
            pass

        # シミュレーションの実行
        start_time = time.time()
        terminateds["__all__"]=False
        truncateds["__all__"]=False
        red_error = False
        blue_error = False
        error = False
        red_error_message = ''
        blue_error_message = ''
        logs = {}
        red_fighter_names = sorted([a().getFullName() for a in self.env.manager.getAgents() if 'Red' in a().getFullName()])
        blue_fighter_names = sorted([a().getFullName() for a in self.env.manager.getAgents() if 'Blue' in a().getFullName()])
        if self.match_config['make_log']:
            print('Making logs')
            logs = {'Red':{'observations': [], 'actions':[], 'infos': []}, 'Blue':{'observations': [], 'actions': [], 'infos': []}}
        time_out = self.match_config['time_out']
        step_count = 0
        while not (terminateds["__all__"] or truncateds["__all__"]):
            observation_space = self.env.get_observation_space()
            action_space = self.env.get_action_space()

            # get Red action
            red_actions = {}
            try:
                red_actions = step_foward(func = wrap_get_action_from_obs, args=(policies, rewards, terminateds, truncateds, infos, observation_space, action_space, policyMapper, obs, 'Red'), time_out=time_out)
            except Exception as e:
                print('Catched error in Red. {}'.format(e))
                red_error_message = traceback.format_exc()
                print(red_error_message)
                red_error = True

            # get Blue action
            blue_actions = {}
            try:
                blue_actions = step_foward(func = wrap_get_action_from_obs, args=(policies, rewards, terminateds, truncateds, infos, observation_space, action_space, policyMapper, obs, 'Blue'), time_out=time_out)
            except Exception as e:
                print('Catched error in Blue. {}'.format(e))
                blue_error_message = traceback.format_exc()
                print(blue_error_message)
                blue_error = True

            if red_error or blue_error:
                break
            if self.match_config['make_log']:
                self.update_logs(red_fighter_names, blue_fighter_names, logs, obs, infos, red_actions, blue_actions)
            
            actions = dict(red_actions, **blue_actions)

            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            if self.match_config.get('god_log'):
                _god_step(self.env.manager, godlog)

            step_count += 1

        finishtime = time.time() - start_time
        print('Time elapsed: {}[s]'.format(finishtime))

        # 戦闘スコアを計算
        remainedNumMissiles = defaultdict(lambda: 0)
        launchedMissiles = defaultdict(lambda: 0) # 射撃弾数
        episodeScores ={team: 0.0 for team in teams}
        for team in teams:
            for wFighter in self.env.manager.getAssets(lambda a:a.getTeam()==team and isinstance(a,Fighter)):
                # 戦闘機も護衛対象機もFighterクラス
                fighter = wFighter() # weakref(weak_ptr)から強参照に変更
                remainedNumMissiles[team] += fighter.remMsls

        for team in teams:
            launchedMissiles[team] = initialNumMissiles[team] - remainedNumMissiles[team]

        ruler = self.env.manager.getRuler()()
        endReason = ruler.getEndReason()
        spentTime = self.env.manager.getElapsedTime()

        if red_error or blue_error:
            winner = 'nocontest'
            loser = 'nocontest'
            error = True
        else:
            if self.env.manager.getRuler()().winner in ['Red', 'Blue']:
                winner = self.env.manager.getRuler()().winner
                loser = opponents[winner]
            else:
                winner = 'Draw'
                loser = 'Draw'
        if not error:
            if endReason == 'PENALTY':
                # 失格で決着した場合
                if winner == 'Draw':
                    # 双方失格の場合は引き分けなのでどちらも+0.5点
                    for team in teams:
                        episodeScores[team] += 0.5
                else:
                    # 一方だけ失格の場合は、失格になっていない方に護衛対象機撃破時と同等の対戦スコアを加算し、失格者には（1-勝者の得点）を加算
                    winnerScore = (
                        0.6
                        + 0.3 * (hitCounts[winner] / launchedMissiles[winner] if launchedMissiles[winner] > 0 else 0.0)
                        + 0.1 * (1 if spentTime < 900.0 else max(0, 2.0-spentTime/900.0))
                    )
                    episodeScores[winner] += winnerScore
                    episodeScores[loser] += 1.0 - winnerScore

            elif endReason == 'ELIMINATION':
                # 護衛対象機の撃破で決着した場合
                if winner == 'Draw':
                    # 引き分けの場合は双方に+0.5点加算
                    for team in teams:
                        episodeScores[team] += 0.5
                else:
                    # そうでない場合は、勝者に命中率と撃墜までの速さを加味した得点を加算し、敗者には（1-勝者の得点）を加算
                    winnerScore = (
                        0.6
                        + 0.3 * (hitCounts[winner] / launchedMissiles[winner] if launchedMissiles[winner] > 0 else 0.0)
                        + 0.1 * (1 if spentTime < 900.0 else max(0, 2.0-spentTime/900.0))
                    )
                    episodeScores[winner] += winnerScore
                    episodeScores[loser] += 1.0 - winnerScore

            else:
                # 時間切れの場合は引き分けなのでどちらも+0.5点
                for team in teams:
                    episodeScores[team] += 0.5
                pass

        detail = {
            "finishedTime": finishtime,
            "step_count": step_count,
            "numAlives":{team:float(np.sum([f.isAlive() for f in [f() for f in self.env.manager.getAssets(lambda a:isinstance(a,Fighter) and a.getTeam()==team)]])) for team in self.env.manager.getRuler()().score},
            "endReason":self.env.manager.getRuler()().endReason.name,
            "scores": episodeScores,
            "error": error,
            "red_error_message": red_error_message,
            "blue_error_message": blue_error_message,
            "logs": logs,
            "spentTime": spentTime,
            "events": events,
            "godlog": _god_finish(godlog, events) if self.match_config.get('god_log') else None,
            "hits": dict(hitCounts),
            "launched": dict(launchedMissiles),
            "winner": winner,
            "config": None
        }
        print("winner=", winner)

        return winner, detail


def map_r(x, callback_fn=None):
    # recursive map function
    if isinstance(x, (list, tuple, set)):
        return type(x)(map_r(xx, callback_fn) for xx in x)
    elif isinstance(x, dict):
        return type(x)((key, map_r(xx, callback_fn)) for key, xx in x.items())
    return callback_fn(x) if callback_fn is not None else None


# policyMapperの定義(基本はデフォルト通り)
def policyMapper(fullName):
    agentName, modelName, policyName = fullName.split(":")
    return policyName


def wrap_get_action_from_obs(args, time_out):
    @timeout(time_out)
    def get_action_from_obs(policies, rewards, terminateds, truncateds, infos, observation_space, action_space, policyMapper, obs, team):
        actions = {}
        for k, o in obs.items():
            suff_k = k.split('_')[0]
            if policyMapper(k) in policies:
                if team in suff_k:
                    actions[k] = policies[policyMapper(k)].step(o, rewards[k], terminateds[k] or truncateds[k], infos[k], k, observation_space[k], action_space[k])
        return actions
    
    return get_action_from_obs(*args)


def step_foward(func, args, time_out):
    """
    func: decorated with timeout
    """
    try:
        r = func(args, time_out)
        return r
    except TimeoutError:
        print('TimeOut')
        team = args[9]
        full_sample = args[6].sample()
        return {k: v for k, v in full_sample.items() if team in k.split('_')[0]}


def managerConfigReplacer(config,matchInfo):
    """
    対戦カードを表すmatchInfoに従い、SimulationManagerのコンフィグを置き換える関数。
    configはSimulationManager::getManagerConfig()で得られるものであり、
    Simulationmanagerのコンストラクタに渡すenv_configのうち"Manager"キー以下の部分となる。
    """
    ret=copy.deepcopy(config)
    agentConfigDispatcher=ret["AgentConfigDispatcher"]
    assetConfigDispatcher=ret["AssetConfigDispatcher"]
    for team, info in matchInfo.get("teams",matchInfo).items():
        if team == "Factory":
            continue
        agentConfigDispatcher[team+"Agents"]=info["Agent"]
        if "InitialState" in info:
            num=len(assetConfigDispatcher[team+"InitialState"]["elements"])
            for i in range(num):
                assetConfigDispatcher[team+"InitialState"]["elements"][i]["value"]["instanceConfig"]=info["InitialState"][i]
    return ret


def factoryConfigReplacer(config,matchInfo):
    """
    対戦カードを表すmatchInfoに従い、SimulationManagerのFactory側のコンフィグを置き換える関数。
    configはSimulationManager::getFactoryModelConfig(True)で得られるものであり、
    Simulationmanagerのコンストラクタに渡すenv_configのうち"Factory"キー以下の部分となる。
    """
    ret=nljson(config)
    if "Factory" in matchInfo.get("common", {}):
        ret.merge_patch(matchInfo["common"]["Factory"])

    return ret


def wrapEnvForTwoTeamCombatMatchMaking(base_class):
    """
    MatchMakerの付帯情報をgym.Envに付加するラッパー
    """
    class WrappedEnvForTwoTeamCombatMatchMaking(base_class):
        def setMatch(self,matchInfo):
            self.matchInfo=matchInfo
            self.requestReconfigure(
                managerConfigReplacer(unwrap_if_needed(self.manager.getManagerConfig()()),matchInfo),
                factoryConfigReplacer(unwrap_if_needed(self.manager.getFactoryModelConfig(True)()),matchInfo)
            )
    return WrappedEnvForTwoTeamCombatMatchMaking


class R8ContestMatchMakerForEvaluation(R8ContestTwoTeamCombatMatchMaker):
    # R8年度コンテストの評価サーバ用のMatchMaker
    def initialize(self,config):
        super().initialize(config)
        self.teams = list(self.match_config["teams"].keys())
        self.team_config=self.match_config["teams"]
        self.number=4

    def setAgent(self,ret,matchType,worker_index):
        for team in self.teams:
            userModelID=self.team_config[team]["userModelID"]
            isSingle=self.team_config[team]["isSingle"]
            if(isSingle):
                # 1個のインスタンスで1機なので異なるnameを割り当て、portはどちらも"0"
                ret[team]={"Agent":{
                    "type": "group",
                    "order": "fixed",
                    "elements": [
                        {"type": "External", "model": "Agent_"+userModelID, "policy": "Policy_" +
                            userModelID, "name": team+"_"+userModelID+"_"+str(i+1), "port": "0"} for i in range(self.number)
                    ]
                }}
            else:
                # 1個のインスタンスで2機分なので同じnameを割り当てることで1個のインスタンスとし、それぞれ異なるport("0"からstr(self.number-1))を割り当てる
                ret[team]={"Agent":{
                    "type": "group",
                    "order": "fixed",
                    "elements": [
                        {"type": "External", "model": "Agent_"+userModelID,
                            "policy": "Policy_"+userModelID, "name": team+"_"+userModelID, "port": str(i)} for i in range(self.number)
                    ]
                }}
        return ret


def _god_state(asset):
    m = asset.motion
    return [float(x) for x in m.pos()] + [float(x) for x in m.vel()]


def _god_step(manager, G):
    from ASRCAISim1.core import Missile
    t = manager.getElapsedTime()
    u2n = G['uid2name']
    if not u2n:
        for wa in manager.getAssets(lambda a: isinstance(a, Fighter)):
            a = wa()
            u2n[str(a.getUUIDForTrack())] = a.getFullName()
    fighters = {}
    for wa in manager.getAssets(lambda a: isinstance(a, Fighter)):
        a = wa()
        fighters[a.getFullName()] = a
    for wm in manager.getAssets(lambda a: isinstance(a, Missile)):
        m = wm()
        if not m.hasLaunched:
            continue
        nm = m.getFullName()
        rec = G['missiles'].get(nm)
        if rec is None:
            shooter = '/'.join(nm.split('/')[:2])
            tname = u2n.get(str(m.target.truth), '?')
            sh = fighters.get(shooter)
            tg = fighters.get(tname)
            rec = {'name': nm, 'team': m.getTeam(), 't0': t, 'shooter': shooter, 'target': tname,
                   's0': _god_state(sh) if sh is not None else None,
                   'g0': _god_state(tg) if tg is not None else None,
                   'thrust': None, 'traj': [], 'alive_end': None}
            try:
                rec['thrust'] = float(m.modelConfig()['thrust'])
            except Exception:
                pass
            G['missiles'][nm] = rec
        if m.isAlive():
            tg = fighters.get(rec['target'])
            if tg is not None and (len(rec['traj']) == 0 or t - rec['traj'][-1][0] >= 2.0):
                ms = _god_state(m)
                ts = _god_state(tg)
                rec['traj'].append([t] + ms + ts + [str(m.mode)])
            rec['t_last'] = t


def _god_finish(G, events):
    hits = {}
    for e in events:
        if e[1] == 'hit':
            hits.setdefault(e[2], []).append(e[3])
    out = []
    for nm, r in G['missiles'].items():
        r['hit'] = nm in hits
        r['hit_targets'] = hits.get(nm, [])
        out.append(r)
    return out
