# train.py
import random, time
import numpy as np
import torch
from agent import D3QNAgent, AgentType
from logger import log_scenario, log_loss
from car import CarConfigStatus, Car
from typing import cast
from collections import deque
from enum import Enum, auto
import libsumo

class SimulationState(Enum):
    Collsion= auto()
    Clear = auto()
    NotEnd = auto()

class AIM:
    def __init__(self) -> None:
        
        # ---------- 환경 경로 ----------
        self.SUMO_CFG    = "config.sumocfg"

        SEED = 11

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

        # ---------- 하이퍼파라미터 ----------
        self.MAX_SCENARIOS    = 50_000       # 학습에 돌릴 총 시나리오 수
        self.MAX_STEPS        = 3_00         # 시나리오 하나의 최대 step
        self.TEMPERATURE      = 1.0           # PSR softmax 온도
        self.BATCH_SIZE       = 256
        self.EPS_START, self.EPS_END, self.EPS_DECAY = 1.0, 0.05, 5e-5
        self.DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
        self.ENTERED_RADIUS = 30.0   # 교차로에 ‘들어왔다’고 간주할 거리
        self.EXITED_RADIUS  = 40.0   # 교차로를 ‘완전히 빠져나갔다’고 간주할 거리
        self.J_POS = (44, 55)               # 교차로 중심(수동 입력)
        self.LOG_INTERVAL = 100

        self.GLOBAL_RANGE = 100
        self.NBR_RANGE = 40
        self.N_NBR = 5
        self.MAX_SPEED = 15
        self.DIVISION = 4
        self.USE_GUI = False
        self.N_CAR = 2

        self.DT = 0.2
        self.veh_speeds = [i * self.MAX_SPEED / (self.DIVISION - 1) for i in range(self.DIVISION)]
        print(self.veh_speeds)
        # ---------- Agent 3개 초기화 ----------
        self.agents = {
            AgentType.LEFT: D3QNAgent("left", ego_dim=3, nbr_dim=2, n_nbr=self.N_NBR, n_actions=self.DIVISION, device=self.DEVICE),
            AgentType.STRAIGHT: D3QNAgent("straight", ego_dim=3, nbr_dim=2, n_nbr=self.N_NBR, n_actions=self.DIVISION, device=self.DEVICE),
            AgentType.RIGHT: D3QNAgent("right", ego_dim=3, nbr_dim=2, n_nbr=self.N_NBR, n_actions=self.DIVISION, device=self.DEVICE)
        }

        self.cmd = 'sumo-gui' if self.USE_GUI else 'sumo'
        self.config = [
            "-c",self.SUMO_CFG,
            "--step-length",f"{self.DT}",
            "--no-warnings","true",
            "--collision.action", "none",
            # "--delay", "200",
            "--no-step-log", "true",
            "--verbose", "false",
            "--start"
        ]

        self.epsilon = self.EPS_START
    
    # ---------- 시나리오 자동 생성 ----------
    def spawn_scenario(self) -> dict[str, Car] :
        """4 개 진입로(N/S/E/W)에서 1~4대씩 무작위로 진입"""
        cars={}
        route_opts = {
            "N":["n2s","n2e","n2w"],
            "S":["s2n","s2e","s2w"],
            "E":["e2n","e2s","e2w"],
            "W":["w2n","w2s","w2e"]
        }    

        for prefix,routes in route_opts.items():
            for i in range(self.N_CAR):
                rid   = random.choice(routes)
                vid   = f"{prefix}_{int(time.time()*1e6)%1_000_000}_{i}"
                depart= cast(float, libsumo.simulation.getTime()) + i*2

                car = Car(vehID=vid, routeID=rid, typeID="autonomous", depart=depart, j_pos=self.J_POS)

                cars[vid] = car

        return cars

# ---------- 보조 함수 ----------
    def state_of(self, vid, sub_ids, cars: dict[str, Car]):
        x, y = cars[vid].get_pos()
        
        others = []
        for other_id in sub_ids:
            if other_id == vid:
                continue
            
            other_car = cars[other_id]
            ox, oy = other_car.get_pos()

            distance = np.linalg.norm([x-ox, y-oy])

            if distance > self.NBR_RANGE:
                continue

            rel_x = (x-ox) / (self.NBR_RANGE * 2) + 0.5
            rel_y = (y-oy) / (self.NBR_RANGE * 2) + 0.5

            others.append((distance, [rel_x, rel_y]))
        
        data = cars[vid].get_junc_oriented_state()
        
        data[0] = data[0] / (self.GLOBAL_RANGE * 2) + 0.5
        data[1] = data[1] / (self.GLOBAL_RANGE * 2) + 0.5
        data[2] = data[2] / self.MAX_SPEED

        closest_10 = sorted(others)[:self.N_NBR]
        
        while len(closest_10) < self.N_NBR:
            closest_10.append((float('inf'), [0.0, 0.0]))
        for _, other_data in closest_10:
            data.extend(other_data)
        
        return np.array(data, dtype=np.float32)

    def check_collision(self, vid, sub_ids, cars: dict[str, Car], thr=3.0):
        if vid not in sub_ids:
            return False
        
        x1, y1 = cars[vid].get_pos()
        
        for other_id in sub_ids:
            if other_id == vid:
                continue

            x2, y2 = cars[other_id].get_pos()

            if np.hypot(x1-x2, y1-y2) < thr:
                return True
        return False

    def check_departed(self, cars: dict[str, Car]):
        for vid in libsumo.simulation.getDepartedIDList():
            if not cars[vid].is_unconfigured():
                continue

            cars[vid].subscribe()

    def proceed_action(self, sub_ids, cars: dict[str, Car], is_train):
        for vid in sub_ids:
            car = cars[vid]
            s = self.state_of(vid, sub_ids, cars)
            epsilon = self.epsilon if is_train else 0
            a_idx = self.agents[car.agent_type].select_action(s, epsilon)            

            car.set_speed(self.veh_speeds[a_idx])
            car.set_action_history(s, a_idx)

    def set_cars_data(self, cars: dict[str, Car]):
        all_sub_results = libsumo.vehicle.getAllSubscriptionResults()

        for vid, data in all_sub_results.items():
            car = cars[vid]
            car.set_data(data)

        return list(all_sub_results.keys())

    def train(self):
        losses = []
        for agent_type in AgentType:
            loss = self.agents[agent_type].train()
            losses.append(loss) if not loss is None else None 

        return losses

    def simulate_single(self, cars, is_train=True):
        step = 0
        tot_reward = 0
        losses = []
        print_q = False
        while step < self.MAX_STEPS:
            start_time = time.time()
            libsumo.simulationStep()
            step += 1

            if any(car.is_unconfigured() for car in cars.values()):
                self.check_departed(cars)

            sub_ids = self.set_cars_data(cars)

            collision_flag = False

            for vid in sub_ids:
                car = cars[vid]
                s, a_idx = car.get_action_history()

                if s is None or a_idx is None:
                    continue
                
                collision_flag |= self.check_collision(vid, sub_ids, cars)
                exited_flag = car.update_car_config_status(self.ENTERED_RADIUS, self.EXITED_RADIUS) == CarConfigStatus.Exited

                next_s = self.state_of(vid, sub_ids, cars)

                scale = 1
                reward  = self.veh_speeds[a_idx] * self.DT/5 if self.veh_speeds[a_idx]>0 else -scale/5
                
                if collision_flag:
                    reward = - 50.0 * scale
                if exited_flag:
                    reward = 10.0 * scale
                    car.delete()
                    sub_ids.remove(vid)

                if is_train:
                    self.agents[car.agent_type].store(s, a_idx, reward, next_s, collision_flag | exited_flag)
                tot_reward += reward

            if collision_flag:
                return SimulationState.Collsion, tot_reward
            elif all(car.is_exited() for car in cars.values()):
                return SimulationState.Clear, tot_reward
            

            self.proceed_action(sub_ids, cars, is_train)
            
            if is_train:
                losses = self.train()


        return SimulationState.NotEnd, tot_reward

    def evaluate_policy(self, n_episodes):
        """표준 정책 평가 (ε=0)"""
        successes = 0
        rewards = np.array([])

        for _ in range(n_episodes):
            libsumo.load(self.config)

            cars = self.spawn_scenario()
            result, tot_reward = self.simulate_single(cars, False)  # 탐험 없음
            if result == SimulationState.Clear:
                successes += 1
            
            rewards = np.append(rewards, tot_reward)
            
        return np.mean(rewards), successes / n_episodes * 100

    def start(self):
        libsumo.start([self.cmd] + self.config)  # 시나리오 시작
        # explore_success = np.array()
        # rewards = np.array()
        start_time = time.time()
        for scen_idx in range(1, self.MAX_SCENARIOS+1):
            cars = self.spawn_scenario()                       # 차량 배치
            result, tot_reward = self.simulate_single(cars)

            if result == SimulationState.NotEnd:
                raise Exception(f'scenario not end in id:{scen_idx}')

            # idx = scen_idx % self.LOG_INTERVAL
            # explore_success[idx] = result
            # rewards[idx] = tot_reward

            # explore_success_rate = explore_success.count(SimulationState.Clear) / len(explore_success) if len(explore_success) > 0 else 0
            # avg_reward = np.mean(rewards)
            if scen_idx % self.LOG_INTERVAL == 0:
                avg_reward, success_rate = self.evaluate_policy(20)

                cur = time.time()
                log_scenario(scen_idx, avg_reward, success_rate, self.epsilon, cur - start_time)
                start_time = cur

            self.epsilon = max(self.EPS_END, self.epsilon-self.EPS_DECAY)
            libsumo.load(self.config)


aim = AIM()

aim.start()