# train.py
import os, random, time, traci, sumolib
import numpy as np
import torch
from agent import D3QNAgent, AgentType
from logger import log_scenario, log_loss
from car import CarConfigStatus, Car
from typing import cast
from collections import deque
from enum import Enum, auto

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
        self.MAX_SCENARIOS    = 10_000       # 학습에 돌릴 총 시나리오 수
        self.MAX_STEPS        = 3_00         # 시나리오 하나의 최대 step
        self.TEMPERATURE      = 1.0           # PSR softmax 온도
        self.BATCH_SIZE       = 256
        self.EPS_START, self.EPS_END, self.EPS_DECAY = 1.0, 0.05, 1e-3
        DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
        self.ENTERED_RADIUS = 30.0   # 교차로에 ‘들어왔다’고 간주할 거리
        self.EXITED_RADIUS  = 40.0   # 교차로를 ‘완전히 빠져나갔다’고 간주할 거리
        self.J_POS = (44, 55)               # 교차로 중심(수동 입력)
        self.LOG_INTERVAL = 1

        self.GLOBAL_RANGE = 100
        self.NEIGHBOR_RANGE = 40
        self.NEIGHBOR_COUNT = 5
        self.MAX_SPEED = 15
        self.DIVISION = 4
        self.USE_GUI = True

        self.veh_speeds = [i * self.MAX_SPEED / (self.DIVISION - 1) for i in range(self.DIVISION)]
        print(self.veh_speeds)
        # ---------- Agent 3개 초기화 ----------
        self.agents = {
            AgentType.LEFT: D3QNAgent("left", input_shape=3+2*self.NEIGHBOR_COUNT, n_actions=self.DIVISION, device=DEVICE),
            AgentType.STRAIGHT: D3QNAgent("straight", input_shape=3+2*self.NEIGHBOR_COUNT, n_actions=self.DIVISION, device=DEVICE),
            AgentType.RIGHT: D3QNAgent("right", input_shape=3+2*self.NEIGHBOR_COUNT, n_actions=self.DIVISION, device=DEVICE)
        }

        self.cmd = 'sumo-gui' if self.USE_GUI else 'sumo'
        self.config = [
            "-c",self.SUMO_CFG,
            "--step-length","0.2",
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
            for i in range(2):
                rid   = random.choice(routes)
                vid   = f"{prefix}_{int(time.time()*1e6)%1_000_000}_{i}"
                depart= cast(float, traci.simulation.getTime()) + i*2

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

            if distance > self.NEIGHBOR_RANGE:
                continue

            rel_x = (x-ox) / (self.NEIGHBOR_RANGE * 2) + 0.5
            rel_y = (y-oy) / (self.NEIGHBOR_RANGE * 2) + 0.5

            others.append((distance, [rel_x, rel_y]))
        
        data = cars[vid].get_junc_oriented_state()
        
        data[0] = data[0] / (self.GLOBAL_RANGE * 2) + 0.5
        data[1] = data[1] / (self.GLOBAL_RANGE * 2) + 0.5
        data[2] = data[2] / self.MAX_SPEED

        closest_10 = sorted(others)[:self.NEIGHBOR_COUNT]
        
        while len(closest_10) < self.NEIGHBOR_COUNT:
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
        for vid in traci.simulation.getDepartedIDList():
            if not cars[vid].is_unconfigured():
                continue

            cars[vid].subscribe()

    def proceed_action(self, sub_ids, cars: dict[str, Car]):
        for vid in sub_ids:
            car = cars[vid]
            s = self.state_of(vid, sub_ids, cars)
            a_idx = self.agents[car.agent_type].select_action(s, self.epsilon)            

            car.set_speed(self.veh_speeds[a_idx])
            car.set_action_history(s, a_idx)

    def set_cars_data(self, cars: dict[str, Car]):
        all_sub_results = traci.vehicle.getAllSubscriptionResults()

        for vid, data in all_sub_results.items():
            car = cars[vid]
            car.set_data(data)

        return list(all_sub_results.keys())

    def train(self, print_q):
        losses = []
        for agent_type in AgentType:
            loss = self.agents[agent_type].train(print_q)
            losses.append(loss) if not loss is None else None 

        return losses

    def simulate_single(self, cars):
        step = 0
        losses = []
        print_q = False
        while step < self.MAX_STEPS:
            traci.simulationStep()
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

                scale = 0.05
                reward  = self.veh_speeds[a_idx] * scale/5 if self.veh_speeds[a_idx]>0 else -scale
                
                if collision_flag:
                    reward = - 10.0 * scale
                if exited_flag:
                    reward = 10.0 * scale
                    car.delete()
                    sub_ids.remove(vid)

                self.agents[car.agent_type].store(s, a_idx, reward, next_s, collision_flag | exited_flag)

            if collision_flag:
                return SimulationState.Collsion, losses
            elif all(car.is_exited() for car in cars.values()):
                return SimulationState.Clear, losses
            
            self.proceed_action(sub_ids, cars)

            losses = self.train(print_q)[:]
            print_q |= True


        return SimulationState.NotEnd, []

    def start(self):
        traci.start([self.cmd] + self.config)  # 시나리오 시작
        success = deque(maxlen=30)

        for scen_idx in range(1, self.MAX_SCENARIOS+1):
            cars = self.spawn_scenario()                       # 차량 배치
            result, losses = self.simulate_single(cars)

            if scen_idx % self.LOG_INTERVAL == 0 and len(losses) > 2:
                log_loss(scen_idx, losses, self.epsilon)

            success_rate = success.count(True) / len(success) if len(success) > 0 else 0
            match result:
                case SimulationState.Collsion:
                    if scen_idx % self.LOG_INTERVAL == 0:
                        log_scenario(scen_idx, success_rate * 100, self.epsilon, True)
                    success.append(False)
                case SimulationState.Clear:
                    if scen_idx % self.LOG_INTERVAL == 0:
                        log_scenario(scen_idx, success_rate * 100, self.epsilon, False)
                    success.append(True)
                case SimulationState.NotEnd:
                    raise Exception(f'scenario not end in id:{scen_idx}')
            

            self.epsilon = max(self.EPS_END, self.epsilon-self.EPS_DECAY)
            traci.load(self.config)


aim = AIM()

aim.start()