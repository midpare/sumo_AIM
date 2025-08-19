# train.py
import os, random, time, traci, sumolib
import numpy as np
import torch
from agent import D3QNAgent, AgentType
from logger import log_scenario, log_loss
from car import CarConfigStatus, Car
from typing import cast


class AIM:
    def __init__(self) -> None:
        
        # ---------- 환경 경로 ----------
        self.SUMO_CFG    = "config.sumocfg"

        # ---------- 하이퍼파라미터 ----------
        self.MAX_SCENARIOS    = 10_000        # 학습에 돌릴 총 시나리오 수
        self.MAX_STEPS        = 1_000         # 시나리오 하나의 최대 step
        self.TEMPERATURE      = 1.0           # PSR softmax 온도
        self.BATCH_SIZE       = 256
        self.EPS_START, self.EPS_END, self.EPS_DECAY = 1.0, 0.05, 1e-4
        self.VEH_SPEEDS       = [0.0, 13]   # action→speed 매핑
        DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
        self.ENTERED_RADIUS = 40.0   # 교차로에 ‘들어왔다’고 간주할 거리
        self.EXITED_RADIUS  = 30.0   # 교차로를 ‘완전히 빠져나갔다’고 간주할 거리
        self.J_POS = (44, 55)               # 교차로 중심(수동 입력)
        self.LOG_INTERVAL = 10

        # ---------- Agent 3개 초기화 ----------
        self.agents = {
            AgentType.LEFT: D3QNAgent("left", input_shape=5*11, n_actions=2, device=DEVICE),
            AgentType.STRAIGHT: D3QNAgent("straight", input_shape=5*11, n_actions=2, device=DEVICE),
            AgentType.RIGHT: D3QNAgent("right", input_shape=5*11, n_actions=2, device=DEVICE)
        }

        self.config = [
            "-c",self.SUMO_CFG,
            # "--step-length","0.5",
            "--no-warnings","true", 
            # "--delay", "100", 
            "--collision.action", "none",
            # "--quiet"        
            "--no-step-log", "true",        # 🔥 스텝 로그 끄기
            "--verbose", "false",           # 🔥 상세 로그 끄기
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
                depart= cast(float, traci.simulation.getTime()) + i*0.5       # 0.5 s 간격
                
                car = Car(vehID=vid, routeID=rid, typeID="autonomous", depart=depart, j_pos=self.J_POS)

                cars[vid] = car

        return cars

# ---------- 보조 함수 ----------
    def state_of(self, vid, sub_ids, cars: dict[str, Car]):
        ego_state = cars[vid].get_normalized_state()
        
        # 다른 차량들의 정보와 거리 계산
        others = []
        for other_id in sub_ids:
            if other_id == vid:
                continue
            
            other_car = cars[other_id]
            other_state = other_car.get_normalized_state()

            distance = np.linalg.norm([ego_state[0]-other_state[0], ego_state[1]-other_state[1]])

            if distance > 0.2:
                continue

            rel_x = (other_state[0] - ego_state[0]) * 5/2 + 0.5
            rel_y = (other_state[1] - ego_state[1]) * 5/2 + 0.5

            others.append((distance, [rel_x, rel_y, other_state[2], other_state[3], other_state[4]]))
        
        # 가장 가까운 10대 선택하고 데이터 추가
        closest_10 = sorted(others)[:10]
        
        while len(closest_10) < 10:
            closest_10.append((float('inf'), [0.0, 0.0, 0.0, 0.0, 0.0]))
        for _, other_data in closest_10:
            ego_state.extend(other_data)
        
        return np.array(ego_state, dtype=np.float32)

    def collision(self, vid, sub_ids, cars: dict[str, Car], thr=3.0):
        if vid not in sub_ids:
            return False
        
        x1, y1 = cars[vid].get_state()[:2]
        
        for other_id in sub_ids:
            if other_id == vid:
                continue

            x2, y2 = cars[other_id].get_state()[:2]

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
            
            car.set_speed(self.VEH_SPEEDS[a_idx])
            car.set_action_history(s, a_idx)

    def set_cars_data(self, cars: dict[str, Car]):
        all_sub_results = traci.vehicle.getAllSubscriptionResults()

        for vid, data in all_sub_results.items():
            car = cars[vid]
            car.set_data(data)

        return list(all_sub_results.keys())

    def start(self):
        traci.start(['sumo'] + self.config)  # 시나리오 시작

        scen_start = time.time()

        for scen_idx in range(1, self.MAX_SCENARIOS+1):
            # print(f"Scenario {scen_idx} set in {time.time() - scen_start:.3f}s")
            cars = self.spawn_scenario()                       # 차량 배치
            step, tot_reward = 0, 0.0
            collision_flag = False

            traci.simulationStep()
            step += 1

            if not all(not car.is_unconfigured() for car in cars.values()):
                self.check_departed(cars)
                    
            sub_ids = self.set_cars_data(cars)

            self.proceed_action(sub_ids, cars)
            # 이제 첫 simulationStep 실행
            while step < self.MAX_STEPS:
                traci.simulationStep()
                step += 1

                if not all(not car.is_unconfigured() for car in cars.values()):
                    self.check_departed(cars)

                sub_ids = self.set_cars_data(cars)

                for vid in sub_ids:
                    car = cars[vid]
                    s, a_idx = car.get_action_history()

                    if s is None or a_idx is None:
                        continue
                
                    next_s = self.state_of(vid, sub_ids, cars)  # 액션 실행 후 상태

                    scale = 0.05

                    car.update_car_config_status(self.ENTERED_RADIUS, self.EXITED_RADIUS)

                    reward  = 0.05 if self.VEH_SPEEDS[a_idx]>0 else -scale
                    if self.collision(vid, sub_ids, cars):
                        reward, collision_flag = -10.0 *scale, True
                    

                    if car.get_config_status() == CarConfigStatus.Exited:
                        reward = 10.0 * scale
                        tot_reward += reward
                        car.ep_trans.append((s, a_idx, reward, next_s, collision_flag))
                        car.delete()

                        sub_ids.remove(vid)

                        continue

                    tot_reward += reward
                    car.ep_trans.append((s, a_idx, reward, next_s, collision_flag))

                self.proceed_action(sub_ids, cars)

                if collision_flag:
                    if scen_idx % self.LOG_INTERVAL == 0:
                        log_scenario(scen_idx, tot_reward, step, self.epsilon, True)
                    break
                    
                if all(car.is_exited() for car in cars.values()):         
                    if scen_idx % self.LOG_INTERVAL == 0:
                        log_scenario(scen_idx, tot_reward, step, self.epsilon, False)

                    break

            # print(f"Scenario {scen_idx} ended in {time.time() - scen_start:.3f}s, step: {step}")
            
            # scenario_score = -tot_reward          # 점수 계산(충돌 패널티)

            for vid, car in cars.items():
                if car.is_transition_empty(): 
                    continue  

                agent_type = car.agent_type
                for t in car.ep_trans:
                    self.agents[agent_type].store(*t)

            losses = []
            for agent_type in AgentType:
                loss = self.agents[agent_type].train()
                if not loss is None:
                    losses.append(loss)
                    # print(1)
            
            if scen_idx % self.LOG_INTERVAL == 0 and len(losses) > 2:
                log_loss(scen_idx, losses, self.epsilon)
            self.epsilon = max(self.EPS_END, self.epsilon-self.EPS_DECAY)
            traci.load(self.config)


aim = AIM()

aim.start()