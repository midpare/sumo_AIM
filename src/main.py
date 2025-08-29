import random
import time
import torch
import libsumo
import yaml
import numpy as np
from box import Box
from typing import cast
from enum import Enum, auto

from agent2 import D3QNAgent, AgentType
from logger import log_scenario
from car import CarConfigStatus, Car
from visualize import plot_agent_performance

class SimulationState(Enum):
    Collsion= auto()
    Clear = auto()
    NotEnd = auto()

class AIM:
    def __init__(self) -> None:
        self.load_config()
        self.setup_environment()
        self.initialize_agents()
        self.setup_simulation()

    def load_config(self):
        conf_url = './configs/car1_nbr3.yaml'
        with open(conf_url, 'r') as f:
            config_yaml = yaml.load(f, Loader=yaml.SafeLoader)
            self.config = Box(config_yaml)

    def setup_environment(self):
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        torch.cuda.manual_seed_all(self.config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.cmd = 'sumo-gui' if self.config.USE_GUI else 'sumo'

    def initialize_agents(self):
        self.veh_speeds = [i * self.config.MAX_SPEED / (self.config.DIVISION - 1) 
                          for i in range(self.config.DIVISION)]
        
        self.veh_speeds[0] = 1
        self.logged_agents = set()
        self.agents = {
            AgentType.LEFT: self.create_agent(AgentType.LEFT.value),
            AgentType.STRAIGHT: self.create_agent(AgentType.STRAIGHT.value),
            AgentType.RIGHT: self.create_agent(AgentType.RIGHT.value)
        }

        self.epsilon = self.config.EPS_START

    def setup_simulation(self):
        self.DT = self.config.sumo_config.step_length
        self.sim_cfg = self.build_sim_config()

    def create_agent(self, name: str) -> D3QNAgent:
        return D3QNAgent(
            name,
            ego_dim=self.config.EGO_DIM,
            nbr_dim=self.config.NBR_DIM,
            n_nbr=self.config.N_NBR,
            n_actions=self.config.DIVISION,
            mean_size=self.config.agent.mean_size,
            gamma=self.config.agent.gamma,
            batch_size=self.config.agent.batch_size,
            update_freq=self.config.agent.update_freq,
            lr = self.config.agent.lr,
            per_cfg=self.config.agent.per_cfg,
            device=self.DEVICE
    )

    def build_sim_config(self):
        cfg = self.config.sumo_config
        cmd = ["-c", cfg.config_file]
        
        cmd.extend(["--step-length", f"{self.DT}"])

        for key, value in cfg.options.items():
            option_name = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                cmd.extend([option_name, str(value).lower()])
            else:
                cmd.extend([option_name, str(value)])
        
        return cmd
    
    def spawn_scenario(self) -> dict[str, Car]:
        cars={}
        route_opts = {
            "N":["n2s","n2e","n2w"],
            "S":["s2n","s2e","s2w"],
            "E":["e2n","e2s","e2w"],
            "W":["w2n","w2s","w2e"]
        }    

        for prefix,routes in route_opts.items():
            for i in range(self.config.N_CAR):
                rid   = random.choice(routes)
                vid   = f"{prefix}_{int(time.time()*1e6)%1_000_000}_{i}"
                depart= cast(float, libsumo.simulation.getTime()) + i*2

                car = Car(vehID=vid, routeID=rid, typeID="autonomous", depart=depart, j_pos=self.config.J_POS)

                cars[vid] = car

        return cars
    
    def parse_to_junction_oriented(self, entry, x, y):
        match entry:
            case "N2J":
                return self.config.J_POS[0] - x, self.config.J_POS[1] - y
            case "W2J":
                return self.config.J_POS[1] - y, x - self.config.J_POS[0]
            case "S2J":
                return x - self.config.J_POS[0], y - self.config.J_POS[1]
            case "E2J":
                return y - self.config.J_POS[1], self.config.J_POS[0] - x
            case _:
                return None, None

    def state_of(self, vid, sub_ids, cars: dict[str, Car]):
        ego_x, ego_y = cars[vid].get_pos()
        entry = cars[vid].entry

        ego_x, ego_y = self.parse_to_junction_oriented(entry, ego_x, ego_y)

        if ego_x is None or ego_y is None:
            raise Exception("no entry!")
        
        others = []
        for other_id in sub_ids:
            if other_id == vid:
                continue
            
            other_car = cars[other_id]
            ox, oy = other_car.get_pos()
            ox, oy = self.parse_to_junction_oriented(entry, ox, oy)

            distance = np.linalg.norm([ego_x-ox, ego_y-oy])

            if distance > self.config.NBR_RANGE:
                continue
            
            ox = (ox - ego_x) / (self.config.NBR_RANGE * 2) + 0.5
            oy = (oy - ego_y) / (self.config.NBR_RANGE * 2) + 0.5

            if ox is None or oy is None:
                raise Exception("no entry!")
        
            others.append((distance, [ox, oy]))
        
        
        data = [0 for _ in range(self.config.EGO_DIM + self.config.N_NBR*self.config.NBR_DIM)]
        data[0] = ego_x /(self.config.GLOBAL_RANGE * 2) + 0.5
        data[1] = ego_y / (self.config.GLOBAL_RANGE * 2) + 0.5
        data[2] = cars[vid].speed / self.config.MAX_SPEED

        closest_10 = sorted(others)[:self.config.N_NBR]
        
        while len(closest_10) < self.config.N_NBR:
            closest_10.append((float('inf'), [0.0, 0.0]))
        for i, (_, (ox, oy)) in enumerate(closest_10):             
            data[self.config.EGO_DIM + i*self.config.NBR_DIM] = ox
            data[self.config.EGO_DIM + i*self.config.NBR_DIM+1] = oy
                    
        return np.array(data, dtype=np. float32)

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

    def update_vehicle_states(self, cars: dict[str, Car]):
        if any(car.is_unconfigured() for car in cars.values()):
            for vid in libsumo.simulation.getDepartedIDList():
                if cars[vid].is_unconfigured():
                    cars[vid].subscribe()

        all_sub_results = libsumo.vehicle.getAllSubscriptionResults()
        for vid, data in all_sub_results.items():
            if vid in cars:
                cars[vid].set_data(data)

        return list(all_sub_results.keys())

    def log_training_progress(self, training_results: dict, data: dict):
        for agent_type, values in training_results.items():
            if not values:
                continue

            log_parts = [f"name: {agent_type.value}"]
            
            for metric, value in values.items():
                if value is None:
                    continue
                
                if isinstance(value, (float, np.floating)):
                    log_parts.append(f"{metric}: {value:.4f}")
                else:
                    log_parts.append(f"{metric}: {value}")

                if metric == "step":
                    data[agent_type.value][metric] = value
                elif metric in ["Q-mean", "Q-std", "TD-error"]:
                    data[agent_type.value][metric].append(value)

            if len(log_parts) > 1:
                print(", ".join(log_parts))

    def evaluate_and_log_performance(self, scen_idx: int, data: dict, start_time: float) -> float:
        avg_reward, success_rate, avg_step = self.evaluate_policy(20)

        data["episodes"].append(scen_idx)
        data["total_reward"].append(avg_reward)
        data["success_rate"].append(success_rate)

        current_time = time.time()
        log_scenario(scen_idx, avg_reward, success_rate, avg_step, self.epsilon, current_time - start_time)
        
        return current_time
    
    def select_and_execute_actions(self, cars: dict[str, Car], sub_ids: list[str]):
        for vid in sub_ids:
            if vid not in cars:
                continue
                
            car = cars[vid]
            if not car.config_status == CarConfigStatus.Configured:
                car.set_speed(self.veh_speeds[2])
                continue

            current_state = self.state_of(vid, sub_ids, cars)
            action_idx = self.agents[car.agent_type].select_action(current_state, self.epsilon)
            
            car.set_speed(self.veh_speeds[action_idx])
            car.set_action_history(current_state, action_idx)

    def process_transitions_and_train(self, cars: dict[str, Car], sub_ids: list[str], scen_idx: int, step: int):
        should_log_scenario = (scen_idx % self.config.LOG_INTERVAL) == 0
        training_results = {}
        collision_occurred = False

        for vid in list(sub_ids):
            if vid not in cars:
                continue
                
            car = cars[vid]
            prev_state, prev_action = car.get_action_history()
            
            if prev_state is None or prev_action is None:
                continue
            
            current_state = self.state_of(vid, sub_ids, cars)
            config_status = car.update_car_config_status(self.config.ENTERED_RADIUS, self.config.EXITED_RADIUS)

            is_collision = self.check_collision(vid, sub_ids, cars)
            is_exited = (config_status == CarConfigStatus.Exited)
            done = is_collision or is_exited
            
            reward = self.calculate_final_reward(prev_action, is_collision, is_exited, car, sub_ids, cars, step)
            
            collision_occurred |= is_collision
            
            agent = self.agents[car.agent_type]
            
            if config_status in [CarConfigStatus.Entered, CarConfigStatus.Exited]:
                agent.store(prev_state, prev_action, reward, current_state, done)

            if is_exited:
                car.delete()
                if vid in sub_ids:
                    sub_ids.remove(vid)

        for agent_type, agent in self.agents.items():
            should_log = should_log_scenario and agent_type not in self.logged_agents

            result = agent.train(store_result=should_log)
            if should_log and result:
                training_results[agent_type] = result
                self.logged_agents.add(agent_type)
                 
        return training_results, collision_occurred
        
    def calculate_reward(self, action_idx: int, is_collision: bool, is_exited: bool, 
                        car: Car, sub_ids: list, cars: dict[str, Car]) -> float:
        
        # 기본 이동 보상 (최소한만)
        speed = self.veh_speeds[action_idx]
        base_reward = 0.01 * (speed / self.config.MAX_SPEED) if speed > 0 else -0.02
        
        # 안전 거리 유지 보상 (핵심)
        safety_reward = self.calculate_safety_reward(car, sub_ids, cars)
        base_reward += safety_reward
        
        # 최종 결과만 큰 보상
        if is_collision:
            base_reward -= 1.5
        elif is_exited:
            base_reward += 2.0
        
        return base_reward
    def get_min_distance_to_others(self, car: Car, sub_ids: list, cars: dict) -> float:
        ego_x, ego_y = car.get_pos()
        min_distance = float('inf')
        
        for other_id in sub_ids:
            if other_id == car.vehID or other_id not in cars:
                continue
                
            other_car = cars[other_id]
            ox, oy = other_car.get_pos()
            distance = np.hypot(ego_x - ox, ego_y - oy)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def calculate_safety_reward(self, car: Car, sub_ids: list, cars) -> float:
        min_distance = self.get_min_distance_to_others(car, sub_ids, cars)
        
        if min_distance == float('inf'):
            return 0.0
        
        # 안전거리만 평가
        if min_distance < 3.0:
            return -0.1 * (4.0 - min_distance)  # 3m 이하에서 페널티
        elif min_distance > 6.0:
            return 0.02  # 충분한 거리 유지시 소량 보상
        else:
            return 0.0

    def calculate_final_reward(self, action_idx: int, is_collision: bool, is_exited: bool, 
                            car: Car, sub_ids: list, cars: dict[str, Car], step: int) -> float:
        total_reward = 0.0
        
        speed = self.veh_speeds[action_idx] 
        total_reward += 0.01 * (speed / self.config.MAX_SPEED) if speed > 0 else -0.01
        
        total_reward += self.calculate_safety_reward(car, sub_ids, cars)
                
        if is_collision:
            total_reward -= 1.0
        elif is_exited:
            time_bonus = max(0, 50 - step) * 0.01
            total_reward += 1.5 + time_bonus
        
        return np.clip(total_reward, -1.5, 2.0)
    def check_termination_conditions(self, cars: dict[str, Car], collision_occurred: bool) -> SimulationState:
        if collision_occurred:
            return SimulationState.Collsion
        elif all(car.is_exited() for car in cars.values()):
            return SimulationState.Clear
        else:
            return SimulationState.NotEnd

    def evaluate_policy(self, n_episodes):
        successes = 0
        rewards = []
        steps = 0
        for _ in range(n_episodes):
            libsumo.load(self.sim_cfg)
            cars = self.spawn_scenario()
            
            step = 0
            total_reward = 0
            
            while step < self.config.MAX_STEPS:
                libsumo.simulationStep()
                step += 1

                sub_ids = self.update_vehicle_states(cars)
                collision_flag = False

                for vid in list(sub_ids):
                    if vid not in cars:
                        continue
                        
                    car = cars[vid]
                    prev_state, prev_action = car.get_action_history()

                    if prev_state is not None and prev_action is not None:
                        current_state = self.state_of(vid, sub_ids, cars)

                        config_status = car.update_car_config_status(self.config.ENTERED_RADIUS, self.config.EXITED_RADIUS)

                        is_collision = self.check_collision(vid, sub_ids, cars)
                        is_exited = (config_status == CarConfigStatus.Exited)
                        
                        reward = self.calculate_final_reward(prev_action, is_collision, is_exited, car, sub_ids, cars, step)
                        
                        collision_flag |= is_collision
                        total_reward += reward
                        
                        if is_exited:
                            car.delete()
                            sub_ids.remove(vid)

                # 다음 액션 선택 (ε=0으로 평가)
                for vid in sub_ids:
                    if vid not in cars:
                        continue
                    car = cars[vid]
                    current_state = self.state_of(vid, sub_ids, cars)
                    action_idx = self.agents[car.agent_type].select_action(current_state, 0.0)  # 탐험 없음
                    
                    car.set_speed(self.veh_speeds[action_idx])
                    car.set_action_history(current_state, action_idx)

                if collision_flag:
                    break
                elif all(car.is_exited() for car in cars.values()):
                    successes += 1
                    break
            
            rewards.append(total_reward)
            steps += step
            
        return np.mean(rewards), successes / n_episodes * 100, steps / n_episodes

    def initialize_data_structure(self):
        """데이터 구조 초기화"""
        return {
            "episodes": [],
            'success_rate': [], 
            'total_reward': [],
            **{agent.value: {
                'Q-mean': [],
                'Q-std': [],
                'TD-error': [],
                'step': 0
            } for agent in AgentType}
        }
    
    def save_models(self):
        for agent_type in AgentType:
            torch.save(self.agents[agent_type].q_net.state_dict(), f"./{agent_type.value}_weight.pth")
            print(f"Model saved: {agent_type.value}")

    def start(self):
        libsumo.start([self.cmd] + self.sim_cfg)

        start_time = time.time()
        data = self.initialize_data_structure()
        
        for scen_idx in range(1, self.config.MAX_SCENARIOS+1):
            self.logged_agents.clear()
            cars = self.spawn_scenario()
            step = 0

            while step < self.config.MAX_STEPS:
                libsumo.simulationStep()
                step += 1

                sub_ids = self.update_vehicle_states(cars)

                if step > 1:
                    training_results, collision_occurred = self.process_transitions_and_train(cars, sub_ids, scen_idx, step)

                    if training_results:
                        self.log_training_progress(training_results, data)
                else:
                    collision_occurred = False

                sim_state = self.check_termination_conditions(cars, collision_occurred)
                if sim_state in [SimulationState.Collsion, SimulationState.Clear]:
                    break

                self.select_and_execute_actions(cars, sub_ids)

            if step >= self.config.MAX_STEPS:
                raise Exception(f'Scenario {scen_idx} did not terminate properly')

            if scen_idx % self.config.LOG_INTERVAL == 0:
                start_time = self.evaluate_and_log_performance(scen_idx, data, start_time)

            self.epsilon = max(self.config.EPS_END, self.epsilon - self.config.EPS_DECAY)
            libsumo.load(self.sim_cfg)

        self.save_models()
        plot_agent_performance(data, "agent_performance.png")
        libsumo.close()

aim = AIM()

aim.start()