import random, time, torch, libsumo, yaml
import numpy as np
from box import Box
from typing import cast
from enum import Enum, auto

from agent import D3QNAgent, AgentType
from logger import log_scenario
from car import CarConfigStatus, Car

class SimulationState(Enum):
    Collsion= auto()
    Clear = auto()
    NotEnd = auto()

class AIM:
    def __init__(self) -> None:
        
        conf_url = './config/car1_nbr3.yaml'
        with open(conf_url, 'r') as f:
            config_yaml = yaml.load(f, Loader=yaml.SafeLoader)
            self.config = Box(config_yaml)


        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        torch.cuda.manual_seed_all(self.config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.cmd = 'sumo-gui' if self.config.USE_GUI else 'sumo'

        self.veh_speeds = [i * self.config.MAX_SPEED / (self.config.DIVISION - 1) for i in range(self.config.DIVISION)]
        print(self.veh_speeds)
        # ---------- Agent 3개 초기화 ----------
        self.agents = {
            AgentType.LEFT: self.create_agent("left_agent"),
            AgentType.STRAIGHT: self.create_agent("straight_agent"),
            AgentType.RIGHT: self.create_agent("right_agent")
        }

        self.epsilon = self.config.EPS_START
        self.DT = self.config.sumo_config.step_length # too long!
        self.sim_cfg = self._build_sim_config()
    
    def _build_sim_config(self):
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
            per_cfg=self.config.agent.per_cfg,
            device=self.DEVICE
    )

    def spawn_scenario(self) -> dict[str, Car] :
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

# ---------- 보조 함수 ----------
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


    def simulate_single(self, cars, is_train=True):
        step = 0
        tot_reward = 0
        while step < self.config.MAX_STEPS:
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
                exited_flag = car.update_car_config_status(self.config.ENTERED_RADIUS, self.config.EXITED_RADIUS) == CarConfigStatus.Exited

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
                for agent_type in AgentType:
                    self.agents[agent_type].train()



        return SimulationState.NotEnd, tot_reward

    def evaluate_policy(self, n_episodes):
        """표준 정책 평가 (ε=0)"""
        successes = 0
        rewards = np.array([])

        for _ in range(n_episodes):
            libsumo.load(self.sim_cfg)

            cars = self.spawn_scenario()
            result, tot_reward = self.simulate_single(cars, False)  # 탐험 없음
            if result == SimulationState.Clear:
                successes += 1
            
            rewards = np.append(rewards, tot_reward)
            
        return np.mean(rewards), successes / n_episodes * 100

    def start(self):
        libsumo.start([self.cmd] + self.sim_cfg)  # 시나리오 시작

        start_time = time.time()
        for scen_idx in range(1, self.config.MAX_SCENARIOS+1):
            cars = self.spawn_scenario()                       # 차량 배치
            result, _ = self.simulate_single(cars)

            if result == SimulationState.NotEnd:
                raise Exception(f'scenario not end in id:{scen_idx}')

            if scen_idx % self.config.LOG_INTERVAL == 0:
                avg_reward, success_rate = self.evaluate_policy(20)

                cur = time.time()
                log_scenario(scen_idx, avg_reward, success_rate, self.epsilon, cur - start_time)
                start_time = cur

            self.epsilon = max(self.config.EPS_END, self.epsilon - self.config.EPS_DECAY)
            libsumo.load(self.sim_cfg)


aim = AIM()

aim.start()