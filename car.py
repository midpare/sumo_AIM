import traci, traci.constants
from agent import AgentType
import numpy as np
from enum import Enum, auto

class CarConfigStatus(Enum):
    Unconfigured = auto()
    Configured = auto()
    Entered = auto()
    Exited = auto()

class Car:
    def __init__(self, vehID, routeID, typeID, depart, j_pos):
        traci.vehicle.add(vehID=vehID, routeID=routeID, typeID=typeID, depart=depart)
        traci.vehicle.setSpeedMode(vehID, 0)  # 차량 속도 모드 설정
        traci.vehicle.setSpeed(vehID, 0.0)    # 초기

        self.vehID = vehID
        self.routeID = routeID
        self.jx, self.jy = j_pos

        self.agent_type: AgentType
        self.config_status: CarConfigStatus = CarConfigStatus.Unconfigured

        self.action_history = None
        self.ep_trans = []

        self.pos = j_pos
        self.v = 0
        self.h = 0
        self.d = 0

        self.pick_agent(*traci.vehicle.getRoute(vehID)[:2])

    def pick_agent(self, entry, exit):
        left_routes = {("N2J","J2W"),("S2J","J2E"),("E2J","J2N"),("W2J","J2S")}
        right_routes = {("N2J","J2E"),("S2J","J2W"),("E2J","J2S"),("W2J","J2N")}

        if (entry, exit) in left_routes:
            self.agent_type = AgentType.LEFT
        elif (entry, exit) in right_routes:
            self.agent_type = AgentType.RIGHT
        else:
            self.agent_type = AgentType.STRAIGHT
        
    def subscribe(self):
        self.config_status = CarConfigStatus.Configured

        traci.vehicle.subscribe(self.vehID, [
            traci.constants.VAR_POSITION,
            traci.constants.VAR_SPEED,
            traci.constants.VAR_ANGLE,
        ])

    def delete(self):
        traci.vehicle.unsubscribe(self.vehID)
        traci.vehicle.remove(self.vehID)

    def update_car_config_status(self, enter_r, exit_r):       
        if self.config_status == CarConfigStatus.Configured and self.d < enter_r:
            self.config_status = CarConfigStatus.Entered
        elif self.config_status == CarConfigStatus.Entered and self.d > exit_r:
            self.config_status = CarConfigStatus.Exited

        return self.config_status

    def set_data(self, data):        
        self.x, self.y = data[traci.constants.VAR_POSITION]
        self.s = data[traci.constants.VAR_SPEED]
        self.h = data[traci.constants.VAR_ANGLE]
        self.d    = np.linalg.norm([self.x-self.jx, self.y-self.jy])

    def set_speed(self, speed):
        traci.vehicle.setSpeed(self.vehID, speed)

    def set_action_history(self, state, action_index):
        self.action_history = state, action_index

    def get_action_history(self):
        if self.action_history is None:
            return None, None
        return self.action_history

    def get_config_status(self):
        return self.config_status
    
    def get_state(self):
        return [self.x, self.y, self.s, self.h, self.d]

    def get_normalized_state(self):
        nomalized_x = (self.x-self.jx) / 200.0 + 0.5  # 교차로 중심 기준으로 정규화
        nomalized_y = (self.y-self.jy) / 200.0 + 0.5

        x = np.clip(nomalized_x, 0, 1.0)
        y = np.clip(nomalized_y, 0, 1.0)
        s = np.clip(self.s / 13.0, 0, 1.0)
        h = np.clip(self.h / 360.0, 0, 1.0)  # 각도 정규화(0~1)
        d = np.clip(self.d / 200.0, 0, 1.0)

        return [x, y, s, h, d]
    
    def is_unconfigured(self):
        return self.config_status == CarConfigStatus.Unconfigured

    def is_exited(self):
        return self.config_status == CarConfigStatus.Exited
    
    def is_transition_empty(self):
        return len(self.ep_trans) == 0
