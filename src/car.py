import libsumo
import numpy as np
from enum import Enum, auto

from agent2 import AgentType

class CarConfigStatus(Enum):
    Unconfigured = auto()
    Configured = auto()
    Entered = auto()
    Exited = auto()

class Car:
    def __init__(self, vehID, routeID, typeID, depart, j_pos):
        self.vehID = vehID
        self.routeID = routeID
        self.typeID = typeID
        self.depart = depart
        self.jx, self.jy = j_pos
        self.state = []

        self.spawn()

        self.entry, self.exit = libsumo.vehicle.getRoute(vehID)[:2]
        
        self.agent_type: AgentType = self.pick_agent()
        self.config_status: CarConfigStatus = CarConfigStatus.Unconfigured

        self.action_history = None
        self.ep_trans = []

        self.pos = j_pos
        self.v = 0

    def pick_agent(self):
        left_routes = {("N2J","J2W"),("S2J","J2E"),("E2J","J2N"),("W2J","J2S")}
        right_routes = {("N2J","J2E"),("S2J","J2W"),("E2J","J2S"),("W2J","J2N")}

        if (self.entry, self.exit) in left_routes:
            return AgentType.LEFT
        elif (self.entry, self.exit) in right_routes:
            return AgentType.RIGHT
        else:
            return AgentType.STRAIGHT
        
    def subscribe(self):
        self.config_status = CarConfigStatus.Configured

        libsumo.vehicle.subscribe(self.vehID, [
            libsumo.constants.VAR_POSITION,
            libsumo.constants.VAR_SPEED,
        ])

    def spawn(self):
        libsumo.vehicle.add(vehID=self.vehID, routeID=self.routeID, typeID=self.typeID, depart=self.depart)
        libsumo.vehicle.setSpeedMode(self.vehID, 0)
        libsumo.vehicle.setSpeed(self.vehID, 0)

    def delete(self):
        libsumo.vehicle.unsubscribe(self.vehID)
        libsumo.vehicle.remove(self.vehID)

    def update_car_config_status(self, enter_r, exit_r):    
        if self.config_status == CarConfigStatus.Configured and self.d < enter_r:
            self.config_status = CarConfigStatus.Entered
        elif self.config_status == CarConfigStatus.Entered and self.d > exit_r:
            self.config_status = CarConfigStatus.Exited

        return self.config_status

    def set_data(self, data):
        self.x, self.y = data[libsumo.constants.VAR_POSITION]
        self.speed = data[libsumo.constants.VAR_SPEED]
        self.d    = np.linalg.norm([self.x-self.jx, self.y-self.jy])

    def set_speed(self, speed):
        libsumo.vehicle.setSpeed(self.vehID, speed)

    def set_action_history(self, state, action_index):
        self.action_history = state, action_index
    
    def get_action_history(self):
        if self.action_history is None:
            return None, None
        return self.action_history

    def get_config_status(self):
        return self.config_status
    
    def get_pos(self):
        return self.x, self.y


    def is_unconfigured(self):
        return self.config_status == CarConfigStatus.Unconfigured

    def is_exited(self):
        return self.config_status == CarConfigStatus.Exited
    
    def is_transition_empty(self):
        return len(self.ep_trans) == 0
