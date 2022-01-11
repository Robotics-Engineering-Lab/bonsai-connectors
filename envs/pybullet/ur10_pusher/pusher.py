import logging
from os import read
from typing import Any, Dict


from gym_connectors import BonsaiConnector, PyBulletSimulator
from tur10_env import UR10
log = logging.getLogger("pusher")

class Pusher(PyBulletSimulator):
    """ Implements the methods specific to Hopper environment
    """


    def __init__(self, iteration_limit=200, skip_frame=1):
        """ Initializes the Pusher environment
        """
        self.bonsai_state = None

        super().__init__(iteration_limit, skip_frame)
    def make_environment(self, headless):
        self._env = UR10(is_train=headless, is_dense=True)

    def gym_to_state(self, observation) -> Dict[str, Any]:
        """ Converts openai environment state to Bonsai state, as defined in inkling
        """
        observation = observation["observation"]
        keys = ("gripper_x", "gripper_y", "gripper_z", 
                "gripper_vel_x", "gripper_vel_y", "gripper_vel_z", 
                "target_x", "target_y", "target_z",
                "object_x", "object_y", "object_z", 
                "object_vel_x", "object_vel_y", "object_vel_z",
                "dst_ot_x", "dst_ot_y", "dst_ot_z",
                "dst_og_x", "dst_og_y", "dst_og_z"
                )

        self.bonsai_state = dict()
        for index, item in enumerate(observation):
            key = "STATE_" + str(index) if index >= len(keys) else keys[index]
            self.bonsai_state[key] = float(item)

        # info and reward func params, necessary
        self.bonsai_state["rew"] = float(self.get_last_reward())
        self.bonsai_state["episode_rew"] = float(self.get_episode_reward())
        
        return self.bonsai_state
        
    def action_to_gym(self, action: Dict[str, Any]):
        """ Converts Bonsai action type into openai environment action.
        """
        
        # Pusher environment expects an array of actions
        return [action['x_offset'], action['y_offset'], action['z_offset']]

    def get_state(self) -> Dict[str, Any]:
        """ Returns the current state of the environment
        """
        log.debug('get_state: {}'.format(self.bonsai_state))
        return self.bonsai_state

    def episode_start(self, config: Dict[str, Any] = None) -> None:
        """Reset the prev_potential at the beginning of each episode
        """
        self.prev_potential = None    

        super().episode_start(config)


if __name__ == "__main__":
    """ Creates a Pusher environment, passes it to the BonsaiConnector 
        that connects to the Bonsai service that can use it as a simulator  
    """
    logging.basicConfig()
    log = logging.getLogger("pusher")
    log.setLevel(level='INFO')

    pusher = Pusher()
    connector = BonsaiConnector(pusher)

    while connector.run():
        continue
