import logging
from os import read
from typing import Any, Dict


from gym_connectors import BonsaiConnector, PyBulletSimulator
from tur10_env import UR10
log = logging.getLogger("reacher")

class Reacher(PyBulletSimulator):
    """ Implements the methods specific to Hopper environment
    """


    def __init__(self, iteration_limit=200, skip_frame=1):
        """ Initializes the Reacher environment
        """
        self.bonsai_state = None

        super().__init__(iteration_limit, skip_frame)
    def make_environment(self, headless):
        self._env = UR10(is_train=headless, is_dense=True)

    def gym_to_state(self, observation) -> Dict[str, Any]:
        """ Converts openai environment state to Bonsai state, as defined in inkling
        """
        observation = observation["observation"]
        self.bonsai_state = {
                             "gripper_x": float(observation[0]),
                             "gripper_y": float(observation[1]),
                             "gripper_z": float(observation[2]),
                             "target_x": float(observation[3]),
                             "target_y": float(observation[4]),
                             "target_z": float(observation[5]),
                             "dst_x": float(observation[6]),
                             "dst_y": float(observation[7]),
                             "dst_z": float(observation[8]),
                            
                             "rew": float(self.get_last_reward()),
                             "episode_rew": float(self.get_episode_reward()),
                             "progress": 0}
        '''
        11.01 1:15: The state value received from simulator id 293575395_10.244.17.127 contains one or more values that were provided as a string when a numeric value was expected or vice versa. This episode will be terminated. This indicates bug in your simulator or Inkling code.
        Invalid state data: { episode_rew: '-0.6206386420435054' }
        Previous state: { gripper_z: 1.0220847432714522, rew: 0.0, dst_z: 0.24733937391152072, episode_rew: 0.0, ... }
        Previous action: { x_offset: 0.3827580511569977, y_offset: -0.5512598156929016, z_offset: 0.5411996245384216 }
        '''
        return self.bonsai_state
        
    def action_to_gym(self, action: Dict[str, Any]):
        """ Converts Bonsai action type into openai environment action.
        """

        # Reacher environment expects an array of actions
        #print("!!!!!!!!!!!!!!!!!!!!", action)
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
    """ Creates a Reacher environment, passes it to the BonsaiConnector 
        that connects to the Bonsai service that can use it as a simulator  
    """
    logging.basicConfig()
    log = logging.getLogger("reacher")
    log.setLevel(level='INFO')

    reacher = Reacher()
    connector = BonsaiConnector(reacher)

    while connector.run():
        continue
