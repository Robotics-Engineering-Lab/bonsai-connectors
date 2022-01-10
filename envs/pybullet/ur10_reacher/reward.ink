inkling "2.0"
using Number
using Math

# Type that represents the per-iteration state returned by simulator
type SimState {
    gripper_x:number,
    gripper_y:number,
    gripper_z:number,
    target_x:number,
    target_y:number,
    target_z:number,
    dst_x:number,
    dst_y:number,
    dst_z:number,
    rew:number,
    episode_rew:number,
    progress:number

}


type SimConfig{
    episode_iteration_limit: Number.UInt64
}
# State that represents the input to the policy
type ObservableState {
	gripper_x:number,
    gripper_y:number,
    gripper_z:number,
    target_x:number,
    target_y:number,
    target_z:number,
    dst_x:number,
    dst_y:number,
    dst_z:number
}

# Type that represents the per-iteration action accepted by the simulator
type SimAction {
    x_offset:number,
    y_offset:number,
    z_offset:number

}

# Define a concept graph with a single concept
graph (input: ObservableState): SimAction {
    concept Reacher(input): SimAction {
        curriculum {
            # The source of training for this concept is a simulator
            # that takes an action as an input and outputs a state.
            source simulator (Action: SimAction,Config: SimConfig): SimState {
            }
            algorithm {
                Algorithm: "PPO",
                BatchSize : 10000,
                PolicyLearningRate:0.0001
            }
            reward GetReward

            training {
                EpisodeIterationLimit: 500
            }
            lesson walking{
              scenario {
                    episode_iteration_limit: 500
                }
            }
        }
        
    }
    
    
}

function GetReward(State: SimState, Action: SimAction) {

   # var electricity_cost = -0.1 * (Math.Abs(Action.central_joint_torque * State.theta_velocity) +     
   #    Math.Abs(Action.elbow_joint_torque * State.gama_velocity)) 
   #    -0.01 * (Math.Abs(Action.central_joint_torque) + Math.Abs(Action.elbow_joint_torque))   

    #var stuck_joint_cost = 0.0
    #if Math.Abs(Math.Abs(State.gama)-1) < 0.01
    #{
    # stuck_joint_cost = -0.1
    #}

    #var rew = State.progress + electricity_cost + stuck_joint_cost
    return State.rew
}