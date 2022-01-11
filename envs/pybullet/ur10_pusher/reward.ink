inkling "2.0"
using Number
using Math

# Type that represents the per-iteration state returned by simulator
type SimState {
    gripper_x:Number.Float32,
    gripper_y:Number.Float32,
    gripper_z:Number.Float32,

    gripper_vel_x:Number.Float32,
    gripper_vel_y:Number.Float32,
    gripper_vel_z:Number.Float32,

    target_x:Number.Float32,
    target_y:Number.Float32,
    target_z:Number.Float32,

    object_x:Number.Float32,
    object_y:Number.Float32,
    object_z:Number.Float32,

    object_vel_x:Number.Float32,
    object_vel_y:Number.Float32,
    object_vel_z:Number.Float32,

    dst_ot_x:Number.Float32,
    dst_ot_y:Number.Float32,
    dst_ot_z:Number.Float32,

    dst_og_x:Number.Float32,
    dst_og_y:Number.Float32,
    dst_og_z:Number.Float32,

    rew:Number.Float32,
    episode_rew:Number.Float32,
}


type SimConfig{
    episode_iteration_limit: Number.UInt64
}
# State that represents the input to the policy
type ObservableState {
	gripper_x:Number.Float32,
    gripper_y:Number.Float32,
    gripper_z:Number.Float32,

    gripper_vel_x:Number.Float32,
    gripper_vel_y:Number.Float32,
    gripper_vel_z:Number.Float32,

    target_x:Number.Float32,
    target_y:Number.Float32,
    target_z:Number.Float32,

    object_x:Number.Float32,
    object_y:Number.Float32,
    object_z:Number.Float32,

    object_vel_x:Number.Float32,
    object_vel_y:Number.Float32,
    object_vel_z:Number.Float32,

    dst_ot_x:Number.Float32,
    dst_ot_y:Number.Float32,
    dst_ot_z:Number.Float32,

    dst_og_x:Number.Float32,
    dst_og_y:Number.Float32,
    dst_og_z:Number.Float32
}

# Type that represents the per-iteration action accepted by the simulator
type SimAction {
    x_offset:Number.Float32,
    y_offset:Number.Float32,
    z_offset:Number.Float32

}

# Define a concept graph with a single concept
graph (input: ObservableState): SimAction {
    concept Pusher(input): SimAction {
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
                EpisodeIterationLimit: 700
            }
            lesson walking{
              scenario {
                    episode_iteration_limit: 700
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