from tqdm import tqdm
from matplotlib import pyplot as plt
def mini_batch_train(env, agents, max_episodes, max_steps=30, batch_size=30):
    episode_rewards = []

    for episode in range(max_episodes):
        # state = env.reset()
        episode_reward = 0
        
        highest_image=False
        lowest_image=False

        lowest_score=False
        highest_score=False
        
        
        for step in tqdm(range(max_steps)):
            state = env._next_observation()
            #get action
            actions = []
            for agent in agents:
              actions.append(agent.get_action(state).item())
            avg = sum(actions) / len(actions)
            
            # print(f"The scores each agents gave was {actions} with the average of {avg}\n")
            next_state, reward, done, _ = env.step(actions)

            #see if the score they gave is low or high and record
            if avg < lowest_score or lowest_score == False:
              lowest_image = state
              lowest_score = avg
            if avg > highest_score or highest_score == False:
              highest_image = state
              highest_score = avg

            #save it to the replay buffer
            for idx, agent in enumerate(agents):
              agent.replay_buffer.push(state, actions[idx], reward, next_state, done)

            #sum of the rewards in each step
            # print(f"reward in the step {step} is {reward}\n")
            episode_reward += reward

            if len(agents[0].replay_buffer) > batch_size:
              for agent in agents:
                agent.update(batch_size)  

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward) + "\n")
                break

            state = next_state
        
        env.reset()

        plt.imshow(highest_image.permute(1,2,0)/255)
        plt.show()
        plt.imshow(lowest_image.permute(1,2,0)/255)
        plt.show()

    return episode_rewards