import gym
from bear.env_manager.manager import SubprocEnvManager
from bear.controller.base import Controller
from bear.data.experience import RolloutExperienceBuffer
from examples.cartpole.adapter import CartPoleAdapter
from examples.cartpole.exploration import CartPoleExplorer
from examples.cartpole.model import MLPActor, MLPExtractor, MLPCritic
from examples.cartpole.model import ACModel
from bear.policy.modelfree.a2c import A2CPolicy
from bear.policy.modelfree.ppo_clip_a2c import PPO2A2C
from torch.distributions import Categorical
import torch
import numpy as np
import time
import os
from tensorboardX import SummaryWriter


def run_evaluate_episodes(policy, adapter, explorer):
    avg_reward = []
    eval_episodes = 5
    test_env = gym.make('CartPole-v0')
    test_env.seed(int(time.time()))
    for _ in range(eval_episodes):
        obs = test_env.reset()
        done = False
        episode_reward = 0.
        while not done:
            obs = adapter.adapt_obs(obs)
            act_logits = policy.choose_action(np.asarray([obs]))

            action = explorer.reverse_action(act_logits)
            next_obs, reward, done, info = test_env.step(action[0])
            test_env.render()
            episode_reward += reward
            obs = next_obs
        avg_reward.append(episode_reward)

    test_env.close()
    print("eval episode rewards:  ", avg_reward)
    return np.mean(avg_reward), np.min(avg_reward), np.max(avg_reward)


if __name__ == "__main__":
    logger = SummaryWriter('./logs/')
    training_num = 6
    seed = 10
    max_buffer_size = 1000000
    feat_size = 4
    hidden_size = 128
    act_size = 2
    lr = 1e-2
    epochs = 1000000
    best_avg_reward = 0.

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    adapter = CartPoleAdapter()
    explorer = CartPoleExplorer()
    buffer = RolloutExperienceBuffer(max_size=max_buffer_size)

    # extractor = MLPExtractor(feat_shape=feat_size, hidden_shape=hidden_size)
    # actor_model = MLPActor(extractor,
    #                        hidden_shape=hidden_size,
    #                        act_shape=act_size)
    # critic_model = MLPCritic(extractor,
    #                          hidden_shape=hidden_size)
    # actor_model.load_state_dict(torch.load('./checkpoints/actor1.pt', map_location=torch.device('cpu')))
    model = ACModel(model_type="ac",feat_size=feat_size,act_size=act_size,hidden_size=hidden_size)

    optim = torch.optim.Adam(set(model.actor.parameters()).union(set(model.critic.parameters())),
                             lr=lr)
    dist_fn = Categorical

    # a2c_policy = A2CPolicy(model, optim, dist_fn,
    #                        value_loss_weight=0.5, entropy_loss_weight=0.001,
    #                        reward_norm=False, device=device)

    a2c_policy = PPO2A2C(model, optim, dist_fn,
                         value_loss_weight=0.5, entropy_loss_weight=0.001,
                         returns_norm=True)

    train_envs_manager = SubprocEnvManager([lambda: gym.make("CartPole-v0") for _ in range(training_num)])
    train_envs_manager.seed(seed)

    controller = Controller(a2c_policy, adapter, explorer, train_envs_manager, buffer)

    for epoch in range(epochs):
        controller.collect(n_episode=1)
        loss = controller.learn()

        if epoch % 100 == 0:
            print('---------------epoch: ', epoch, '--------------')
            avg_reward, min_reward, max_reward = run_evaluate_episodes(a2c_policy, adapter, explorer)
            print("avg_reward: ", avg_reward)
            print("loss: ", loss)
            logger.add_scalar("evaluate_avg5_reward", avg_reward, epoch)
            logger.add_scalar("evaluate_min5_reward", min_reward, epoch)
            logger.add_scalar("evaluate_max5_reward", max_reward, epoch)
            for k, v in loss.items():
                logger.add_scalar(k, v, epoch)
            if avg_reward >= best_avg_reward:
                best_avg_reward = avg_reward
                # torch.save(actor_model.state_dict(), os.path.join('./checkpoints', 'actor1.pt'))
                print('best eval avg reward up to ', best_avg_reward)

    train_envs_manager.close()
