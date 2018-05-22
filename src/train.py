import tensorflow as tf
import cv2
import sys
import random
import numpy as np
from collections import deque
from ple.games.flappybird import FlappyBird
from ple import PLE
import argparse
from config import *
import prepossessing
from model import Network
import os



def train(resume, reward_type):
    sess = tf.InteractiveSession()


    net = Network()

    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(net.cost())

    # open up a game state to communicate with emulator
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=True)
    p.init()
    reward = 0.0

    # store the previous observations in replay memory
    D = deque()

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")

    if resume == 1:
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    # get the first state by doing nothing and preprocess the image to 80x80x4

    actions = p.getActionSet()
    p.act(actions[1])

    s_t = prepossessing.transform_image(p.getScreenRGB())

    # start training

    total_reward = 0
    best_reward = 0
    actual_reward = 0

    best_rewards = []
    Qs = []
    rewards_game = []

    reward_sum = 0
    nb_reward_sum = 0

    epsilon = INITIAL_EPSILON
    t = 0

    train_writer = tf.summary.FileWriter('./logs/1/train ', sess.graph)


    while t < MAX_ITE:
        if p.game_over():
            p.reset_game()
            terminal = True
        else:
            terminal = False

        # choose an action epsilon greedily
        readout_t = net.readout.eval(feed_dict={net.s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        action = int(np.argmax(a_t))
        if action == 0:
            action = 1
        else:
            action = 0

        if reward_type == 0:

            r_t = p.act(actions[action])

            if r_t == -5:
                nb_reward_sum += 1
                reward_sum += actual_reward
                if nb_reward_sum % 10 == 0:
                    rewards_game.append(reward_sum / 10.0)
                    reward_sum = 0
                actual_reward = 0

            elif r_t > 0:
                actual_reward += 1
                total_reward += 1

            if actual_reward > best_reward:
                best_reward = actual_reward

        else:

            r_t = p.act(actions[action])

            if r_t == -5:
                nb_reward_sum += 1
                reward_sum += actual_reward
                if nb_reward_sum % 10 == 0:
                    rewards_game.append(reward_sum / 10.0)
                    reward_sum = 0
                actual_reward = 0

            elif r_t > 0:
                actual_reward += 1
                total_reward += 1

            if actual_reward > best_reward:
                best_reward = actual_reward

            if r_t == 0:
                r_t = 1


        s_t1 = prepossessing.transform_image_stacked(p.getScreenRGB(), s_t)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = net.readout.eval(feed_dict={net.s: s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA *
                                   np.max(readout_j1_batch[i]))

            # perform gradient step
            merge = tf.summary.merge_all()
            summary = train_step.run([merge], feed_dict={
                net.y: y_batch,
                net.a: a_batch,
                net.s: s_j_batch}
            )
            train_writer.add_summary(summary, counter)


        # update the old values
        s_t = s_t1
        t += 1

        if t % 100 == 0:
            Qs.append(np.max(readout_t))
            best_rewards.append(best_reward)

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'pyth/' + "bird" + '-dqn', global_step=t)
            np.save("q.npz", np.array(Qs))
            np.save("best.npz", np.array(best_rewards))
            np.save("game_rewards.npz", np.array(rewards_game))


        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state,
              "/ EPSILON", epsilon, "/ ACTION", action_index, "\ PIPES", total_reward, "\ BEST", best_reward, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--resume', default=0,
                        help="int, 1 if you want to continue a previous training else 0.", type=int)
    parser.add_argument('--video', default=1,
                        help="int, 1 if you want to continue a previous training else 0.", type=int)
    parser.add_argument('--reward', default=0,
                        help="int, 1 if you want a modified reward else 0.", type=int)

    args = parser.parse_args()

    if args.video == 0:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    train(args.resume, args.reward)
