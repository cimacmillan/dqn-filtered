from action_buffer import ActionBuffer
from action_buffer_non_uniform import ActionBufferNonUniform
from delta import ReplayFilter

import numpy as np
import random
import time


def random_frame():
    return (np.random.rand(84, 84, 1) * 255)


def random_action():
    return random.randint(0, 20)


def random_reward():
    return random.uniform(-20.0, 20.0)


def f(x):
    return x

print("Benchmarking")

SIZE = 100000
replay_filter = ReplayFilter((84, 84), 0, 0, 100)
action_buffer = ActionBuffer(SIZE, 4, replay_filter)
non_uniform_action_buffer = ActionBufferNonUniform(SIZE, 4, replay_filter, f)

for i in range(110000):
    action_buffer.store_frame(random_frame())
    action_buffer.possibly_store_effect(random_action(), random_reward(), False)
    non_uniform_action_buffer.store_frame(random_frame())
    non_uniform_action_buffer.possibly_store_effect(random_action(), random_reward(), False)

non_uniform_action_buffer.sample(32)


STEPS = 1000
x = time.time()
for i in range(STEPS):
    non_uniform_action_buffer.store_frame(random_frame())
    non_uniform_action_buffer.possibly_store_effect(random_action(), random_reward(), False)
    sample = i % 4
    if sample is 0:
        non_uniform_action_buffer.sample(32)

print("Simu: non Uniform", (time.time() - x))

# -------------------
INITIALISATION = 1000

x = time.time()
for i in range(INITIALISATION):
    action_buffer.store_frame(random_frame())
    action_buffer.possibly_store_effect(random_action(), random_reward(), False)


action_total = time.time() - x
print("Init - Action Buffer:", (time.time() - x))

x = time.time()
for i in range(INITIALISATION):
    non_uniform_action_buffer.store_frame(random_frame())
    non_uniform_action_buffer.possibly_store_effect(random_action(), random_reward(), False)


non_uniform_total = time.time() - x
print("Init - Non-Uniform Action Buffer:", (time.time() - x))


# -------------------
BATCH = 32
SAMPLE = 250

x = time.time()
for i in range(SAMPLE):
    action_buffer.sample(BATCH)

action_total += time.time() - x
print("Sample - Action Buffer:", (time.time() - x))

x = time.time()
for i in range(SAMPLE):
    non_uniform_action_buffer.sample(BATCH)

non_uniform_total += time.time() - x
print("Sample - Non-Uniform Action Buffer:", (time.time() - x))


print("Action Buffer Total: ", action_total)
print("Non-Uniform total: ", non_uniform_total)

