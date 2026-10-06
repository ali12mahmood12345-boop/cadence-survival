import random
import math
import numpy as np

from cadence import Brain, LearnerConfig, ActorCriticConfig
from world import SurvivalWorld


ACTIONS = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "WAIT",
]

EPISODES = 20
MAX_STEPS = 300

# Every mode gets exactly the same worlds.
WORLD_SEEDS = [
    101, 102, 103, 104, 105,
    106, 107, 108, 109, 110,
    111, 112, 113, 114, 115,
    116, 117, 118, 119, 120,
]


class TestBrain:

    def __init__(self, reward_mode):

        self.reward_mode = reward_mode

        learning = LearnerConfig(
            beta=0.1,
            temperature=0.2,
            tolerance=3e-3,
            free_steps=512,
            nudged_steps=12,
            eta=0.003,
            momentum=0.9,
            normalize=0.99,
            normalize_floor=1e-4,
        )

        reward_config = ActorCriticConfig(
            gamma=0.97,
            lam=0.9,
            eta=0.001,
            eta_critic=0.3,
            momentum=0.9,
            normalize=0.99,
        )

        self.brain = Brain.compose(
            5,
            len(ACTIONS),
            seed=42,
            learning=learning,
            reward=reward_config,
        )

        self.first_step = True

    def transform_reward(self, reward):

        if reward is None:
            return 0.0

        reward = float(reward)

        if self.reward_mode == "RAW":
            return reward

        if self.reward_mode == "SCALED":
            return reward * 0.25

        if self.reward_mode == "SIGN":
            return float(np.sign(reward))

        return reward

    def think(self, perception, reward=None, done=False):

        x = np.asarray(
            [perception],
            dtype=np.float64
        )

        try:

            if self.first_step:

                action_index = int(
                    self.brain.step(x)[0]
                )

                self.first_step = False

            else:

                signal = self.transform_reward(reward)

                action_index = int(
                    self.brain.step(
                        x,
                        reward=np.asarray(
                            [signal],
                            dtype=np.float64
                        ),
                        done=np.asarray(
                            [bool(done)]
                        )
                    )[0]
                )

            return ACTIONS[action_index]

        except RuntimeError:
            return "WAIT"


def perception(world):

    ax, ay = world.agent
    fx, fy = world.food
    dx, dy = world.danger

    size = max(1, world.size - 1)

    return [
        (fx - ax) / size,
        (fy - ay) / size,
        (dx - ax) / size,
        (dy - ay) / size,
        world.energy / 100.0,
    ]


def distance(a, b):

    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def calculate_reward(
    old_agent,
    old_food,
    old_danger,
    old_health,
    world
):

    reward = -0.02

    old_food_distance = distance(
        old_agent,
        old_food
    )

    new_food_distance = distance(
        world.agent,
        old_food
    )

    if new_food_distance < old_food_distance:
        reward += 0.20

    elif new_food_distance > old_food_distance:
        reward -= 0.10

    old_danger_distance = distance(
        old_agent,
        old_danger
    )

    new_danger_distance = distance(
        world.agent,
        old_danger
    )

    if new_danger_distance < old_danger_distance:
        reward -= 0.15

    elif new_danger_distance > old_danger_distance:
        reward += 0.05

    if old_food != world.food:
        reward += 2.0

    if world.health < old_health:
        reward -= 2.0

    if world.agent == old_agent:
        reward -= 0.05

    if not world.alive:
        reward -= 2.0

    return reward


def run_episode(brain, seed):

    # This makes the world reproducible.
    random.seed(seed)

    world = SurvivalWorld()

    reward = None
    steps = 0

    while world.alive and steps < MAX_STEPS:

        senses = perception(world)

        action = brain.think(
            senses,
            reward=reward,
            done=False
        )

        old_agent = world.agent.copy()
        old_food = world.food.copy()
        old_danger = world.danger.copy()
        old_health = world.health

        world.move(action)

        reward = calculate_reward(
            old_agent,
            old_food,
            old_danger,
            old_health,
            world
        )

        steps += 1

    final_senses = perception(world)

    brain.think(
        final_senses,
        reward=reward,
        done=True
    )

    return {
        "score": world.score,
        "food": max(0, world.score // 10),
        "steps": steps,
    }


def run_mode(mode):

    brain = TestBrain(mode)

    results = []

    print()
    print("--------------------------------------------")
    print("TESTING:", mode)
    print("--------------------------------------------")

    for number, seed in enumerate(
        WORLD_SEEDS,
        start=1
    ):

        result = run_episode(
            brain,
            seed
        )

        results.append(result)

        print(
            f"{mode:6s} | "
            f"Episode {number:02d} | "
            f"Score: {result['score']:4d} | "
            f"Food: {result['food']:2d} | "
            f"Steps: {result['steps']:3d}"
        )

    return results


def average(results, key):

    return sum(
        result[key]
        for result in results
    ) / len(results)


def main():

    print()
    print("============================================")
    print("       CADENCE // REWARD BENCHMARK")
    print("============================================")
    print()
    print("SAME WORLD SEEDS FOR EVERY MODE")
    print("SAME CADENCE BRAIN SEED")
    print()

    modes = [
        "RAW",
        "SCALED",
        "SIGN",
    ]

    all_results = {}

    for mode in modes:

        all_results[mode] = run_mode(mode)

    print()
    print("============================================")
    print("              FINAL COMPARISON")
    print("============================================")
    print()

    for mode in modes:

        results = all_results[mode]

        first = results[:10]
        last = results[10:]

        first_score = average(
            first,
            "score"
        )

        last_score = average(
            last,
            "score"
        )

        first_food = average(
            first,
            "food"
        )

        last_food = average(
            last,
            "food"
        )

        first_steps = average(
            first,
            "steps"
        )

        last_steps = average(
            last,
            "steps"
        )

        print(mode)
        print(
            "  Score :",
            round(first_score, 2),
            "->",
            round(last_score, 2)
        )

        print(
            "  Food  :",
            round(first_food, 2),
            "->",
            round(last_food, 2)
        )

        print(
            "  Steps :",
            round(first_steps, 2),
            "->",
            round(last_steps, 2)
        )

        print(
            "  Score improvement:",
            round(
                last_score - first_score,
                2
            )
        )

        print()

    print("============================================")
    print("BENCHMARK COMPLETE")
    print("============================================")


if __name__ == "__main__":
    main()