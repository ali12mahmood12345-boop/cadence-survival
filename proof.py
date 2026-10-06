import math
import random

from world import SurvivalWorld
from brain import SurvivalBrain


PRE_EPISODES = 20
POST_EPISODES = 40
MAX_STEPS = 300

RUN_SEEDS = [
    1001,
    2002,
    3003,
    4004,
    5005,
]


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


def random_new_position(world, avoid):
    while True:
        pos = [
            random.randint(0, world.size - 1),
            random.randint(0, world.size - 1),
        ]

        if pos not in avoid:
            return pos


def apply_action(world, action, shifted):

    if not world.alive:
        return None

    x, y = world.agent

    if action == "UP":
        y -= 1
    elif action == "DOWN":
        y += 1
    elif action == "LEFT":
        x -= 1
    elif action == "RIGHT":
        x += 1

    x = max(0, min(world.size - 1, x))
    y = max(0, min(world.size - 1, y))

    world.agent = [x, y]

    world.energy -= 1
    world.steps += 1

    event = None

    # NORMAL WORLD
    # F = good
    # X = bad
    if not shifted:

        if world.agent == world.food:

            world.energy = min(
                100,
                world.energy + 30
            )

            event = "GOOD"

            world.food = random_new_position(
                world,
                [world.agent, world.danger]
            )

        elif world.agent == world.danger:

            world.health -= 35
            event = "BAD"

            world.danger = random_new_position(
                world,
                [world.agent, world.food]
            )

    # SHIFTED WORLD
    # X = good
    # F = bad
    else:

        if world.agent == world.danger:

            world.energy = min(
                100,
                world.energy + 30
            )

            event = "GOOD"

            world.danger = random_new_position(
                world,
                [world.agent, world.food]
            )

        elif world.agent == world.food:

            world.health -= 35
            event = "BAD"

            world.food = random_new_position(
                world,
                [world.agent, world.danger]
            )

    if world.energy <= 0:
        world.health -= 5

    if world.health <= 0:
        world.alive = False

    return event


def calculate_reward(
    old_agent,
    old_food,
    old_danger,
    world,
    shifted,
    event
):

    reward = -0.02

    old_f = distance(
        old_agent,
        old_food
    )

    new_f = distance(
        world.agent,
        old_food
    )

    old_x = distance(
        old_agent,
        old_danger
    )

    new_x = distance(
        world.agent,
        old_danger
    )

    if not shifted:

        if new_f < old_f:
            reward += 0.20
        elif new_f > old_f:
            reward -= 0.10

        if new_x < old_x:
            reward -= 0.15
        elif new_x > old_x:
            reward += 0.05

    else:

        if new_x < old_x:
            reward += 0.20
        elif new_x > old_x:
            reward -= 0.10

        if new_f < old_f:
            reward -= 0.15
        elif new_f > old_f:
            reward += 0.05

    if event == "GOOD":
        reward += 2.0

    elif event == "BAD":
        reward -= 2.0

    if world.agent == old_agent:
        reward -= 0.05

    if not world.alive:
        reward -= 2.0

    return reward


def run_episode(brain, shifted, world_seed):

    random.seed(world_seed)

    world = SurvivalWorld()

    reward = None
    total_reward = 0.0
    steps = 0
    good = 0
    bad = 0

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

        event = apply_action(
            world,
            action,
            shifted
        )

        reward = calculate_reward(
            old_agent,
            old_food,
            old_danger,
            world,
            shifted,
            event
        )

        if event == "GOOD":
            good += 1

        elif event == "BAD":
            bad += 1

        total_reward += reward
        steps += 1

    brain.think(
        perception(world),
        reward=reward,
        done=True
    )

    return {
        "reward": total_reward,
        "steps": steps,
        "good": good,
        "bad": bad,
    }


def average(items, key):
    return sum(
        item[key]
        for item in items
    ) / len(items)


def run_condition(run_seed, reset_at_shift):

    random.seed(run_seed)

    brain = SurvivalBrain()

    pre_results = []
    post_results = []

    # Same deterministic world sequence
    # for both conditions.
    world_seeds = [
        run_seed + i * 17
        for i in range(
            PRE_EPISODES + POST_EPISODES
        )
    ]

    # --------------------------------
    # NORMAL WORLD
    # --------------------------------

    for episode in range(PRE_EPISODES):

        result = run_episode(
            brain,
            shifted=False,
            world_seed=world_seeds[episode]
        )

        pre_results.append(result)

    # --------------------------------
    # WORLD SHIFT
    # --------------------------------

    if reset_at_shift:
        brain = SurvivalBrain()

    for episode in range(POST_EPISODES):

        result = run_episode(
            brain,
            shifted=True,
            world_seed=world_seeds[
                PRE_EPISODES + episode
            ]
        )

        post_results.append(result)

    return {
        "pre": pre_results,
        "post": post_results,
    }


def summarize(result):

    pre_late = result["pre"][-5:]

    shock = result["post"][:5]

    late = result["post"][-10:]

    return {
        "before_reward":
            average(pre_late, "reward"),

        "shock_reward":
            average(shock, "reward"),

        "late_reward":
            average(late, "reward"),

        "shock_good":
            average(shock, "good"),

        "late_good":
            average(late, "good"),

        "shock_steps":
            average(shock, "steps"),

        "late_steps":
            average(late, "steps"),
    }


def mean(values):
    return sum(values) / len(values)


def main():

    print()
    print("==============================================")
    print("       CADENCE // CONTROLLED PROOF TEST")
    print("==============================================")
    print()

    print("5 FIXED RUN SEEDS")
    print("SAME WORLD SEQUENCE FOR BOTH CONDITIONS")
    print()

    print("A = CONTINUOUS CADENCE BRAIN")
    print("B = RESET BRAIN AT WORLD SHIFT")
    print()

    continuous_summaries = []
    reset_summaries = []

    for index, seed in enumerate(
        RUN_SEEDS,
        start=1
    ):

        print(
            f"RUN {index}/5 - SEED {seed}"
        )

        continuous = run_condition(
            seed,
            reset_at_shift=False
        )

        reset = run_condition(
            seed,
            reset_at_shift=True
        )

        continuous_summary = summarize(
            continuous
        )

        reset_summary = summarize(
            reset
        )

        continuous_summaries.append(
            continuous_summary
        )

        reset_summaries.append(
            reset_summary
        )

        print(
            "  CONTINUOUS late reward:",
            round(
                continuous_summary[
                    "late_reward"
                ],
                2
            )
        )

        print(
            "  RESET      late reward:",
            round(
                reset_summary[
                    "late_reward"
                ],
                2
            )
        )

        print()

    # ========================================
    # FINAL AGGREGATE RESULTS
    # ========================================

    continuous_late_reward = mean([
        x["late_reward"]
        for x in continuous_summaries
    ])

    reset_late_reward = mean([
        x["late_reward"]
        for x in reset_summaries
    ])

    continuous_shock_reward = mean([
        x["shock_reward"]
        for x in continuous_summaries
    ])

    reset_shock_reward = mean([
        x["shock_reward"]
        for x in reset_summaries
    ])

    continuous_late_good = mean([
        x["late_good"]
        for x in continuous_summaries
    ])

    reset_late_good = mean([
        x["late_good"]
        for x in reset_summaries
    ])

    continuous_late_steps = mean([
        x["late_steps"]
        for x in continuous_summaries
    ])

    reset_late_steps = mean([
        x["late_steps"]
        for x in reset_summaries
    ])

    continuous_recovery = (
        continuous_late_reward -
        continuous_shock_reward
    )

    reset_recovery = (
        reset_late_reward -
        reset_shock_reward
    )

    print()
    print("==============================================")
    print("              PROOF REPORT")
    print("==============================================")
    print()

    print("CONTINUOUS CADENCE")
    print(
        "Shock reward :",
        round(
            continuous_shock_reward,
            2
        )
    )
    print(
        "Late reward  :",
        round(
            continuous_late_reward,
            2
        )
    )
    print(
        "Recovery     :",
        round(
            continuous_recovery,
            2
        )
    )
    print(
        "Late good    :",
        round(
            continuous_late_good,
            2
        )
    )
    print(
        "Late steps   :",
        round(
            continuous_late_steps,
            2
        )
    )

    print()

    print("RESET CONTROL")
    print(
        "Shock reward :",
        round(
            reset_shock_reward,
            2
        )
    )
    print(
        "Late reward  :",
        round(
            reset_late_reward,
            2
        )
    )
    print(
        "Recovery     :",
        round(
            reset_recovery,
            2
        )
    )
    print(
        "Late good    :",
        round(
            reset_late_good,
            2
        )
    )
    print(
        "Late steps   :",
        round(
            reset_late_steps,
            2
        )
    )

    print()
    print("----------------------------------------------")

    advantage = (
        continuous_late_reward -
        reset_late_reward
    )

    print(
        "CONTINUOUS LATE REWARD ADVANTAGE:",
        round(
            advantage,
            2
        )
    )

    wins = 0

    for continuous, reset in zip(
        continuous_summaries,
        reset_summaries
    ):

        if (
            continuous["late_reward"]
            >
            reset["late_reward"]
        ):
            wins += 1

    print(
        "CONTINUOUS WINS:",
        f"{wins}/5"
    )

    print("----------------------------------------------")
    print()

    if advantage > 0 and wins >= 3:

        print(
            "RESULT: CONTINUOUS MEMORY ADVANTAGE OBSERVED"
        )

    else:

        print(
            "RESULT: NO CONSISTENT CONTINUOUS ADVANTAGE"
        )

    print()
    print("==============================================")


if __name__ == "__main__":
    main()