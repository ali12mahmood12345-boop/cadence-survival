import math
import random

from world import SurvivalWorld
from brain import SurvivalBrain


TOTAL_EPISODES = 40
SHIFT_EPISODE = 21
MAX_STEPS = 300


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
    """
    Executes one action.

    NORMAL:
        F gives energy
        X causes damage

    SHIFT:
        F causes damage
        X gives energy
    """

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

    # ---------------------------------
    # NORMAL WORLD
    # ---------------------------------

    if not shifted:

        if world.agent == world.food:
            world.energy = min(
                100,
                world.energy + 30
            )

            world.score += 10
            event = "GOOD_TARGET"

            world.food = random_new_position(
                world,
                [world.agent, world.danger]
            )

        elif world.agent == world.danger:
            world.health -= 35
            world.score -= 10
            event = "BAD_TARGET"

            world.danger = random_new_position(
                world,
                [world.agent, world.food]
            )

    # ---------------------------------
    # SHIFTED WORLD
    # Meanings are swapped.
    # ---------------------------------

    else:

        if world.agent == world.food:
            world.health -= 35
            world.score -= 10
            event = "BAD_TARGET"

            world.food = random_new_position(
                world,
                [world.agent, world.danger]
            )

        elif world.agent == world.danger:
            world.energy = min(
                100,
                world.energy + 30
            )

            world.score += 10
            event = "GOOD_TARGET"

            world.danger = random_new_position(
                world,
                [world.agent, world.food]
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

    old_f_distance = distance(
        old_agent,
        old_food
    )

    new_f_distance = distance(
        world.agent,
        old_food
    )

    old_x_distance = distance(
        old_agent,
        old_danger
    )

    new_x_distance = distance(
        world.agent,
        old_danger
    )

    # ---------------------------------
    # NORMAL
    # F good, X bad
    # ---------------------------------

    if not shifted:

        if new_f_distance < old_f_distance:
            reward += 0.20
        elif new_f_distance > old_f_distance:
            reward -= 0.10

        if new_x_distance < old_x_distance:
            reward -= 0.15
        elif new_x_distance > old_x_distance:
            reward += 0.05

    # ---------------------------------
    # SHIFTED
    # X good, F bad
    # ---------------------------------

    else:

        if new_x_distance < old_x_distance:
            reward += 0.20
        elif new_x_distance > old_x_distance:
            reward -= 0.10

        if new_f_distance < old_f_distance:
            reward -= 0.15
        elif new_f_distance > old_f_distance:
            reward += 0.05

    if event == "GOOD_TARGET":
        reward += 2.0

    elif event == "BAD_TARGET":
        reward -= 2.0

    if world.agent == old_agent:
        reward -= 0.05

    if not world.alive:
        reward -= 2.0

    return reward


def run_episode(brain, episode, shifted):

    world = SurvivalWorld()

    reward = None
    steps = 0

    total_reward = 0.0
    good_targets = 0
    bad_targets = 0

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

        if event == "GOOD_TARGET":
            good_targets += 1

        elif event == "BAD_TARGET":
            bad_targets += 1

        total_reward += reward
        steps += 1

    final_senses = perception(world)

    brain.think(
        final_senses,
        reward=reward,
        done=True
    )

    return {
        "episode": episode,
        "reward": total_reward,
        "steps": steps,
        "good": good_targets,
        "bad": bad_targets,
    }


def avg(data, key):
    return sum(
        item[key] for item in data
    ) / len(data)


def show_block(title, data):

    print()
    print(title)
    print(
        "Average reward :",
        round(avg(data, "reward"), 2)
    )
    print(
        "Average steps  :",
        round(avg(data, "steps"), 2)
    )
    print(
        "Good targets   :",
        round(avg(data, "good"), 2)
    )
    print(
        "Bad targets    :",
        round(avg(data, "bad"), 2)
    )


def main():

    print()
    print("==============================================")
    print("       CADENCE // WORLD SHIFT V2")
    print("==============================================")
    print()
    print("ONE CONTINUOUS CADENCE BRAIN")
    print()
    print("EPISODES 01-20")
    print("F = ENERGY")
    print("X = DANGER")
    print()
    print("EPISODES 21-40")
    print("F = DANGER")
    print("X = ENERGY")
    print()
    print("THE BRAIN WILL NOT BE RESET.")
    print()

    brain = SurvivalBrain()

    results = []

    for episode in range(
        1,
        TOTAL_EPISODES + 1
    ):

        shifted = episode >= SHIFT_EPISODE

        if episode == SHIFT_EPISODE:

            print()
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            print(
                "             WORLD SHIFT DETECTED"
            )
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            print()
            print("TARGET MEANINGS HAVE SWAPPED")
            print("F -> DANGER")
            print("X -> ENERGY")
            print()
            print("SAME CADENCE BRAIN CONTINUES")
            print()

        result = run_episode(
            brain,
            episode,
            shifted
        )

        results.append(result)

        phase = (
            "SHIFT"
            if shifted
            else "NORMAL"
        )

        print(
            f"Episode {episode:02d} | "
            f"{phase:6s} | "
            f"Reward: {result['reward']:7.2f} | "
            f"Good: {result['good']:2d} | "
            f"Bad: {result['bad']:2d} | "
            f"Steps: {result['steps']:3d}"
        )

    # Last 5 episodes before shift
    before = results[15:20]

    # First 5 after shift
    shock = results[20:25]

    # Last 5 after shift
    late = results[35:40]

    print()
    print("==============================================")
    print("             ADAPTATION REPORT")
    print("==============================================")

    show_block(
        "BEFORE SHIFT - EPISODES 16-20",
        before
    )

    show_block(
        "SHIFT SHOCK - EPISODES 21-25",
        shock
    )

    show_block(
        "LATE SHIFT - EPISODES 36-40",
        late
    )

    shock_reward = avg(
        shock,
        "reward"
    )

    late_reward = avg(
        late,
        "reward"
    )

    recovery = (
        late_reward -
        shock_reward
    )

    shock_good = avg(
        shock,
        "good"
    )

    late_good = avg(
        late,
        "good"
    )

    print()
    print("----------------------------------------------")
    print(
        "REWARD RECOVERY :",
        round(recovery, 2)
    )

    print(
        "GOOD TARGET CHANGE :",
        round(
            late_good - shock_good,
            2
        )
    )

    if recovery > 0 and late_good > shock_good:
        print("RESULT : ADAPTATION SIGNAL OBSERVED")

    elif recovery > 0:
        print("RESULT : PARTIAL RECOVERY OBSERVED")

    else:
        print("RESULT : NO RECOVERY OBSERVED")

    print("----------------------------------------------")

    print()
    print("SAME BRAIN BEFORE + AFTER SHIFT")
    print("NO MODEL RESET AT EPISODE 21")
    print()
    print("==============================================")


if __name__ == "__main__":
    main()