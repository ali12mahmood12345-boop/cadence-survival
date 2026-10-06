import math

from world import SurvivalWorld
from brain import SurvivalBrain


TOTAL_EPISODES = 30
SHIFT_EPISODE = 16
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


def calculate_reward(
    old_agent,
    old_food,
    old_danger,
    old_health,
    world,
    shifted
):
    reward = -0.02

    old_food_distance = distance(old_agent, old_food)
    new_food_distance = distance(world.agent, old_food)

    # ==========================================
    # BEFORE WORLD SHIFT
    # Food is attractive.
    # ==========================================

    if not shifted:

        if new_food_distance < old_food_distance:
            reward += 0.20

        elif new_food_distance > old_food_distance:
            reward -= 0.10

        if old_food != world.food:
            reward += 2.0

    # ==========================================
    # AFTER WORLD SHIFT
    # The learned food rule is reversed.
    # ==========================================

    else:

        if new_food_distance > old_food_distance:
            reward += 0.20

        elif new_food_distance < old_food_distance:
            reward -= 0.10

        if old_food != world.food:
            reward -= 2.0

    # Danger remains dangerous in both worlds.

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

    if world.health < old_health:
        reward -= 2.0

    if world.agent == old_agent:
        reward -= 0.05

    if not world.alive:
        reward -= 2.0

    return reward


def run_episode(brain, episode_number, shifted):

    world = SurvivalWorld()

    reward = None
    steps = 0

    total_reward = 0.0

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
            world,
            shifted
        )

        total_reward += reward
        steps += 1

    final_senses = perception(world)

    brain.think(
        final_senses,
        reward=reward,
        done=True
    )

    return {
        "episode": episode_number,
        "score": world.score,
        "steps": steps,
        "reward": total_reward,
    }


def average(data, key):
    return sum(x[key] for x in data) / len(data)


def main():

    print()
    print("============================================")
    print("     CADENCE // WORLD SHIFT EXPERIMENT")
    print("============================================")
    print()
    print("ONE CONTINUOUS CADENCE BRAIN")
    print("30 EPISODES")
    print()
    print("EPISODES 01-15 : FOOD = GOOD")
    print("EPISODES 16-30 : FOOD RULE REVERSED")
    print()
    print("THE BRAIN WILL NOT BE RESET.")
    print()

    brain = SurvivalBrain()

    results = []

    for episode in range(1, TOTAL_EPISODES + 1):

        shifted = episode >= SHIFT_EPISODE

        if episode == SHIFT_EPISODE:

            print()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("          WORLD SHIFT DETECTED")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print()
            print("FOOD REWARD HAS REVERSED")
            print("SAME CADENCE BRAIN CONTINUES")
            print()

        result = run_episode(
            brain,
            episode,
            shifted
        )

        results.append(result)

        phase = "SHIFT" if shifted else "NORMAL"

        print(
            f"Episode {episode:02d} | "
            f"{phase:6s} | "
            f"World Score: {result['score']:4d} | "
            f"Reward: {result['reward']:7.2f} | "
            f"Steps: {result['steps']:3d}"
        )

    # ------------------------------------------
    # Compare performance around the shift.
    # ------------------------------------------

    learned_normal = results[10:15]
    shock_period = results[15:20]
    adapted_period = results[25:30]

    normal_reward = average(
        learned_normal,
        "reward"
    )

    shock_reward = average(
        shock_period,
        "reward"
    )

    adapted_reward = average(
        adapted_period,
        "reward"
    )

    normal_steps = average(
        learned_normal,
        "steps"
    )

    shock_steps = average(
        shock_period,
        "steps"
    )

    adapted_steps = average(
        adapted_period,
        "steps"
    )

    print()
    print("============================================")
    print("          ADAPTATION REPORT")
    print("============================================")
    print()

    print("BEFORE SHIFT")
    print("Episodes 11-15")
    print(
        "Average reward :",
        round(normal_reward, 2)
    )
    print(
        "Average steps  :",
        round(normal_steps, 2)
    )

    print()
    print("IMMEDIATELY AFTER SHIFT")
    print("Episodes 16-20")
    print(
        "Average reward :",
        round(shock_reward, 2)
    )
    print(
        "Average steps  :",
        round(shock_steps, 2)
    )

    print()
    print("LATE AFTER SHIFT")
    print("Episodes 26-30")
    print(
        "Average reward :",
        round(adapted_reward, 2)
    )
    print(
        "Average steps  :",
        round(adapted_steps, 2)
    )

    recovery = adapted_reward - shock_reward

    print()
    print("--------------------------------------------")
    print(
        "POST-SHIFT REWARD RECOVERY :",
        round(recovery, 2)
    )

    if recovery > 0:
        print("RESULT : RECOVERY OBSERVED")
    else:
        print("RESULT : NO RECOVERY OBSERVED")

    print("--------------------------------------------")
    print()
    print("IMPORTANT:")
    print("The same Cadence brain was used")
    print("before and after the world shift.")
    print()
    print("============================================")


if __name__ == "__main__":
    main()