import math

from world import SurvivalWorld
from brain import SurvivalBrain


EPISODES = 20
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
    world
):
    reward = -0.02

    old_food_distance = distance(old_agent, old_food)
    new_food_distance = distance(world.agent, old_food)

    if new_food_distance < old_food_distance:
        reward += 0.20
    elif new_food_distance > old_food_distance:
        reward -= 0.10

    old_danger_distance = distance(old_agent, old_danger)
    new_danger_distance = distance(world.agent, old_danger)

    if new_danger_distance < old_danger_distance:
        reward -= 0.15
    elif new_danger_distance > old_danger_distance:
        reward += 0.05

    # Food moved after the agent collected it.
    if old_food != world.food:
        reward += 2.0

    if world.health < old_health:
        reward -= 2.0

    if world.agent == old_agent:
        reward -= 0.05

    if not world.alive:
        reward -= 2.0

    return reward


def run_episode(brain, episode_number):

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

    # Close the episode and return the final outcome
    # of the last executed action.
    final_senses = perception(world)

    brain.think(
        final_senses,
        reward=reward,
        done=True
    )

    food_eaten = max(0, world.score // 10)

    return {
        "episode": episode_number,
        "score": world.score,
        "food": food_eaten,
        "steps": steps,
    }


def main():

    print()
    print("============================================")
    print("       CADENCE // SURVIVAL TRAINING")
    print("============================================")
    print()
    print("ONE CADENCE BRAIN")
    print("20 EPISODES")
    print("BRAIN IS NOT RESET BETWEEN EPISODES")
    print()

    brain = SurvivalBrain()

    results = []

    for episode in range(1, EPISODES + 1):

        result = run_episode(
            brain,
            episode
        )

        results.append(result)

        print(
            f"Episode {episode:02d} | "
            f"Score: {result['score']:4d} | "
            f"Food: {result['food']:2d} | "
            f"Steps: {result['steps']:3d}"
        )

    first_half = results[:10]
    second_half = results[10:]

    first_score = sum(
        x["score"] for x in first_half
    ) / len(first_half)

    second_score = sum(
        x["score"] for x in second_half
    ) / len(second_half)

    first_food = sum(
        x["food"] for x in first_half
    ) / len(first_half)

    second_food = sum(
        x["food"] for x in second_half
    ) / len(second_half)

    first_steps = sum(
        x["steps"] for x in first_half
    ) / len(first_half)

    second_steps = sum(
        x["steps"] for x in second_half
    ) / len(second_half)

    print()
    print("============================================")
    print("               RESULTS")
    print("============================================")

    print()
    print("EPISODES 1-10")
    print("Average score :", round(first_score, 2))
    print("Average food  :", round(first_food, 2))
    print("Average steps :", round(first_steps, 2))

    print()
    print("EPISODES 11-20")
    print("Average score :", round(second_score, 2))
    print("Average food  :", round(second_food, 2))
    print("Average steps :", round(second_steps, 2))

    print()
    print("CHANGE")
    print(
        "Score change :",
        round(second_score - first_score, 2)
    )
    print(
        "Food change  :",
        round(second_food - first_food, 2)
    )
    print(
        "Steps change :",
        round(second_steps - first_steps, 2)
    )

    print()
    print("============================================")


if __name__ == "__main__":
    main()