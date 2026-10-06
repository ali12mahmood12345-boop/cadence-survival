import time
import math

from world import SurvivalWorld
from brain import SurvivalBrain


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
    old_energy,
    world
):
    reward = -0.02
    reasons = []

    # Did we move closer to food?
    old_food_distance = distance(old_agent, old_food)
    new_food_distance = distance(world.agent, old_food)

    if new_food_distance < old_food_distance:
        reward += 0.20
        reasons.append("CLOSER TO FOOD")

    elif new_food_distance > old_food_distance:
        reward -= 0.10
        reasons.append("FARTHER FROM FOOD")

    # Did we move closer to danger?
    old_danger_distance = distance(old_agent, old_danger)
    new_danger_distance = distance(world.agent, old_danger)

    if new_danger_distance < old_danger_distance:
        reward -= 0.15
        reasons.append("CLOSER TO DANGER")

    elif new_danger_distance > old_danger_distance:
        reward += 0.05
        reasons.append("AWAY FROM DANGER")

    # Eating food is strongly positive.
    if world.score > 0 and old_food != world.food:
        reward += 2.0
        reasons.append("ATE FOOD")

    # Losing health means danger was hit.
    if world.health < old_health:
        reward -= 2.0
        reasons.append("HIT DANGER")

    # Waiting still consumes energy.
    if world.agent == old_agent:
        reward -= 0.05
        reasons.append("NO MOVEMENT")

    # Death is strongly negative.
    if not world.alive:
        reward -= 2.0
        reasons.append("DIED")

    if not reasons:
        reasons.append("ENERGY COST")

    return reward, reasons


def main():

    world = SurvivalWorld()
    brain = SurvivalBrain()

    reward = None

    print("CADENCE // SURVIVAL")
    print("Learning brain connected.")
    time.sleep(1)

    while world.alive:

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
        old_energy = world.energy

        world.move(action)

        reward, reasons = calculate_reward(
            old_agent,
            old_food,
            old_danger,
            old_health,
            old_energy,
            world
        )

        world.draw()

        info = brain.diagnostics()

        print()
        print("============== CADENCE BRAIN ==============")
        print("PERCEPTION :", [round(x, 2) for x in senses])
        print("DECISION   :", action)
        print("REWARD     :", round(reward, 2))
        print("OUTCOME    :", ", ".join(reasons))
        print("DOPAMINE   :", info["dopamine"])
        print("DECISIONS  :", info["decisions"])
        print("OUTCOMES   :", info["outcomes"])
        print("CONTROL    : CADENCE + LEARNING")
        print("===========================================")

        time.sleep(0.20)

    # Send the final consequence back to Cadence.
    final_senses = perception(world)

    try:
        brain.think(
            final_senses,
            reward=reward,
            done=True
        )
    except RuntimeError:
        pass

    world.draw()

    info = brain.diagnostics()

    print()
    print("================ RESULT ===================")
    print("AGENT DIED")
    print("FINAL SCORE :", world.score)
    print("SURVIVED    :", world.steps, "steps")
    print("DECISIONS   :", info["decisions"])
    print("OUTCOMES    :", info["outcomes"])
    print("===========================================")


if __name__ == "__main__":
    main()