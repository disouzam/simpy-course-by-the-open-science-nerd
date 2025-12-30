from collections import namedtuple

import simpy

Machine = namedtuple("Machine", "size, duration")
m1 = Machine(1, 2)  # Small and slow
m2 = Machine(2, 1)  # Big and fast

env = simpy.Environment()
machine_shop = simpy.FilterStore(env, capacity=2)
machine_shop.items = [m1, m2]  # Pre-populate the machine shop


def user(name, env, ms, size):
    print(
        f"\nNow at: {env.now} - Before creating a get request for user {name}\n"
        + f"Get queue ({len(ms.get_queue)}): {ms.get_queue}\n"
        + f"Put queue ({len(ms.put_queue)}): {ms.put_queue}\n"
        + f"Machine shop items: {ms.items}"
    )

    machine = yield ms.get(lambda machine: machine.size == size)

    print(f"\nNow at: {env.now} - {name} got {machine}")

    print(
        f"\nNow at: {env.now} - Before timeout for {machine} for user {name}\n"
        + f"Get queue ({len(ms.get_queue)}): {ms.get_queue}\n"
        + f"Put queue ({len(ms.put_queue)}): {ms.put_queue}\n"
        + f"Machine shop items: {ms.items}"
    )

    yield env.timeout(machine.duration)

    print(
        f"\nNow at: {env.now} - After timeout for {machine} for user {name}\n"
        + f"Get queue ({len(ms.get_queue)}): {ms.get_queue}\n"
        + f"Put queue ({len(ms.put_queue)}): {ms.put_queue}\n"
        + f"Machine shop items: {ms.items}"
    )

    yield ms.put(machine)

    print(f"\nNow at: {env.now} - {name} released {machine}")

    print(
        f"\nNow at: {env.now} - After machine {machine} with user {name} was released\n"
        + f"Get queue ({len(ms.get_queue)}): {ms.get_queue}\n"
        + f"Put queue ({len(ms.put_queue)}): {ms.put_queue}\n"
        + f"Machine shop items: {ms.items}"
    )


users = [env.process(user(i, env, machine_shop, (i % 2) + 1)) for i in range(10)]
env.run()
