"""Generate procedural kitchens and run a single-agent Gym episode on one."""
from cooksim import generate_layout
from cooksim.envs import CookSimGymEnv, RandomPartner


def main():
    for style in ("ring", "divided"):
        lay = generate_layout(width=11, height=7, n_players=2, style=style, seed=7)
        issues = lay.validate()
        print(f"{style:8s} -> {lay.width}x{lay.height}, spawns={lay.start_positions}, valid={not issues}")
        print("\n".join(lay.to_lines()))
        print()

    # Train-style single-agent loop on a generated kitchen.
    env = CookSimGymEnv(
        procedural={"width": 9, "height": 7, "n_players": 2, "style": "ring"},
        partner_policy=RandomPartner(1),
        seed=0,
    )
    obs, _ = env.reset()
    ret = 0.0
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        ret += r
        done = term or trunc
    print(f"gym single-agent random return: {ret:.1f}")


if __name__ == "__main__":
    main()
