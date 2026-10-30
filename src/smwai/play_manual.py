"""Play Super Mario World by hand through stable-retro (for testing the emulator setup).

Keys: arrows = D-pad, Z = B (jump), X = A (spin jump), A = Y (run/fire), S = X,
Enter = Start, Right-Shift = Select, F5 = save state to states/, R = reset, Esc = quit.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pygame
import stable_retro as retro

GAME = "SuperMarioWorld-Snes-v0"
KEYS = {
    pygame.K_UP: "UP", pygame.K_DOWN: "DOWN", pygame.K_LEFT: "LEFT", pygame.K_RIGHT: "RIGHT",
    pygame.K_z: "B", pygame.K_x: "A", pygame.K_a: "Y", pygame.K_s: "X",
    pygame.K_RETURN: "START", pygame.K_RSHIFT: "SELECT",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default="YoshiIsland2", help="built-in state name or path to .state")
    ap.add_argument("--speed", type=float, default=1.0, help="game speed multiplier")
    ap.add_argument("--skip", type=int, default=1, help="draw only every Nth frame")
    args = ap.parse_args()

    env = retro.make(GAME, state=args.state, use_restricted_actions=retro.Actions.ALL,
                     render_mode=None)
    obs, _ = env.reset()
    buttons = list(env.buttons)
    pygame.init()
    h, w = obs.shape[:2]
    # SCALED: SDL scales the native-size surface itself, so no per-frame Python scaling.
    pygame.display.set_mode((w, h), pygame.SCALED | pygame.RESIZABLE)
    frame = pygame.Surface((w, h))
    n = 0
    last_fps = time.time()
    pygame.display.set_caption("Super Mario World - manual play")
    clock = pygame.time.Clock()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                obs, _ = env.reset()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F5:
                path = Path("states") / f"manual_{int(time.time())}.state"
                path.write_bytes(env.unwrapped.em.get_state())
                print("saved", path.resolve(), flush=True)
                pygame.display.set_caption(f"SAVED {path.name}")
                last_fps = time.time() + 3        # keep the caption for a few seconds
        pressed = pygame.key.get_pressed()
        action = np.zeros(len(buttons), dtype=np.int8)
        for key, name in KEYS.items():
            if pressed[key] and name in buttons:
                action[buttons.index(name)] = 1
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
        n += 1
        if n % args.skip == 0:
            pygame.surfarray.blit_array(frame, np.transpose(obs, (1, 0, 2)))
            screen = pygame.display.get_surface()
            screen.blit(frame, (0, 0))
            pygame.display.flip()
        if time.time() - last_fps > 2:
            pygame.display.set_caption(f"Super Mario World - {n / (time.time() - last_fps):.0f} fps")
            n, last_fps = 0, time.time()
        clock.tick(60 * args.speed)


if __name__ == "__main__":
    main()
