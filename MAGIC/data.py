import sys
from pathlib import Path
import gym
from env_wrappers import *

def _ensure_ic3net_envs():
    magic_root = Path(__file__).resolve().parent
    ic3net_root = magic_root / "envs" / "ic3net-envs"
    if ic3net_root.exists() and str(ic3net_root) not in sys.path:
        sys.path.insert(0, str(ic3net_root))
    import ic3net_envs  # noqa: F401

def _ensure_grf_envs():
    magic_root = Path(__file__).resolve().parent
    grf_root = magic_root / "envs" / "grf-envs"
    if grf_root.exists() and str(grf_root) not in sys.path:
        sys.path.insert(0, str(grf_root))
    import grf_envs  # noqa: F401

def init(env_name, args, final_init=True):
    if env_name == 'predator_prey':
        _ensure_ic3net_envs()
        env = gym.make('PredatorPrey-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env, args=args)
    elif env_name == 'traffic_junction':
        _ensure_ic3net_envs()
        env = gym.make('TrafficJunction-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env, args=args)
    elif env_name == 'grf':
        _ensure_grf_envs()
        env = gym.make('GRFWrapper-v0')
        env.multi_agent_init(args)
        env = GymWrapper(env, args=args)

    elif env_name in ('smac', 'starcraft2'):
        repo_root = Path(__file__).resolve().parents[1]
        onpolicy_root = repo_root / 'external' / 'on-policy'
        if str(onpolicy_root) not in sys.path:
            sys.path.insert(0, str(onpolicy_root))
        from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env
        env = StarCraft2Env(args)
        port = getattr(args, "_sc2_port", None)
        if port is not None:
            try:
                env._sc2_port = int(port)
            except Exception:
                env._sc2_port = port
        try:
            seed = getattr(args, 'seed', None)
            if seed is not None and int(seed) >= 0:
                env.seed(int(seed))
        except Exception:
            pass
        env = SmacWrapper(env, args=args)

    else:
        raise RuntimeError("wrong env name")

    return env
