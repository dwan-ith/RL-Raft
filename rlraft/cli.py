from __future__ import annotations

import argparse
import json
import sys
import time

from rlraft.config import ClusterConfig, load_config, save_default_config
from rlraft.web.dashboard import DashboardServer
from rlraft.sim.experiments import run_policy_comparison, run_simulation_comparison
from rlraft.core.supervisor import ClusterSupervisor
from rlraft.rl.training import train_policy
from rlraft.rl.mappo import train_mappo_policy
from rlraft.rl.llm_node import check_llm_node_policy
from rlraft.rl.logger import TrainingLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL-Raft multi-process demo")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="start a live cluster and dashboard")
    start.add_argument("--config", help="path to JSON config")
    start.add_argument("--nodes", type=int, default=None, help="cluster size")
    start.add_argument("--policy", choices=["static", "adaptive", "learned", "mappo", "llm", "rl_stub"], default=None)
    start.add_argument("--host", default=None)
    start.add_argument("--port", type=int, default=None)
    start.add_argument("--no-dashboard", action="store_true")
    start.add_argument("--train-episodes", type=int, default=6000)

    exp = sub.add_parser("run-experiment", help="compare static and adaptive failover")
    exp.add_argument("--repetitions", type=int, default=3)
    exp.add_argument("--nodes", type=int, default=50)
    exp.add_argument("--output-dir", default="runs")

    sim = sub.add_parser("sim-compare", help="deterministic simulator comparison")
    sim.add_argument("--nodes", type=int, default=50)
    sim.add_argument("--episodes", type=int, default=1000)
    sim.add_argument("--output-dir", default="runs")
    sim.add_argument("--seed", type=int, default=7)
    sim.add_argument("--include-llm", action="store_true", help="include direct LLM-node policy with deterministic fallback")
    sim.add_argument("--train-episodes", type=int, default=12000, help="MAPPO episodes if a comparison policy must be trained")

    train = sub.add_parser("train", help="run multi-agent practice rounds")
    train.add_argument("--episodes", type=int, default=6000)
    train.add_argument("--nodes", type=int, default=50)
    train.add_argument("--output", default="runs/policies/learned_ppo.json")
    train.add_argument("--algorithm", choices=["llm_mappo", "mappo", "qlearning"], default="llm_mappo")
    train.add_argument("--advisor", choices=["deterministic", "off"], default="off", help="legacy qlearning only; LLM reward shaping is disabled")
    train.add_argument("--advisor-interval", type=int, default=5000)
    train.add_argument("--require-llm", action="store_true", help="fail unless direct LLM-node policy can make a decision")
    train.add_argument("--hidden-dim", type=int, default=128, help="actor-critic hidden layer size (128/256/512)")

    smoke = sub.add_parser("smoke", help="start a cluster briefly and print final metrics")
    smoke.add_argument("--nodes", type=int, default=50)
    smoke.add_argument("--policy", choices=["static", "adaptive", "learned", "mappo", "llm", "rl_stub"], default="llm")
    smoke.add_argument("--seconds", type=float, default=5.0)
    smoke.add_argument("--train-episodes", type=int, default=6000)
    smoke.add_argument("--snapshot", default=None)

    cfg = sub.add_parser("write-config", help="write a default config file")
    cfg.add_argument("path", nargs="?", default="config.json")

    sub.add_parser("check-llm", help="check direct LLM-node policy access without printing secrets")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "write-config":
        save_default_config(args.path)
        print(f"Wrote {args.path}")
        return 0
    if args.command == "run-experiment":
        csv_path = run_policy_comparison(
            repetitions=args.repetitions,
            cluster_size=args.nodes,
            output_dir=args.output_dir,
        )
        print(f"Wrote comparison metrics to {csv_path}")
        return 0
    if args.command == "sim-compare":
        from pathlib import Path
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_simulation_comparison(
            nodes=args.nodes,
            episodes=args.episodes,
            output_dir=str(output_dir),
            seed=args.seed,
            include_llm=args.include_llm,
            train_episodes=args.train_episodes,
        )
        print(f"Wrote deterministic simulation comparison to {csv_path}")
        return 0
    if args.command == "train":
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if args.require_llm:
            llm_status = check_llm_node_policy()
            if llm_status["source"] not in {"llm", "llm_cache", "groq", "openrouter", "deepseek"}:
                print(json.dumps(llm_status, indent=2))
                return 1
        if args.algorithm in {"llm_mappo", "mappo"}:
            hidden_dim = getattr(args, "hidden_dim", 128)
            logger = TrainingLogger(
                episodes=args.episodes,
                nodes=args.nodes,
                algorithm=args.algorithm,
                config={
                    "hidden_dim": hidden_dim,
                    "use_llm_prior": args.algorithm == "llm_mappo",
                    "require_llm": args.require_llm,
                    "output": args.output,
                },
            )
            try:
                result = train_mappo_policy(
                    episodes=args.episodes,
                    nodes=args.nodes,
                    output_path=args.output,
                    use_llm_prior=args.algorithm == "llm_mappo",
                    require_llm=args.require_llm,
                    hidden_dim=hidden_dim,
                    logger=logger,
                )
            finally:
                logger.close()
        else:
            result = train_policy(
                args.episodes,
                args.nodes,
                args.output,
                advisor=args.advisor,
                advisor_interval=args.advisor_interval,
            )
        print(
            f"Trained {args.algorithm} timeout policy: "
            f"success={result.success_rate:.3f} "
            f"split={result.split_vote_rate:.3f} "
            f"best_node_wins={result.best_node_win_rate:.3f} "
            f"avg_failover={result.average_failover_ms:.1f}ms "
            f"path={result.policy_path}"
        )
        return 0
    if args.command == "check-llm":
        print(json.dumps(check_llm_node_policy(), indent=2))
        return 0
    if args.command == "smoke":
        return smoke_cluster(args)
    if args.command == "start":
        return start_cluster(args)
    return 2


def start_cluster(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.nodes is not None:
        config.cluster_size = args.nodes
    if args.policy is not None:
        config.policy_mode = args.policy
    if args.host is not None:
        config.dashboard_host = args.host
    if args.port is not None:
        config.dashboard_port = args.port
    if config.policy_mode in {"learned", "mappo"}:
        from pathlib import Path

        if not Path(config.learned_policy_path).exists():
            result = train_mappo_policy(
                episodes=args.train_episodes,
                nodes=config.cluster_size,
                output_path=config.learned_policy_path,
                seed=config.random_seed,
            )
            print(
                "Trained learned timeout policy before launch: "
                f"success={result.success_rate:.3f}, "
                f"split={result.split_vote_rate:.3f}, "
                f"best_node_wins={result.best_node_win_rate:.3f}"
            )

    supervisor = ClusterSupervisor(config)
    dashboard: DashboardServer | None = None
    supervisor.start()
    try:
        if not args.no_dashboard:
            dashboard = DashboardServer(supervisor, config.dashboard_host, config.dashboard_port)
            dashboard.start()
            print(f"Dashboard: http://{config.dashboard_host}:{config.dashboard_port}")
        print(
            f"Started RL-Raft cluster with {config.cluster_size} nodes "
            f"(policy={config.policy_mode}). Press Ctrl+C to stop."
        )
        while True:
            snapshot = supervisor.snapshot()
            metrics = snapshot["metrics"]
            print(
                f"leader={metrics['leader_id']} term={metrics['current_term']} "
                f"elections={metrics['elections_started']} splits={metrics['split_votes']} "
                f"dropped={metrics['messages_dropped']}",
                end="\r",
            )
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping cluster...")
    finally:
        if dashboard:
            dashboard.stop()
        supervisor.stop()
    return 0


def smoke_cluster(args: argparse.Namespace) -> int:
    config = ClusterConfig(cluster_size=args.nodes, policy_mode=args.policy)
    if config.policy_mode in {"learned", "mappo"}:
        from pathlib import Path

        if not Path(config.learned_policy_path).exists():
            train_mappo_policy(
                episodes=args.train_episodes,
                nodes=config.cluster_size,
                output_path=config.learned_policy_path,
                seed=config.random_seed,
            )
    supervisor = ClusterSupervisor(config)
    supervisor.start()
    try:
        leader = supervisor.wait_for_leader(timeout_s=args.seconds)
        time.sleep(max(0.0, args.seconds - 1.0))
        snapshot = supervisor.snapshot()
        if args.snapshot:
            supervisor.save_snapshot(args.snapshot)
        metrics = snapshot["metrics"]
        print(
            f"nodes={args.nodes} policy={args.policy} leader={leader} "
            f"term={metrics['current_term']} elections={metrics['elections_started']} "
            f"splits={metrics['split_votes']} delivered={metrics['messages_delivered']} "
            f"dropped={metrics['messages_dropped']}"
        )
        return 0 if leader is not None else 1
    finally:
        supervisor.stop()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
