# Final Project: Multi-Agent Competition in SoccerTwos

> This file is a Markdown transcription of the PDF for AI-agent use.
> Source of truth (SSOT): [Final Project Instructions Document.pdf](./Final%20Project%20Instructions%20Document.pdf)
> If this Markdown and the PDF ever disagree, follow the PDF.

## Overview

In this final project, you will move beyond single-agent environments and step into the arena of Multi-Agent Reinforcement Learning (MARL). You will be training agents to play SoccerTwos, a physics-based, 2-vs-2 competitive soccer environment provided by Unity ML-Agents.

This project introduces the complexity of adversarial dynamics. Your agent must learn not only how to control its own body and manipulate the ball but also how to coordinate with a teammate and outmaneuver an opposing team that is learning alongside it.

You will work in teams of 2. Teams of any other size must receive written permission from the instructor.

- Team signup sheet: https://docs.google.com/spreadsheets/d/12AlQluvbrXTxEDJHGbos8WTez5dqbtM-Xu0DFokDmSA/edit?usp=sharing
- Starter code: https://github.com/mdas64/soccer-twos-starter/tree/main[^1]

## The Environment

The match takes place in a walled arena. Each team consists of two agents: a Striker and a Goalie (or two generalists, depending on your strategy).

- Observations: Multiple rays cast in different directions to detect objects, agent's own velocity, rotation, position relative to ball/goals, ball position and velocity relative to agent, and position of own goal and opponent's goal.
- Actions: Agents operate in a continuous action space, controlling movement and rotation to strike the ball.
- Rewards: The default environment provides sparse rewards (getting a goal).

## The Challenge

Training a competitive agent requires more than simply running a standard algorithm like PPO or SAC. You will encounter the sparse reward problem (goals are rare events) and the instability of self-play (your opponent keeps changing).

You are required to:

1. Modify the Learning Environment: You must alter the Observation Space or the Reward Function to accelerate learning or encourage complex strategies (e.g., passing, blocking, or positioning).
2. Run Experiments: scientifically validate your modifications by comparing them against a baseline.

## Rubric

**Submission Policy:** Students must submit multiple trained agents to meet all assignment requirements. In both the agent description and the report, clearly identify which agent file corresponds to each evaluation criterion (e.g., Agent1 - policy performance, Agent2 - reward modification, Agent3 - imitation learning, etc.).

Training plots are required for every agent that is discussed or submitted. Additionally, include a direct performance comparison across agents, such as overlaid learning curves, to support your analysis.

### Hands-On Implementation and Code (100 points)

#### Submission Integrity (10 points)

Note: Must be satisfied for all submitted agents; otherwise points will not be awarded.

- (2 pts) Implemented a class that inherits from `soccer_twos.AgentInterface` and implements an `act` method.
- (2 pts) Filled in agent's information in the `README.md` file (agent name, authors and emails, and description).
- (2 pts) Compressed agent's module folder as `.zip`.
- (4 pts) The unzipped model loads without errors.

#### Reward, Observation, or Architecture Modification (40 points)

- (20 pts) Added or altered the reward or observation space (visible in source code).
- (20 pts) Code is syntactically correct and logical.
- (+5 pts) Introduce novel concept of learning (e.g., curriculum learning, imitation learning, etc.).

#### Policy Performance (50 points)

- (25 pts) The agent wins 9 out of 10 matches against the Random Agent.
- (25 pts) The agent wins 9 out of 10 matches against the Baseline Agent.
- (+5 pts) The agent wins 9 out of 10 matches against competitive agent2.[^2]

### Report (100 points)

#### Formatting & Structure (10 points)

- (5 pts) The report is 1-2 pages excluding references.
- (2.5 pts) Figures and charts are included and legible.
- (2.5 pts) References included.

#### Methodology (25 points in the PDF; likely intended to be 35 points)

- Editor note: the four rubric items under this section sum to 35 points.

- (10 pts) Clearly states the algorithm used (e.g., PPO, SAC), the library used, and provides an explanation of the theoretical background of the algorithm/architecture.
- (10 pts) Includes a table or list of the specific hyperparameters used for the final training run (Learning Rate, Batch Size, Buffer Size, etc.).
- (5 pts) Explicitly writes out what specific modification was made to the rewards or observations.
- (10 pts) The report explains the hypothesis or motivation behind the chosen modification (e.g., why they believed this specific change would improve performance).

#### Experimental Results (25 points)

- (10 pts) Includes a training curve plot (Reward vs. Steps/Time) for each agent.
- (10 pts) Graph or table includes comparison between baselines and submitted agents (or between multiple submitted agents).
- (5 pts) The charts/tables are clearly labeled: X-Axis Label (e.g., Steps/Time) and Y-Axis Label (e.g., Cumulative Reward).

#### Analysis and Discussion (30 points)

- (15 pts) The text explicitly states whether their modification resulted in higher reward, faster convergence, or lower performance compared to the baseline.
- (15 pts) The student(s) offers a technical reason why the result happened.

## Included Functionality

### Example Demo Scripts

- `example_random_players.py` - Simple test script running 4v4 matches with random agents
- `example_random_teams.py` - Simple script running a match with all agents taking random actions

### Example Training Scripts

- `example_ray_dqn_sp.py` - Ray Tune DQN algorithm on single-player team vs policy variation
- `example_ray_ma_players_offline.py` - Ray Tune PPO multi-agent player training with offline policy learning
- `example_ray_ma_players.py` - Ray Tune PPO multi-agent player-level control training
- `example_ray_ma_teams.py` - Ray Tune PPO multi-agent team-level control training
- `example_ray_ppo_sp_still.py` - Ray Tune PPO single-player training with stationary opponents
- `example_ray_team_vs_random.py` - Ray Tune PPO team training against random policy opponent
- `train_ray_curriculum.py` - Ray Tune PPO training with curriculum learning that progressively adjusts task difficulty
- `train_ray_selfplay.py` - Ray Tune PPO training with self-play where agents compete against archived versions of themselves

### Example Agent Classes

- `example_player_agent` - Template agent package structure with DQN policy for single-player variation
- `example_team_agent` - Template agent package structure for team-based gameplay

### Example PACE Scripts

- `soccertwos_job.batch` - SLURM batch script for submitting training jobs to Georgia Tech's PACE high-performance computing cluster

## PACE Documentation

We have been granted access to PACE-ICE (Instructional Cluster Environment) to offer students compute for projects. We have recently started offering PACE, so it's still something new to us, seeking to gather knowledge along the way. Feel free to ask questions or post feedback in this thread.

### Important Ground Rules

- Read [PACE-ICE documentation](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042102) and follow any guidelines and rules outlined there.
- DO NOT contact PACE support. PACE does not provide direct support for students.

### Basic Info

- PACE is a remote computing cluster, using the [slurm](https://slurm.schedmd.com/documentation.html) system to allocate resources (CPUs, GPUs, etc.) and schedule jobs (e.g., a `train.sh` script). Most of the workflow requires a secure (`ssh`) connection and is done via the terminal.
- For slurm instructions and commands, check the guidelines below and/or its complete version [here](https://mshalimay.github.io/slurm_pace_guidelines/) (or [PDF](https://drive.google.com/file/d/1PmmU5pDv9rY6Cayrp96GAwpXCfViSvro/view?usp=sharing)), [PACE documentation](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096), and this [cheatsheet](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/slurm-cheatsheet).
- To login, first connect to [GaTech VPN](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0026837) (via GlobalProtect), then `ssh` into the login node `login-ice.pace.gatech.edu` using your GT ID and password (the same you use for Canvas, Buzzport, etc.).
- Interactive jobs and file uploads can also be submitted via [Open OnDemand](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042133), a visual interface to access PACE. This is especially convenient for moving files between your local machine and PACE.
- Do NOT run compute-heavy programs on the login node (the one you land on after `ssh`).
- Make sure to use the scratch directory to store large files, models, and install Python packages, including from virtual environments. The home directory has a 15GB limit, and exceeding this will block creation of new files, disrupting your workflow entirely.

### Known Issues

- High latency to the cluster can cause dropped connections. Unfortunately this is not something neither we nor PACE can solve.
- Demand for resources peaks near project due dates. This means you may have to wait for a long time to get a GPU during these times. Therefore, avoid waiting until the end of the semester to run your jobs.
- Software version availability can vary between PACE nodes. This gives a little headache when using different nodes; this is not uncommon, as not always the same nodes are available.
- Sometimes there are issues connecting macOS to the GT VPN. Here is a thread on how to resolve these issues: ["Could not connect to GlobalProtect Service" on macOS](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0043475).
- If you receive "unknown server certificate error" when connecting to the GT VPN, it may help to force quit the GlobalProtect app.

### Guidelines

Here are some in-depth guidelines to use slurm in the PACE cluster. If you are new to slurm, some terms may be confusing at first. Please feel free to ask questions.

- Guidelines: https://mshalimay.github.io/slurm_pace_guidelines/
- Guidelines PDF: https://drive.google.com/file/d/1PmmU5pDv9rY6Cayrp96GAwpXCfViSvro/view?usp=sharing

## Transcription Notes

- Page breaks from the PDF were removed.
- Wide rubric tables from the PDF were normalized into Markdown sections and bullet lists for easier agent parsing.
- Inference from the PDF: the heading says `Methodology (25 points)`, but the listed items under that heading sum to 35 points. This Markdown preserves the PDF wording and adds an explicit editor note about the likely typo.

[^1]: Referenced from https://github.com/bryanoliveira/soccer-twos-starter.
[^2]: The competitive agent will be released at a later date.
