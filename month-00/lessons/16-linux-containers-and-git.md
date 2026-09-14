# 0.16 Linux, Containers, and Git — SKIM

Models train on Linux, ship in containers, and live in Git. This lesson is deliberately practical: the commands and concepts you will actually reach for, and the failure modes that waste the most time.

| | |
|---|---|
| **Mode** | SKIM |
| **Time** | 45–60 minutes |
| **Assumes** | 0.1 Python (environments, environment variables) |
| **Used by** | every project from here on; Month 7 training infrastructure, Month 8 deployment |

[Month 0 roadmap](../README.md) · [Previous: APIs and Web Fundamentals](15-apis-and-web-fundamentals.md) · [Next: Software Engineering for AI](17-software-engineering-for-ai.md)

## Learning objectives

After this lesson you can:

- navigate a Linux system, inspect processes, and manage environment variables and permissions
- read a Dockerfile, and explain images, layers, volumes, and port mapping
- use Git confidently for branching, committing, and merging or rebasing
- write a `.gitignore` that keeps a repository clean
- explain what must never be committed, and what to do when it already has been

## How to use this lesson

1. Attempt the [exit test](#exit-test) first; this is a topic where people often know 80% and lose hours to the other 20%.
2. Record gaps in [`progress.md`](../progress.md).

## 0.16.1 — Shell basics

```bash
pwd                          # where am I
ls -la                       # list, including hidden, with permissions
cd /path/to/dir
find . -name "*.jsonl"       # search by name
grep -rn "TODO" src/         # search contents, recursive, with line numbers
du -sh *                     # directory sizes, human readable
df -h                        # free disk space
tail -f train.log            # follow a log as it is written
head -n 5 data.jsonl         # peek at a large file without opening it
wc -l data.jsonl             # count records
```

**Pipes and redirection** compose small tools into one job:

```bash
cat data.jsonl | jq -r '.label' | sort | uniq -c        # label distribution
python train.py > train.log 2>&1 &                      # stdout and stderr to a file, background
grep ERROR train.log | wc -l                            # count errors
```

`2>&1` sends stderr to the same place as stdout, which is why a log file sometimes misses your tracebacks when you forget it.

## 0.16.2 — Processes

```bash
ps aux | grep python         # find running processes
top          /  htop         # live resource view
nvidia-smi                   # GPU utilization and memory
kill <pid>                   # ask a process to stop (SIGTERM)
kill -9 <pid>                # force (SIGKILL); no cleanup runs
```

Long jobs should survive a disconnected SSH session. Use `tmux` or `screen`, or `nohup command &`. A training run killed by a closed laptop lid is a rite of passage worth skipping.

`nvidia-smi` is the first thing to check when training is slow: if GPU utilization is low, the bottleneck is usually data loading (0.8.4), not the model. If memory is nearly full, revisit 0.8.7. Note that a process killed with `-9` cannot clean up, so GPU memory may stay held until the process is fully reaped.

## 0.16.3 — Environment variables

```bash
export API_KEY="sk-..."      # set for this shell and its children
echo $API_KEY
env | grep API               # list matching variables
unset API_KEY
```

Variables set with `export` live only in that shell session. For persistence use a shell profile, or a `.env` file loaded by your application (0.1.11) and listed in `.gitignore`.

Ones you will meet: `PATH`, `PYTHONPATH`, `CUDA_VISIBLE_DEVICES` (restricting which GPUs a process sees), `HF_HOME` (Hugging Face cache location, worth moving off a small root disk), and `TOKENIZERS_PARALLELISM`.

## 0.16.4 — Permissions

```bash
ls -l                        # -rw-r--r--  1 user group  1024 Sep 13 10:00 file.txt
chmod +x script.sh           # make executable
chmod 600 ~/.ssh/id_rsa      # owner read/write only
chown user:group file        # change ownership
```

The ten characters are a type flag plus three triples for owner, group, and others, each `rwx`. Numerically, read is 4, write 2, execute 1, so `755` means owner all, others read and execute, and `600` means owner read and write only.

```python
def to_octal(symbolic: str) -> str:
    """Convert the 9 permission characters from `ls -l` into octal: rw-r--r-- -> 644."""
    assert len(symbolic) == 9, "pass just the 9 permission characters"
    weights = {"r": 4, "w": 2, "x": 1, "-": 0}
    triples = [symbolic[0:3], symbolic[3:6], symbolic[6:9]]
    return "".join(str(sum(weights[character] for character in triple)) for triple in triples)


for symbolic in ["rw-r--r--", "rwxr-xr-x", "rw-------", "rwxrwxrwx"]:
    print(f"{symbolic} -> {to_octal(symbolic)}")
```

```text
rw-r--r-- -> 644
rwxr-xr-x -> 755
rw------- -> 600
rwxrwxrwx -> 777
```

Permission errors are common when a container writes to a mounted host directory as root, leaving files the host user cannot delete. Running the container as your own UID avoids it.

## 0.16.5 — Networking basics

```bash
curl -i https://example.com                      # request with response headers
curl -X POST -H "Content-Type: application/json" \
     -d '{"q":"hello"}' http://localhost:8000/search
ss -tulpn | grep 8000                            # what is listening on a port
ping example.com                                 # basic reachability
```

Concepts to keep straight: a **port** identifies a service on a host; `localhost` (127.0.0.1) is the machine itself; **DNS** maps names to IPs. Inside containers, `localhost` means *the container*, which is the single most common container networking mistake: a service binding to `127.0.0.1` inside a container is unreachable from outside it, and must bind `0.0.0.0` instead.

## 0.16.6 — Containers

A container packages an application with its dependencies so it runs identically anywhere. Unlike a virtual machine, it shares the host kernel, so it starts in milliseconds and is far lighter.

| Term | Meaning |
|---|---|
| **Image** | an immutable, layered filesystem snapshot plus a start command |
| **Container** | a running instance of an image |
| **Layer** | one filesystem diff; layers are cached and shared between images |
| **Volume** | host or managed storage mounted into the container, surviving restarts |
| **Port mapping** | exposes a container port on the host |

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# copy dependency files first: this layer is cached until they change
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# copy source last, since it changes most often
COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "myapp.server"]
```

```bash
docker build -t myapp:0.1 .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data myapp:0.1
docker ps                    # running containers
docker logs -f <container>   # follow logs
docker exec -it <container> bash    # shell inside a running container
```

Four things worth internalizing:

- **Layer order determines build speed.** Copying source before installing dependencies invalidates the dependency layer on every code change, turning a two-second rebuild into a two-minute one.
- **Containers are ephemeral.** Anything written inside a container is gone when it is removed. Data, checkpoints, and logs belong in volumes or object storage.
- **Never bake secrets into an image.** Layers persist, so a key added and later deleted is still in the history. Pass secrets at runtime.
- **GPU access needs explicit support**, via the NVIDIA container toolkit and `--gpus all`. Image and host CUDA versions must be compatible, which is a frequent source of confusion.

`PYTHONUNBUFFERED=1` is not decoration: without it, Python buffers stdout and your container logs appear in delayed bursts or vanish on a crash.

## 0.16.7 — Git

The model: a **commit** is a snapshot with a parent; a **branch** is a movable pointer to a commit; `HEAD` points at your current position.

```bash
git status                   # run this constantly
git add -p                   # stage interactively, hunk by hunk
git commit -m "message"
git log --oneline --graph --all
git diff                     # unstaged changes
git diff --staged            # staged changes
```

**Branching and integrating:**

```bash
git switch -c feature/tokenizer      # create and switch
git switch main
git merge feature/tokenizer          # preserves history, may create a merge commit
git rebase main                      # replays your commits on top of main; linear history
```

Merge versus rebase: merge preserves exactly what happened and is always safe. Rebase rewrites commits to produce a linear history, which is cleaner but **must not be done to commits others have already pulled**, because it invalidates their copies. The safe rule is rebase your own unpushed local work, merge anything shared.

**Undoing things**, in increasing order of severity:

```bash
git restore <file>                   # discard uncommitted changes to a file
git restore --staged <file>          # unstage, keeping changes
git commit --amend                   # fix the most recent commit (before pushing)
git revert <commit>                  # a new commit undoing an old one; safe on shared history
git reset --hard <commit>            # move the branch and discard work; destructive
git reflog                           # the safety net: find commits you thought you lost
```

`git reflog` recovers almost anything that was ever committed. Learn it before you need it.

## 0.16.8 — `.gitignore` and what never goes in a repository

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/

# Secrets and local config
.env
*.pem
*.key

# Data and artifacts
data/
*.jsonl
*.parquet
checkpoints/
*.pt
*.ckpt
wandb/
mlruns/

# Editors and OS
.vscode/
.idea/
.DS_Store

# Notebooks: strip outputs instead of ignoring the file
```

Categories that must stay out: **secrets** (keys, tokens, certificates), **large binaries** (datasets and checkpoints, which bloat the repo permanently because Git keeps every version forever), **generated files**, and **local configuration**. Commit a `.env.example` with the key *names* and no values.

Two notes:

- `.gitignore` only affects untracked files. A file already tracked keeps being tracked; use `git rm --cached <file>` first.
- Notebooks are a recurring problem: their outputs make every diff enormous and can embed data or credentials. Strip outputs before committing, with `nbstripout` or a pre-commit hook.

**If a secret is committed, rotate it.** Deleting the file in a new commit does not help; the value is in the history and possibly already cloned. Rewriting history with `git filter-repo` is worth doing, but rotation is what actually makes you safe.

For large files that genuinely belong with the code, use Git LFS or, better, keep them in object storage and commit only a manifest (0.14.5).

## Exercises

### Exercise 1 — A one-line data audit

Using only shell tools on a JSONL file, count records, count distinct labels, find the longest line, and detect exact duplicate lines. Do it without writing a Python script.

### Exercise 2 — Layer caching

Write two Dockerfiles for the same app, one copying source before installing dependencies and one after. Build both, change one source line, rebuild, and compare the times. Explain the difference in terms of layers.

### Exercise 3 — Branch, conflict, resolve

Create a repository, branch, make conflicting edits to the same line on both branches, merge, and resolve. Then do the same with rebase and compare the resulting `git log --graph`.

### Exercise 4 — Ignore file audit

Take any project you have and check whether it contains anything that should not be tracked: `git ls-files | xargs du -h 2>/dev/null | sort -rh | head -20` for large files, and a scan for `.env`, `*.pt`, and notebook outputs. Fix what you find.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```bash
wc -l data.jsonl                                     # record count
jq -r '.label' data.jsonl | sort | uniq -c           # label distribution
awk '{ print length, NR }' data.jsonl | sort -rn | head -1   # longest line and its number
sort data.jsonl | uniq -d | wc -l                    # exact duplicate lines
```

Without `jq`, `grep -o '"label": *[0-9]*' data.jsonl | sort | uniq -c` gets close. The duplicate count matters most: duplicates across splits are the leakage problem from 0.7.6, and finding them costs one command.

### Exercise 2

The slow version:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .                        # any source change invalidates everything below
RUN pip install --no-cache-dir -e .
CMD ["python", "-m", "myapp"]
```

The fast version copies `pyproject.toml` first, runs `pip install`, then copies `src/`. After a one-line source change, the slow version reinstalls every dependency while the fast version reuses the cached install layer and rebuilds only the final copy.

The rule generalizes: **order Dockerfile instructions from least to most frequently changing.** A layer's cache is invalidated by any change to it or to any layer above it.

### Exercise 3

```bash
mkdir /tmp/gitdemo && cd /tmp/gitdemo && git init -q
echo "learning rate: 1e-3" > config.yaml
git add . && git commit -qm "initial config"

git switch -qc experiment
echo "learning rate: 3e-4" > config.yaml
git commit -qam "lower the learning rate"

git switch -q main
echo "learning rate: 1e-2" > config.yaml
git commit -qam "raise the learning rate"

git merge experiment            # CONFLICT
# edit config.yaml to keep the intended value, then:
git add config.yaml && git commit -qm "resolve: keep 3e-4"
git log --oneline --graph --all
```

The merge produces a branching graph with a merge commit recording that two lines of work joined. Repeating the same scenario with `git rebase main` on the experiment branch instead produces a straight line, as though the work had always been done sequentially. Both resolve the same conflict; they differ only in the history they leave behind.

### Exercise 4

```bash
git ls-files | xargs du -h 2>/dev/null | sort -rh | head -20
git ls-files | grep -E '\.(env|pt|ckpt|pem|key)$'
git ls-files '*.ipynb' | xargs grep -l '"output_type"' 2>/dev/null
```

If a tracked file should not be: `git rm --cached <file>`, add it to `.gitignore`, and commit. If it was a secret, rotate it immediately; removing it from the working tree does not remove it from history.

</details>

## Exit test

1. What does `2>&1` do, and when do you need it?
2. How do you keep a long training job alive after disconnecting from SSH?
3. What does low GPU utilization during training usually indicate?
4. What is the difference between `kill` and `kill -9`?
5. What does `chmod 600` mean, and why use it for a private key?
6. Why must a server inside a container bind `0.0.0.0` rather than `127.0.0.1`?
7. What is the difference between an image and a container?
8. Why does Dockerfile instruction order affect build time?
9. Why must secrets never be added to an image, even if deleted in a later layer?
10. What is the difference between merge and rebase, and when is rebase unsafe?
11. What are four categories of things that must never be committed?
12. A secret was committed and pushed. What do you do?

<details>
<summary>Show answers</summary>

1. It redirects stderr to the same destination as stdout. Without it, redirecting output to a log file captures normal output but not errors or tracebacks.
2. Run it under `tmux` or `screen`, or start it with `nohup command &` so it is detached from the terminal session.
3. A data-loading bottleneck: the GPU is idle waiting for batches. Increase `num_workers`, enable prefetching and pinned memory, or use a faster data format.
4. `kill` sends SIGTERM, which the process can catch to clean up. `kill -9` sends SIGKILL, which cannot be caught, so no cleanup runs and resources such as GPU memory may stay held briefly.
5. Read and write for the owner only, with no access for group or others. A private key readable by anyone else is compromised, and SSH will refuse to use it.
6. Inside a container, `127.0.0.1` refers to the container's own loopback interface, which nothing outside can reach. Binding `0.0.0.0` listens on all interfaces so a mapped port works.
7. An image is an immutable layered filesystem plus a start command; a container is a running instance of one. Many containers can run from the same image.
8. Each instruction creates a cached layer, and a change invalidates that layer and everything below it. Putting rarely changing steps such as dependency installation first keeps their caches valid across code changes.
9. Layers are immutable and persist in the image history, so a deleted secret is still recoverable from the earlier layer.
10. Merge joins two branches and preserves the actual history, possibly with a merge commit. Rebase replays commits onto a new base, producing linear history but new commit hashes. It is unsafe on commits others have already pulled, because their history no longer matches.
11. Secrets, large binaries such as datasets and checkpoints, generated files, and local configuration. Notebook outputs are a common fifth.
12. Rotate the credential immediately. Removing it in a new commit does not help, since the value remains in history and may already have been cloned; rewriting history with `git filter-repo` is worth doing afterward but is not a substitute for rotation.

</details>

## Completion criteria

You are done when:

- you can inspect processes, disk, and GPU state without looking commands up
- you can read a Dockerfile and explain why its instructions are in that order
- you can branch, resolve a conflict, and explain merge versus rebase
- you can write a `.gitignore` for a Python ML project from scratch
- you know exactly what to do about a leaked key

## References

**Linux and shell**
- [The Missing Semester of Your CS Education, MIT](https://missing.csail.mit.edu/) — shell, scripting, and Git, tuned for exactly this gap
- [Linux Journey](https://linuxjourney.com/)
- [explainshell](https://explainshell.com/) — decompose any command

**Containers**
- [Docker: get started](https://docs.docker.com/get-started/)
- [What is a container?](https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-a-container/)
- [Dockerfile reference](https://docs.docker.com/reference/dockerfile/)
- [GNU Bash manual](https://www.gnu.org/software/bash/manual/bash.html)
- [Dockerfile best practices](https://docs.docker.com/build/building/best-practices/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — GPU access in containers

**Git**
- [Pro Git](https://git-scm.com/book/en/v2) — free; chapters 2, 3, 7
- [Learn Git Branching](https://learngitbranching.js.org/) — interactive, the fastest way to build the mental model
- [Oh Shit, Git!?!](https://ohshitgit.com/) — recovery recipes
- [git-filter-repo](https://github.com/newren/git-filter-repo) — removing committed secrets from history
- [pre-commit](https://pre-commit.com/) and [nbstripout](https://github.com/kynan/nbstripout)

## Videos and code to read

- [MIT: The Missing Semester](https://missing.csail.mit.edu/) — lecture videos and exercises for the shell, scripting, and Git; the single highest-return resource in this lesson
- [Learn Git Branching](https://learngitbranching.js.org/) — interactive branching and rebasing, which builds the mental model faster than reading
- [pre-commit/pre-commit](https://github.com/pre-commit/pre-commit) and [kynan/nbstripout](https://github.com/kynan/nbstripout) — stop secrets and notebook outputs from ever reaching a commit

## About this lesson

Written to cover section 0.16 of the [Month 0 curriculum](../README.md).
