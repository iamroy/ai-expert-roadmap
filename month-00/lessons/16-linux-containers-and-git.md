# 0.16 — Linux, Containers, and Git

**Depth: SKIM**

**Goal:** diagnose a local AI service, package it reproducibly, and change it through a safe Git workflow.

[Month 0 roadmap](../README.md) · [Previous: APIs](15-apis-and-web-fundamentals.md) · [Next: Software Engineering](17-software-engineering-for-ai.md)

## Shell, processes, and environment

The shell starts processes with arguments, environment variables, a working directory, file descriptors, and permissions. Quoting determines whether spaces, wildcards, and substitutions are interpreted by the shell or passed literally. Treat command strings as code; never interpolate untrusted values into a shell command.

A process has an ID, parent, user/group identity, exit status, and resource usage. Signals request actions such as graceful termination; an application should stop accepting work, finish or cancel bounded in-flight tasks, flush necessary state, and exit within the orchestrator's grace period.

Environment variables are strings inherited by child processes. They suit deployment-specific configuration and secret references, but not large structured config. Validate presence and type at startup. Avoid logging the entire environment.

## Files, permissions, and networking

Unix permissions grant read, write, and execute rights to owner, group, and others. Directory execute permission controls traversal. Run services as a non-root user, grant least privilege, and keep secrets readable only where required.

A service binds an IP address and port. Binding to `127.0.0.1` exposes it only within the current network namespace; binding to `0.0.0.0` accepts traffic on all interfaces subject to firewall/routing. A container has its own namespace, so a service intended for port publishing normally listens on the container interface, not only its loopback address.

Useful diagnosis follows layers: process running, port listening, local request succeeds, container/service routing works, then remote policy/DNS/TLS. Distinguish connection refusal, timeout, and application error.

## Containers, images, volumes, and ports

An image is an immutable template of filesystem layers and metadata. A container is a running process environment created from an image. Containers share the host kernel; they are not small virtual machines and do not form a security boundary by themselves.

Build reproducibly with pinned base images/dependencies, a minimal build context, non-root runtime user, and separate build/runtime stages. Put frequently changing source after dependency installation to improve layer reuse. Never bake credentials into an image layer.

Container writable layers are ephemeral. Use a volume or external object/database service for durable state. A bind mount exposes a host path and is convenient in development but couples behavior to that host.

Port publishing maps a host address/port to a container port. `EXPOSE` documents a port but does not publish it. Health checks should distinguish process liveness from readiness to accept useful work; loading a large model can make the process alive before it is ready.

## Git commits, branches, merge, and rebase

Git stores snapshots connected by commits. A branch name points to a commit; `HEAD` identifies the checked-out position. The staging area selects the exact snapshot for the next commit.

Before committing, inspect status and the staged diff. Keep secrets, datasets, checkpoints, and generated artifacts ignored. `.gitignore` does not remove a secret already committed; revoke it and rewrite history if needed.

Merge combines histories and may create a merge commit. Rebase reapplies commits onto a new base, rewriting their identities. Rebase private/local work when useful; avoid rewriting a shared published branch without team agreement. Resolve conflicts by understanding intended content, then run tests—conflict markers are not the only possible error.

## Production workflow

```text
small change → inspect diff → test → commit → review → integrate → deploy artifact
```

Build an artifact from a specific commit and record its digest. Deploy the same artifact through environments rather than rebuilding from mutable sources. This connects Git identity, container identity, and model/data versions into traceable releases.

## Checkpoint

1. Why might a service work on host localhost but be unreachable through a container port?
2. What is the difference between an image and a container?
3. Why does deleting a secret and adding it to `.gitignore` not remove exposure?
4. How do merge and rebase differ in history?

<details>
<summary>Show answers</summary>

1. It may bind only to the wrong loopback namespace, the port may not be published, or routing/firewall configuration may block it.
2. An image is a versioned template; a container is a running process environment instantiated from it with writable runtime state.
3. Earlier commits still contain the secret. Revoke/rotate it immediately and use an appropriate history-removal procedure.
4. Merge preserves both existing histories and joins them; rebase creates new commits by replaying changes on another base.

</details>

## Exercise — Containerize a model API on paper

Specify a container for a CPU inference API: build stages, runtime user, model storage, configuration, ports, health checks, shutdown, and release identity.

<details>
<summary>Show exercise solution</summary>

Use a pinned language/runtime base and dependency lock file in a build stage, then copy only the installed environment and application into a smaller runtime image. Create an unprivileged user and read-only application directory. Fetch or mount a verified model artifact at deployment; do not bake secrets into build arguments or layers.

Validate environment configuration at startup, listen on the configured container interface, and publish only the required host port. Liveness checks the process/event loop; readiness verifies the model is loaded and dependencies are usable without running an expensive full inference every probe. Handle termination with bounded draining. Label the image with source commit and dependencies, publish by immutable digest, and record the model digest separately.

</details>

## Completion criteria

Diagnose processes and ports, explain permissions/environment, distinguish images/containers/volumes, and use Git staging, branching, merging, and rebasing safely.

## Primary references

- [The Linux command line](https://www.gnu.org/software/bash/manual/bash.html)
- [Docker: What is a container?](https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-a-container/)
- [Dockerfile reference](https://docs.docker.com/reference/dockerfile/)
- [Pro Git](https://git-scm.com/book/en/v2)
