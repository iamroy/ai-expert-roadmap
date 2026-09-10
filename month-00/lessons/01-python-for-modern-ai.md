# 0.1 — Python for Modern AI

The goal is not to become a Python language expert. It is to know enough Python that modern ML and LLM code never feels mysterious, and to avoid the handful of Python mistakes that repeatedly break training scripts and AI services.

| | |
|---|---|
| **Mode** | SKIM |
| **Time** | 60–90 minutes for the core path; add about 45 minutes for the hands-on exercise |
| **Assumes** | You have written Python before and can run a script from a terminal |
| **Used by** | 0.8 PyTorch Fundamentals, the tiny text classifier project, 0.15 APIs, and every later month |

## Learning objectives

After this lesson you can:

- read typed, object-oriented PyTorch and LLM-client code without decoding syntax
- structure a small project as an importable package with an isolated, reproducible environment
- load configuration and secrets safely, log usefully, and fail with clear errors
- stream large datasets with generators and fan out I/O-bound API calls with `asyncio`
- choose between threads, processes, and async for a given workload
- recognize the Python pitfalls that most often cause silent bugs in ML code

## How to use this lesson

1. **Diagnose first.** Attempt [Exit test A](#exit-test-a--reading-the-code), [Exit test B](#exit-test-b--integrated-ai-code-patterns), and [Exit test C](#exit-test-c--find-the-bugs) before reading.
2. **Read selectively.** Study the sections behind the questions you missed. Read every **Pitfall** callout even when the syntax is familiar; those are where production bugs come from.
3. **Build.** Complete the [hands-on exercise](#hands-on-exercise--a-data-and-inference-harness). It produces patterns you can reuse in the Month 0 project.
4. **Record gaps** in [`progress.md`](../progress.md).

## Complete coverage map

| Required area | Covered in |
|---|---|
| Data structures, comprehensions, loops, conditionals, `enumerate`/`zip`, references versus copies | 0.1.1 |
| Functions, arguments, keyword-only arguments, mutable defaults, lambdas, scope | 0.1.2 |
| Classes, methods, attributes, special methods used by PyTorch | 0.1.3 |
| Inheritance, overriding, and `super()` | 0.1.4 |
| Dataclasses and decorators, including writing a simple one | 0.1.5 |
| Type hints, `Optional`, unions, `Literal`, `Callable`, `Iterator`, `TypedDict`, `Protocol` | 0.1.6 |
| Modules, packages, imports, `src` layout, running with `-m` | 0.1.7 |
| Iterators, generators, generator expressions, streaming JSONL | 0.1.8 |
| Context managers, resource cleanup, `contextlib` | 0.1.9 |
| Exceptions, `finally`, raising, chaining, custom errors | 0.1.10 |
| Environment variables, `.env` files, typed configuration | 0.1.11 |
| Production logging | 0.1.12 |
| Virtual environments, packages, pinning, `pyproject.toml`, editable installs, entry points | 0.1.13 |
| Async, `await`, the event loop, concurrency limits, timeouts, async streaming | 0.1.14 |
| Threads versus processes, the GIL, `DataLoader` workers | 0.1.15 |
| JSON, JSONL, pickle, and safe checkpoint loading | 0.1.16 |

## 0.1.1 — Core data structures and control flow

The four structures used constantly in AI code are `list`, `dict`, `tuple`, and `set`.

```python
models = ["gpt", "bert", "dinov3"]     # ordered and mutable
config = {"lr": 1e-4, "epochs": 10}    # key-value pairs
shape = (32, 512, 768)                  # ordered and usually fixed
classes = {"cat", "dog", "bird"}       # unique values
```

Typical uses:

- lists: samples, layers, messages, batches, and results
- dictionaries: configuration, batches, metrics, JSON payloads, and model outputs
- tuples: tensor shapes and fixed groups
- sets: uniqueness and fast membership checks

Dictionaries preserve insertion order. Sets have no defined order, so never build a vocabulary by iterating over a set and assigning IDs; sort it first, or IDs can change between runs.

### Indexing and membership

```python
batch = {
    "input_ids": [1, 52, 93],
    "attention_mask": [1, 1, 1],
}

input_ids = batch["input_ids"]

if "loss" in batch:
    print(batch["loss"])
```

### Comprehensions

You will see comprehensions constantly:

```python
scores = [x * 2 for x in range(5)]
metrics = {name: 0.0 for name in ["loss", "accuracy"]}
valid_scores = [x for x in scores if x > 2]
```

The first expression is equivalent to:

```python
scores = []

for x in range(5):
    scores.append(x * 2)
```

Comprehensions are useful when one short expression clearly describes a transformation. Use an ordinary loop when the logic has several steps or would be hard to read on one line.

### Loops and conditionals

```python
for score in scores:
    if score >= 5:
        label = "high"
    elif score >= 2:
        label = "medium"
    else:
        label = "low"
    print(score, label)
```

Recognize `for`, `while`, `if`, `elif`, `else`, `break`, and `continue`. AI code commonly loops over epochs, batches, records, prompts, or generated tokens.

### `enumerate`, `zip`, and unpacking

Training and evaluation loops use these constantly:

```python
batches = [[1, 2], [3, 4], [5, 6]]
for step, batch in enumerate(batches):
    print(step, batch)          # 0 [1, 2], then 1 [3, 4], ...

texts = ["good", "bad"]
labels = [1, 0]
for text, label in zip(texts, labels, strict=True):
    print(text, label)

for name, value in {"loss": 0.4, "accuracy": 0.9}.items():
    print(f"{name}={value:.3f}")
```

> **Pitfall:** Without `strict=True` (Python 3.10+), `zip` silently stops at the shortest input, which can quietly drop labels or predictions. With it, mismatched lengths raise `ValueError`.

### References, not copies

Assignment binds another name to the same object. It does not copy:

```python
base = {"lr": 1e-4, "layers": [256, 128]}
experiment = base
experiment["lr"] = 3e-4
print(base["lr"])   # 0.0003: both names refer to one dictionary
```

A shallow copy duplicates only the outer container, so nested lists and dictionaries are still shared. Use `copy.deepcopy` when deriving experiment configurations from a nested base configuration:

```python
import copy

base = {"lr": 1e-4, "layers": [256, 128]}
shallow = base.copy()
shallow["layers"].append(64)       # also changes base["layers"]
independent = copy.deepcopy(base)
independent["layers"].append(32)   # base is unchanged
```

### Quick check

What does this produce?

```python
{x: x**2 for x in range(3)}
```

<details>
<summary>Show answer</summary>

```python
{0: 0, 1: 1, 2: 4}
```

</details>

## 0.1.2 — Functions

Functions package reusable behavior and make data flow explicit.

```python
def train_model(model, epochs=10):
    print(model, epochs)
    return {"model": model, "epochs": epochs}
```

Here, `model` is required and `epochs` has a default value.

```python
train_model("bert")
train_model("bert", 20)
train_model(model="bert", epochs=20)
```

The first two calls use positional arguments. The third uses keyword arguments. AI libraries use keyword arguments heavily because model configuration can contain many options:

```python
trainer = Trainer(
    model=model,
    batch_size=16,
    learning_rate=1e-4,
    epochs=20,
)
```

> **Pitfall:** In PyTorch, `model.train()` does not start training. It switches layers such as dropout and batch normalization into training behavior, and `model.eval()` switches them back. Both appear in 0.8.

### `*args`

`*args` collects additional positional arguments into a tuple.

```python
def add(*args):
    return sum(args)

total = add(1, 2, 3)
# Inside add, args == (1, 2, 3)
```

You may not use it often, but you must recognize it.

### `**kwargs`

`**kwargs` collects additional keyword arguments into a dictionary.

```python
def create_model(**kwargs):
    print(kwargs)

create_model(hidden_size=768, num_layers=12)
```

Inside the function, `kwargs` is:

```python
{
    "hidden_size": 768,
    "num_layers": 12,
}
```

Configuration unpacking performs the reverse operation:

```python
config = {
    "hidden_size": 768,
    "num_layers": 12,
}

model = SomeModel(**config)
```

That constructor call is equivalent to:

```python
model = SomeModel(hidden_size=768, num_layers=12)
```

This is one of the most important Python patterns in AI repositories.

### Keyword-only arguments

Parameters after a bare `*` must be passed by name:

```python
def generate(prompt: str, *, max_tokens: int = 256, temperature: float = 0.7) -> str:
    return f"{prompt} ({max_tokens}, {temperature})"

generate("hello", max_tokens=64)   # valid
# generate("hello", 64)            # TypeError: too many positional arguments
```

Library authors use this to prevent silently swapped arguments, such as passing a temperature where a token limit was expected. You will see it throughout PyTorch, Hugging Face, and LLM SDK signatures.

### Mutable default arguments

Default values are evaluated once, when the function is defined, not on every call:

```python
def add_example(example, batch=[]):   # bug: one list shared by every call
    batch.append(example)
    return batch

add_example("a")
print(add_example("b"))   # ['a', 'b']
```

Use `None` as the default and create the object inside the function:

```python
def add_example(example, batch=None):
    if batch is None:
        batch = []
    batch.append(example)
    return batch
```

In services, this bug appears as a cache, metrics list, or chat history that keeps growing across unrelated requests.

### Lambda functions

A lambda is a small anonymous function containing one expression.

```python
records = [
    {"name": "a", "loss": 0.8},
    {"name": "b", "loss": 0.3},
]

records.sort(key=lambda record: record["loss"])
```

Use lambdas for short callbacks or keys. Use `def` when the logic deserves a name, documentation, type hints, or multiple statements.

### Local and global scope

Names assigned inside a function are local by default. Functions can read module-level names, but hidden global dependencies make code harder to test.

```python
DEFAULT_DEVICE = "cpu"

def choose_device(requested: str | None = None) -> str:
    device = requested or DEFAULT_DEVICE
    return device
```

Prefer passing dependencies through function arguments or configuration instead of modifying global state. Recognize the `global` and `nonlocal` keywords, but avoid them unless the ownership of state is clear.

## 0.1.3 — Classes

Most AI libraries are object oriented. You need to understand class structure rather than advanced object-oriented design.

```python
class Model:
    def __init__(self, name):
        self.name = name

    def predict(self, x):
        return f"{self.name}: {x}"
```

Usage:

```python
model = Model("classifier")
prediction = model.predict("image")
```

`__init__` initializes a new object. `self` refers to that specific object. `self.name` is an instance attribute, and `predict` is an instance method.

This PyTorch pattern should look familiar:

```python
from torch import nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(768, 2)

    def forward(self, x):
        return self.layer(x)
```

PyTorch models store submodules as instance attributes and define computation in `forward`.

### Special methods PyTorch relies on

Methods named with double underscores ("dunder" methods) let objects work with ordinary Python syntax. A few matter immediately:

```python
class TextDataset:
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:                               # len(dataset)
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:   # dataset[index]
        return self.texts[index], self.labels[index]

    def __repr__(self) -> str:                              # readable debugging output
        return f"TextDataset(size={len(self)})"

dataset = TextDataset(["good movie", "bad movie"], [1, 0])
print(len(dataset), dataset[0], dataset)   # 2 ('good movie', 1) TextDataset(size=2)
```

A PyTorch map-style `Dataset` is exactly this pattern: `DataLoader` calls `len(dataset)` and `dataset[index]` to assemble batches. The Month 0 project requires one.

`__call__` makes an instance callable like a function. `nn.Module` implements `__call__`, which runs any registered hooks and then your `forward` method:

```python
logits = model(inputs)             # preferred
# logits = model.forward(inputs)   # skips hooks; avoid
```

## 0.1.4 — Inheritance and `super()`

Inheritance lets one class build on another.

```python
class Animal:
    def speak(self):
        print("sound")

class Dog(Animal):
    pass

Dog().speak()
```

`Dog` inherits `speak` from `Animal`.

In PyTorch:

```python
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
```

`TransformerBlock` inherits functionality from `nn.Module`. Calling `super().__init__()` initializes the parent class so PyTorch can track parameters and child modules correctly.

A subclass can override a parent method and still reuse the parent's implementation:

```python
class BaseTokenizer:
    def normalize(self, text: str) -> str:
        return text.strip()

class LowercaseTokenizer(BaseTokenizer):
    def normalize(self, text: str) -> str:
        return super().normalize(text).lower()

print(LowercaseTokenizer().normalize("  Hello "))   # hello
```

> **Pitfall:** In an `nn.Module` subclass, call `super().__init__()` before assigning any layer. Assigning `self.layer = nn.Linear(...)` first raises `AttributeError: cannot assign module before Module.__init__() call`.

You do not need to study complex multiple-inheritance hierarchies for this roadmap.

## 0.1.5 — Dataclasses and basic decorators

Dataclasses provide a concise way to hold configuration.

Without a dataclass:

```python
class Config:
    def __init__(self, lr, batch_size, epochs):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
```

With a dataclass:

```python
from dataclasses import dataclass

@dataclass
class Config:
    lr: float
    batch_size: int
    epochs: int

config = Config(lr=1e-4, batch_size=16, epochs=10)
```

Dataclass features that matter for configuration:

```python
from dataclasses import asdict, dataclass, field

@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 16
    label_names: list[str] = field(default_factory=lambda: ["negative", "positive"])

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

config = TrainConfig(lr=3e-4)
print(asdict(config))
# config.lr = 1.0   # dataclasses.FrozenInstanceError
```

- `field(default_factory=...)` creates a new list for each instance. A plain `= []` default raises `ValueError`, for the shared-mutable-default reason described in 0.1.2.
- `frozen=True` prevents accidental reassignment after startup. It does not freeze the contents of a mutable field such as a list.
- `__post_init__` runs after the generated `__init__` and is a natural place for validation.
- `asdict` converts the dataclass to a dictionary, which is useful for logging a run's configuration or saving it as JSON.

`@dataclass` is a decorator. A decorator receives a function or class and returns a function or class, usually adding behavior.

Common decorators to recognize:

```python
class Vocabulary:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens

    @staticmethod
    def normalize(token: str) -> str:
        return token.lower()

    @classmethod
    def from_text(cls, text: str) -> "Vocabulary":
        return cls(tokens=text.split())

    @property
    def size(self) -> int:
        return len(self.tokens)
```

- `@staticmethod`: method does not receive `self` or `cls`
- `@classmethod`: receives the class as `cls`, often used for alternative constructors
- `@property`: exposes a method through attribute syntax

### Writing a simple decorator

A decorator is a function that takes a function and returns a replacement for it:

```python
import functools
import logging
import time

logger = logging.getLogger(__name__)

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("%s took %.1f ms", func.__name__, elapsed_ms)
    return wrapper

@timed
def embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(text))] for text in texts]

embed(["hello", "world"])
```

Writing `@timed` above the definition is equivalent to writing `embed = timed(embed)` after it. `*args, **kwargs` lets the wrapper forward any call unchanged, and `functools.wraps` preserves the original function's name and docstring for logs, debuggers, and documentation tools.

Library decorators you will see in ML code include `@torch.no_grad()` and `@torch.inference_mode()` for gradient-free functions, and `@functools.cache` or `@functools.lru_cache` for memoizing pure functions.

Recognize these patterns and be able to write a simple function decorator like `timed`; defer decorator factories, class-based decorators, and descriptor mechanics.

## 0.1.6 — Type hints

Modern AI projects use type hints heavily.

```python
def tokenize(text: str) -> list[int]:
    return [1, 2, 3]

def predict(text: str, temperature: float = 0.7) -> str:
    return text
```

The hints describe intended inputs and outputs. Python usually does not enforce them at runtime, but editors, readers, and static type checkers use them.

### Optional values and unions

```python
def load_model(path: str | None = None):
    ...
```

`path` can be a string or `None`. Older code may spell this as:

```python
from typing import Optional, Union

path: Optional[str]
device: Union[str, int]
```

### `Literal`, `Callable`, and `Iterator`

```python
from collections.abc import Callable, Iterator
from typing import Literal

Device = Literal["cpu", "cuda", "mps"]

def move(batch: list[float], device: Device = "cpu") -> list[float]:
    return batch

def apply(transform: Callable[[str], str], texts: list[str]) -> list[str]:
    return [transform(text) for text in texts]

def read_lines(path: str) -> Iterator[str]:
    with open(path, encoding="utf-8") as file:
        yield from file
```

- `Literal` restricts a value to specific constants, so a type checker flags `move(batch, "gpu")`.
- `Callable[[str], str]` describes a function that accepts a string and returns a string.
- A generator function is annotated with the type of value it yields, as in `Iterator[str]`.

### Tensor annotations

Type hints can say that a value is a tensor, but not what its shape is. Record shapes in docstrings, comments, or names:

```python
import torch

def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """hidden: [B, N, D] float; mask: [B, N] bool -> [B, D]."""
    ...
```

### `TypedDict`

`TypedDict` describes expected keys and value types in a dictionary.

```python
from typing import TypedDict

class Batch(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]

def token_count(batch: Batch) -> int:
    return sum(batch["attention_mask"])
```

At runtime this is still an ordinary dictionary.

### `Protocol`

A protocol describes behavior an object must provide without requiring inheritance from one specific base class.

```python
from typing import Protocol

class GeneratorClient(Protocol):
    async def generate(self, prompt: str) -> str:
        ...

async def ask(client: GeneratorClient, prompt: str) -> str:
    return await client.generate(prompt)
```

For Month 0, recognize `TypedDict` and `Protocol`; do not study advanced type-system features.

> **Pitfall:** Type hints, `TypedDict`, and `Protocol` are checked by tools such as mypy and pyright, not by Python at runtime. When data crosses a trust boundary, such as an HTTP request body or JSON produced by an LLM, validate it at runtime. Pydantic models are the common choice and return in Month 3's structured generation work.

## 0.1.7 — Modules and packages

A module is a Python file. A package is a directory of related modules, conventionally marked with an `__init__.py` file.

The Month 0 project uses a `src` layout:

```text
tiny-text-classifier/
├── pyproject.toml
├── src/
│   └── text_classifier/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       └── predict.py
└── tests/
    └── test_model.py
```

An absolute import names the full path from the top-level package:

```python
from text_classifier.model import Classifier
```

A relative import starts from the current module's package:

```python
# inside src/text_classifier/train.py
from .data import TextDataset
from .model import Classifier
```

Prefer absolute imports in application code; relative imports are fine between closely related modules in the same package. `..` moves up one package level, but it cannot climb above the top-level package. Trying to fails with `ImportError: attempted relative import beyond top-level package`.

For `text_classifier` to be importable from tests and scripts, install the project into its virtual environment in editable mode (see 0.1.13):

```bash
python -m pip install -e .
```

Then run modules by their import name:

```bash
python -m text_classifier.train          # works, including relative imports
# python src/text_classifier/train.py    # runs a loose script; relative imports fail
```

> **Pitfall:** Fixing import errors by appending directories to `sys.path` hides packaging mistakes and breaks when the code runs on another machine, in CI, or in a container. Fix the package layout or install the project instead.

### `if __name__ == "__main__"`

```python
def main():
    print("training")

if __name__ == "__main__":
    main()
```

This runs `main()` only when the file is executed directly. Importing the file from another module does not automatically execute `main`, which makes the module reusable and testable.

## 0.1.8 — Iterators and generators

An iterable can produce an iterator. An iterator yields one value at a time and remembers its current position.

```python
values = [10, 20, 30]
iterator = iter(values)

first = next(iterator)   # 10
second = next(iterator)  # 20
```

When the iterator is exhausted, `next` raises `StopIteration`. A `for` loop handles this protocol automatically.

A normal function can return all values at once:

```python
def get_numbers():
    return [1, 2, 3]
```

A generator yields them one at a time:

```python
def get_numbers():
    yield 1
    yield 2
    yield 3

for number in get_numbers():
    print(number)
```

Generators matter when processing large datasets, streaming API responses, or reading files that should not be loaded entirely into memory.

A generator expression uses parentheses:

```python
squares = (x**2 for x in range(1_000_000))
```

It computes values lazily rather than building a million-item list immediately.

### Streaming a dataset

Generators let you stream a JSON Lines file (one JSON object per line; see 0.1.16) and group records into batches without loading the whole file:

```python
import json
from collections.abc import Iterable, Iterator

def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error

def batched(records: Iterable[dict], size: int) -> Iterator[list[dict]]:
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch   # final partial batch
```

Python 3.12 also provides `itertools.batched`, which yields tuples.

> **Pitfall:** A generator can be consumed only once. Iterating over it again yields nothing and raises no error, so a validation loop that reuses an exhausted generator silently evaluates zero examples. Create a fresh generator, or use a reusable iterable such as a list or a `Dataset`.

## 0.1.9 — Context managers

A context manager acquires and releases a resource around a block.

```python
with open("config.json", encoding="utf-8") as file:
    data = file.read()
```

The file is closed even if reading or processing raises an exception. Conceptually, this resembles:

```python
file = open("config.json", encoding="utf-8")

try:
    data = file.read()
finally:
    file.close()
```

ML code also uses contexts to change behavior temporarily:

```python
with torch.no_grad():
    predictions = model(inputs)
```

This tells PyTorch not to build a gradient graph within the block.

Custom context-manager classes implement `__enter__` and `__exit__`. You only need the concept now:

```python
import time

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, error_type, error, traceback):
        self.elapsed = time.perf_counter() - self.start

with Timer() as timer:
    sum(range(1_000_000))
print(f"{timer.elapsed:.4f} s")
```

If `__exit__` returns a true value, the exception is suppressed. Returning `None`, as `Timer` does, lets any exception propagate, which is almost always what you want.

For simple cases, `contextlib.contextmanager` turns a generator into a context manager. Code before `yield` runs on entry, and the `finally` block runs on exit:

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: {time.perf_counter() - start:.4f} s")

with timer("tokenize"):
    tokens = "a short example".split()
```

Other PyTorch contexts to recognize are `torch.inference_mode()`, a stricter and faster alternative to `no_grad` for pure inference, and `torch.autocast(...)` for mixed precision (0.8).

## 0.1.10 — Exceptions

Catch specific failures you can handle:

```python
try:
    model = load_model()
except FileNotFoundError as error:
    logger.error("Model not found: %s", error)
```

Use `else` for work that should run only when no exception occurred and `finally` for cleanup that must always happen:

```python
connection = open_connection()

try:
    result = connection.read()
except TimeoutError:
    result = None
else:
    process(result)
finally:
    connection.close()
```

Raise a clear error when an input violates an assumption:

```python
if batch_size <= 0:
    raise ValueError("batch_size must be positive")

if device not in {"cpu", "cuda", "mps"}:
    raise ValueError(f"Unsupported device: {device}")
```

A custom exception gives a domain-specific failure a recognizable type:

```python
class ModelLoadError(RuntimeError):
    pass
```

When translating a low-level failure into a domain error, chain it with `from` so the original traceback is preserved:

```python
def load_checkpoint(path: str) -> bytes:
    try:
        with open(path, "rb") as file:
            return file.read()
    except FileNotFoundError as error:
        raise ModelLoadError(f"Checkpoint not found: {path}") from error
```

For Month 0, understand why custom exceptions exist; do not build elaborate hierarchies.

Avoid this pattern:

```python
try:
    ...
except:
    pass
```

It catches unrelated failures and hides evidence needed to debug them.

## 0.1.11 — Environment variables and configuration

Never hard-code secrets:

```python
# Do not do this
api_key = "secret-value"
```

Read them from the environment:

```python
import os

api_key = os.environ["API_KEY"]
device = os.getenv("DEVICE", "cpu")
```

`os.environ["API_KEY"]` raises `KeyError` if the value is missing. `os.getenv("API_KEY")` returns `None` unless a default is supplied.

> **Pitfall:** Environment variables are always strings. `os.getenv("BATCH_SIZE", "16")` needs an explicit `int(...)`, and `bool(os.getenv("DEBUG", "false"))` is `True` because every non-empty string is truthy. Parse Booleans explicitly.

A `.env` file is a local convenience for defining environment variables during development. A library such as `python-dotenv` can load it into the process environment:

```python
from dotenv import load_dotenv   # python -m pip install python-dotenv

load_dotenv()   # by default, does not override variables already set in the environment
```

Keep `.env` files containing secrets out of Git; this repository's `.gitignore` already excludes them. Commit an `.env.example` with variable names and safe placeholders when useful.

Use typed configuration to parse and validate settings once, at startup:

```python
from dataclasses import dataclass, field
import os

def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class Settings:
    model_name: str
    device: str
    batch_size: int
    debug: bool
    api_key: str | None = field(default=None, repr=False)

def load_settings() -> Settings:
    return Settings(
        model_name=os.getenv("MODEL_NAME", "default-model"),
        device=os.getenv("DEVICE", "cpu"),
        batch_size=int(os.getenv("BATCH_SIZE", "16")),
        debug=env_bool("DEBUG"),
        api_key=os.getenv("API_KEY"),
    )
```

A misconfigured deployment now fails immediately instead of midway through a job. `repr=False` keeps the key out of `print(settings)` and out of any log line that formats the settings object. Libraries such as `pydantic-settings` package this pattern with richer validation.

## 0.1.12 — Logging

`print` is useful for exploration. Production services need severity, timestamps, source information, filtering, aggregation, and exception traces.

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.debug("Batch shape: %s", batch.shape)
logger.info("Training started")
logger.warning("Falling back to CPU")
logger.error("Checkpoint is missing")
```

Inside an exception handler, `logger.exception` includes the stack trace:

```python
try:
    train()
except RuntimeError:
    logger.exception("Training failed")
    raise
```

Use parameterized logging (`logger.info("loss=%s", loss)`) so formatting happens only when that level is enabled.

Production habits worth adopting now:

- Configure logging once, at the application entry point, with `logging.basicConfig` or explicit handlers. Library modules should only call `logging.getLogger(__name__)`.
- Parameterized logging defers formatting, but the arguments are still evaluated. `logger.debug("%s", tensor.tolist())` still converts the whole tensor even when debug logging is off.
- In training loops, log every N steps rather than every step. Calling `loss.item()` on a GPU tensor makes the CPU wait for the GPU, so doing it on every step can slow training noticeably.
- Never log secrets, and treat raw prompts, model outputs, and user documents as potentially sensitive data.

## 0.1.13 — Virtual environments and packages

Virtual environments isolate one project's dependencies from another's.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy
python -m pip list
```

`python -m pip` guarantees that `pip` belongs to the active interpreter, so packages are not accidentally installed into a different Python.

> **Pitfall:** PyTorch wheels are built for specific accelerator stacks. Use the selector on [pytorch.org](https://pytorch.org/get-started/locally/) to get the install command for your OS and CUDA, ROCm, or CPU target, then verify with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

Freeze an existing environment:

```bash
python -m pip freeze > requirements.txt
```

Restore it later:

```bash
python -m pip install -r requirements.txt
```

Version constraints communicate reproducibility and compatibility:

```text
numpy==2.1.0     # exact pin
torch>=2.4,<2.6  # compatible range chosen by the project
```

Exact pins improve repeatability for applications, while compatible ranges are often more suitable for reusable libraries. Lock files produced by modern package managers can record a fully resolved environment.

Modern repositories increasingly use `pyproject.toml` for project metadata, dependencies, build configuration, and tool settings:

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "text-classifier"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["torch>=2.4"]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]

[project.scripts]
train-classifier = "text_classifier.train:main"
```

- `[build-system]` tells installers how to build the package, which `python -m pip install -e .` requires.
- `[project.optional-dependencies]` defines extras, installed with `python -m pip install -e ".[dev]"`.
- `[project.scripts]` defines an entry point: after installation, the `train-classifier` command calls `main()` in `text_classifier/train.py`.

Setuptools discovers packages under `src/` automatically. Tools such as [uv](https://docs.astral.sh/uv/) manage the virtual environment, dependency resolution, and a lock file from the same `pyproject.toml`, and are increasingly common in AI repositories.

You only need to recognize and make basic edits to this file during Month 0.

## 0.1.14 — Async and `await`

Normal blocking code waits for an operation to finish:

```python
response = call_llm()
```

An async coroutine can yield control while waiting for I/O:

```python
async def generate(prompt: str) -> str:
    response = await client.generate(prompt)
    return response
```

AI applications frequently wait for LLM APIs, databases, vector stores, HTTP services, and tool calls. Async lets one worker make progress on other requests during those waits.

The event loop schedules runnable coroutines and resumes a waiting coroutine when its I/O becomes ready. A coroutine does not execute merely because it was defined; it must be awaited or scheduled.

The example below simulates three one-second API calls. Awaited one after another they take about three seconds; scheduled together with `asyncio.gather` they take about one:

```python
import asyncio
import time

async def fake_generate(prompt: str) -> str:
    await asyncio.sleep(1.0)          # stands in for waiting on network I/O
    return f"response to {prompt!r}"

async def main() -> None:
    prompts = ["first", "second", "third"]

    start = time.perf_counter()
    for prompt in prompts:
        await fake_generate(prompt)
    print(f"sequential: {time.perf_counter() - start:.1f} s")   # ~3.0 s

    start = time.perf_counter()
    results = await asyncio.gather(*(fake_generate(p) for p in prompts))
    print(f"concurrent: {time.perf_counter() - start:.1f} s")   # ~1.0 s
    print(results)

asyncio.run(main())
```

`async` does not make model computation faster. It is mainly useful for I/O-bound work. Blocking code inside a coroutine, such as `time.sleep` or a synchronous HTTP client, still blocks the whole event loop. Move unavoidable blocking calls off the loop with `await asyncio.to_thread(blocking_function, ...)`.

By default, `gather` raises the first exception it encounters. Pass `return_exceptions=True` to collect per-request failures, or use `asyncio.TaskGroup` (Python 3.11+) when one failure should cancel the remaining tasks.

### Concurrency limits, timeouts, and streaming

Real LLM APIs enforce rate limits and occasionally hang. Bound concurrency with a semaphore and bound waiting with a timeout:

```python
async def generate_limited(prompt: str, limit: asyncio.Semaphore) -> str:
    async with limit:                     # at most N calls in flight
        async with asyncio.timeout(30):   # raises TimeoutError; Python 3.11+
            return await fake_generate(prompt)

async def run_all(prompts: list[str]) -> list[str]:
    limit = asyncio.Semaphore(4)
    return await asyncio.gather(*(generate_limited(p, limit) for p in prompts))
```

Token streaming uses async iteration. An async generator uses `yield` inside `async def` and is consumed with `async for`:

```python
async def stream_tokens(text: str):
    for token in text.split():
        await asyncio.sleep(0.05)   # stands in for the next network chunk
        yield token

async def show_stream() -> None:
    async for token in stream_tokens("tokens arrive one at a time"):
        print(token, end=" ", flush=True)

asyncio.run(show_stream())
```

> **Pitfall:** `asyncio.run` cannot be called while an event loop is already running, which is the case inside Jupyter notebooks. In a notebook cell, write `await main()` instead.

## 0.1.15 — Multithreading versus multiprocessing

These are different concurrency models:

- **threads** run in one process and share memory; they are often useful for blocking I/O
- **processes** have separate Python interpreters and memory; they can run CPU-bound Python work in parallel but add serialization and coordination costs
- **async** usually uses one thread and cooperatively switches among I/O-bound tasks at `await` points

Threads do not speed up pure-Python CPU work because the standard CPython interpreter has a Global Interpreter Lock (GIL): only one thread executes Python bytecode at a time. Threads still help with I/O because the lock is released while waiting, and many NumPy and PyTorch operations release it inside native code. Python 3.13+ also offers an optional free-threaded build without the GIL; confirm that your dependencies support it before relying on it.

`concurrent.futures` offers the same interface for both models:

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def tokenize(text: str) -> list[str]:
    return text.lower().split()

if __name__ == "__main__":
    texts = ["Hello World", "Tensors Everywhere"] * 1000

    with ThreadPoolExecutor(max_workers=8) as pool:    # suited to I/O-bound work
        _ = list(pool.map(tokenize, texts))

    with ProcessPoolExecutor(max_workers=4) as pool:   # suited to CPU-bound Python work
        tokenized = list(pool.map(tokenize, texts, chunksize=100))
```

PyTorch and NumPy kernels may use native threads internally, and GPU computation follows its own execution model. Avoid adding concurrency until measurement shows a need. Choose based on the workload, library behavior, memory cost, and failure model.

The first place this matters in ML is PyTorch's `DataLoader(..., num_workers=4)`, which loads and preprocesses batches in worker **processes**. When processes are started with `spawn` (the default on macOS and Windows) or `forkserver` (the Linux default from Python 3.14), workers re-import your main module. Training code must therefore sit under an `if __name__ == "__main__":` guard, and everything sent to workers, such as the dataset and collate function, must be picklable.

For Month 0, recognize the distinction; advanced concurrency and distributed execution come later.

## 0.1.16 — Serialization with JSON and pickle

Serialization converts data into a format that can be stored or transmitted.

JSON is interoperable and maps naturally to basic Python values:

```python
import json

config = {"model": "tiny", "temperature": 0.7}
encoded = json.dumps(config)
decoded = json.loads(encoded)
```

JSON supports objects/dictionaries, arrays/lists, strings, numbers, Booleans, and null/`None`. It does not directly encode arbitrary Python objects, tensors, or sets.

Values produced by NumPy and PyTorch are not plain Python numbers, so convert them first:

```python
import numpy as np
import torch

metrics = {"loss": torch.tensor(0.4312), "accuracy": np.float32(0.91)}
# json.dumps(metrics)   # TypeError: Object of type Tensor is not JSON serializable
safe = {name: float(value) for name, value in metrics.items()}
print(json.dumps(safe))
```

Use `.item()` for a one-element tensor and `.tolist()` for larger tensors or arrays. For multilingual text, `json.dumps(record, ensure_ascii=False)` keeps characters readable instead of escaping them.

### JSON Lines

JSONL stores one JSON object per line. It is the usual format for training examples, evaluation sets, and LLM batch jobs because records can be appended and streamed:

```python
records = [{"text": "great film", "label": 1}, {"text": "dull plot", "label": 0}]

with open("train.jsonl", "w", encoding="utf-8") as file:
    for record in records:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
```

Read it back with the `read_jsonl` generator from 0.1.8.

### Pickle

Pickle can preserve many Python-specific objects:

```python
import pickle

encoded = pickle.dumps(config)
decoded = pickle.loads(encoded)
```

Never load pickle data from an untrusted source. Unpickling can execute code. Prefer JSON or another constrained format for data crossing a trust boundary.

Model checkpoints inherit this risk because `torch.save` uses pickle. Since PyTorch 2.6, `torch.load` defaults to `weights_only=True`, which restricts unpickling to tensors and basic container types; do not turn it off for files you did not create. For distributing weights, the `safetensors` format stores only tensors and cannot execute code. Whatever the format, record the code and dependency versions needed to reproduce an artifact.

## The five Python patterns to recognize instantly

### 1. Configuration unpacking

```python
config = {"lr": 1e-4, "epochs": 10}
train(**config)
```

### 2. Model class

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
```

### 3. Typed function

```python
def predict(text: str) -> list[float]:
    ...
```

### 4. Context manager

```python
with torch.no_grad():
    output = model(inputs)
```

### 5. Async API call

```python
response = await client.generate(prompt)
```

If these five patterns feel natural, most modern AI Python code becomes much easier to read.

## Exit test A — Reading the code

Explain every part of this program without looking up Python syntax, then state what it prints.

```python
from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    model_name: str
    batch_size: int = 8
    device: str = "cpu"

class ModelRunner:
    def __init__(self, config: ModelConfig):
        self.config = config

    def run(self, inputs: list[str]) -> list[str]:
        return [text.upper() for text in inputs]

def load_config() -> ModelConfig:
    return ModelConfig(
        model_name=os.getenv("MODEL_NAME", "default-model")
    )

if __name__ == "__main__":
    config = load_config()
    runner = ModelRunner(config)
    print(runner.run(["hello", "ai"]))
```

Also explain what happens in this asynchronous function:

```python
async def call_model(prompt: str) -> str:
    response = await client.generate(prompt)
    return response
```

<details>
<summary>Show answers</summary>

### Answer guide

- `@dataclass` generates standard methods such as `__init__` from the declared fields.
- `model_name` is required. `batch_size` and `device` have defaults.
- `ModelRunner.__init__` stores a `ModelConfig` on each runner instance.
- `run` accepts a list of strings and returns a new list created by a comprehension.
- Each string is converted to uppercase.
- `load_config` reads `MODEL_NAME` from the environment and falls back to `"default-model"`.
- The `__main__` guard runs only when this file is executed directly.
- The program prints `['HELLO', 'AI']`.
- `call_model` defines a coroutine. `await` pauses it until the client's asynchronous operation completes, allowing the event loop to run other ready tasks. A caller must await or schedule `call_model`; defining it alone does not call the model.

</details>

## Exit test B — Integrated AI-code patterns

Identify the dataclass, type hints, class, constructor, instance variable, default argument, list, loop, async method, `await`, `**kwargs`, and environment variable.

```python
from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    model_name: str
    temperature: float = 0.7

class ModelRunner:
    def __init__(self, config: ModelConfig):
        self.config = config

    async def generate(self, prompts: list[str]) -> list[str]:
        results = []

        for prompt in prompts:
            result = await self.call_model(
                prompt,
                temperature=self.config.temperature,
            )
            results.append(result)

        return results

    async def call_model(self, prompt: str, **kwargs) -> str:
        return f"{prompt} - {kwargs}"

config = ModelConfig(
    model_name=os.getenv("MODEL_NAME", "my-model")
)
```

<details>
<summary>Show answers</summary>

### Answer guide

- `@dataclass` decorates `ModelConfig` and generates its initializer.
- `model_name: str`, `temperature: float`, `config: ModelConfig`, `prompts: list[str]`, and the return annotations are type hints.
- `ModelRunner` is a class; `__init__` is its constructor.
- `self.config` is an instance variable.
- `temperature` has the default value `0.7`.
- `prompts` and `results` are lists, and `for prompt in prompts` is the loop.
- `generate` and `call_model` are async methods.
- `await` waits for `call_model` without blocking the event loop from running other ready work.
- `**kwargs` collects extra keyword arguments in a dictionary. Here it receives `{"temperature": 0.7}` unless the configuration changes.
- `os.getenv("MODEL_NAME", "my-model")` reads an environment variable with a fallback.
- The snippet defines the objects but does not run `generate`. A caller must create a runner and await `runner.generate(...)`.

</details>

## Exit test C — Find the bugs

Each snippet runs, or appears to, but contains a bug that commonly reaches production ML code. Identify the bug and give the fix. Assume `torch`, `nn`, `client`, and the helper functions exist.

```python
# 1
def collect_metrics(value, history=[]):
    history.append(value)
    return history

# 2
use_gpu = bool(os.getenv("USE_GPU", "false"))

# 3
class Encoder(nn.Module):
    def __init__(self, dim: int):
        self.proj = nn.Linear(dim, dim)
        super().__init__()

# 4
async def handle(prompts: list[str]) -> list[str]:
    results = []
    for prompt in prompts:
        time.sleep(1)   # stay under the rate limit
        results.append(await client.generate(prompt))
    return results

# 5
batches = (load_batch(path) for path in batch_paths)
train_loss = sum(train_step(batch) for batch in batches)
val_loss = sum(eval_step(batch) for batch in batches)

# 6
try:
    checkpoint = torch.load(path, weights_only=False)
except:
    checkpoint = None

# 7
examples = [(text, label) for text, label in zip(texts, labels)]
```

<details>
<summary>Show answers</summary>

### Answer guide

1. **Mutable default.** One `history` list is shared by every call, so metrics from unrelated runs accumulate. Default to `None` and create the list inside the function.
2. **Truthy strings.** Any non-empty string is truthy, so `"false"` becomes `True`. Parse explicitly, for example `os.getenv("USE_GPU", "false").strip().lower() in {"1", "true", "yes"}`.
3. **Initialization order.** Assigning a submodule before `super().__init__()` raises `AttributeError`. Call `super().__init__()` first.
4. **Blocking the event loop.** `time.sleep` freezes every other task on the loop, and awaiting inside the loop makes the calls sequential anyway. Enforce rate limits with an `asyncio.Semaphore` (or `await asyncio.sleep` if a delay is truly needed) and schedule requests with `asyncio.gather`.
5. **Exhausted generator.** Training consumes `batches`, so validation iterates over nothing and `val_loss` is silently `0`. Create separate iterables. Validation should also use a held-out split, not the training batches.
6. **Two bugs.** The bare `except` hides every failure, including interrupts and genuine loading errors. `weights_only=False` allows a malicious checkpoint to execute code. Catch specific exceptions, log them, and keep the safe default.
7. **Silent truncation.** If `texts` and `labels` have different lengths, `zip` drops the extras without warning, which can misalign or lose labels. Use `zip(texts, labels, strict=True)`.

</details>

## Hands-on exercise — A data and inference harness

Build one small script, `harness.py`, that combines this lesson's patterns. Aim for under 100 lines and use only the standard library. Store it under `month-00/exercises/`, following the repository conventions.

Requirements:

1. A frozen `Settings` dataclass loaded from the environment variables `DATA_PATH` (default `examples.jsonl`), `BATCH_SIZE` (default `4`), and `MAX_CONCURRENCY` (default `2`), validated in `__post_init__`.
2. A `write_examples(path)` function that writes ten `{"text": ..., "label": ...}` records as JSONL.
3. `read_jsonl` and `batched` generators that stream the file in batches.
4. An async `fake_classify(text)` coroutine that sleeps 0.2 seconds and returns a label, and a function that classifies each batch concurrently without ever exceeding `MAX_CONCURRENCY` calls in flight.
5. A `timed` decorator or context manager that logs how long each batch takes.
6. Logging configured once in `main()`, a `ValueError` for invalid settings, and an `if __name__ == "__main__":` guard.

**Check your work:** with ten records, `BATCH_SIZE=4`, and `MAX_CONCURRENCY=2`, you should see three batches of 4, 4, and 2 records, taking about 0.4, 0.4, and 0.2 seconds. `BATCH_SIZE=0` should fail at startup with a clear `ValueError`.

<details>
<summary>Show a reference solution</summary>

```python
"""Month 0.1 exercise: stream JSONL, classify concurrently, and log timings."""
import asyncio
import json
import logging
import os
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    data_path: str
    batch_size: int
    max_concurrency: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("BATCH_SIZE must be positive")
        if self.max_concurrency <= 0:
            raise ValueError("MAX_CONCURRENCY must be positive")


def load_settings() -> Settings:
    return Settings(
        data_path=os.getenv("DATA_PATH", "examples.jsonl"),
        batch_size=int(os.getenv("BATCH_SIZE", "4")),
        max_concurrency=int(os.getenv("MAX_CONCURRENCY", "2")),
    )


def write_examples(path: str, count: int = 10) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for i in range(count):
            file.write(json.dumps({"text": f"example {i}", "label": i % 2}) + "\n")


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def batched(records: Iterable[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for record in records:
        batch.append(record)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


@contextmanager
def timed(label: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s took %.2f s", label, time.perf_counter() - start)


async def fake_classify(text: str) -> int:
    await asyncio.sleep(0.2)                     # stands in for a model API call
    return sum(ord(char) for char in text) % 2


async def classify_batch(batch: list[dict], limit: asyncio.Semaphore) -> list[int]:
    async def classify_one(record: dict) -> int:
        async with limit:
            return await fake_classify(record["text"])

    return await asyncio.gather(*(classify_one(record) for record in batch))


async def run(settings: Settings) -> None:
    limit = asyncio.Semaphore(settings.max_concurrency)
    batches = batched(read_jsonl(settings.data_path), settings.batch_size)
    for index, batch in enumerate(batches):
        with timed(f"batch {index} ({len(batch)} records)"):
            predictions = await classify_batch(batch, limit)
        logger.info("batch %d predictions=%s", index, predictions)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings()
    write_examples(settings.data_path)
    asyncio.run(run(settings))


if __name__ == "__main__":
    main()
```

</details>

## Best reading order

Use the official Python tutorial as the primary source and read only the relevant sections:

1. Data structures and control flow
2. Functions
3. Modules
4. Classes
5. Errors and exceptions
6. Iterators and generators
7. Input and output
8. Virtual environments and packages

Then use short references for dataclasses, typing, logging, and asyncio.

## Defer until needed

These topics are useful, but they have little leverage before the foundation-model portion of this roadmap:

- metaclasses and descriptors
- advanced decorators and multiple inheritance
- custom iterator implementation
- Python interpreter and garbage-collection internals
- a deep study of the Global Interpreter Lock
- advanced concurrency and packaging
- C extensions
- functional-programming theory
- advanced magic methods

## Completion criteria

0.1 is complete when:

- exit tests A and B feel straightforward, and you can find and fix all seven bugs in exit test C
- the five essential patterns are immediately recognizable
- you can explain the difference between a collection, generator, module, class, context manager, coroutine, thread, and process
- you can create an isolated environment, install a `src`-layout package in editable mode, and run it with `python -m`
- you can load and validate configuration safely and add useful logs
- your hands-on harness streams JSONL, respects the concurrency limit, and fails fast on invalid settings
- you can read ordinary typed PyTorch or LLM client code without stopping to decode the Python syntax

## Primary references

- [The Python Tutorial](https://docs.python.org/3/tutorial/)
- [Control Flow Tools](https://docs.python.org/3/tutorial/controlflow.html)
- [Data Structures](https://docs.python.org/3/tutorial/datastructures.html)
- [Modules](https://docs.python.org/3/tutorial/modules.html)
- [Classes](https://docs.python.org/3/tutorial/classes.html)
- [Errors and Exceptions](https://docs.python.org/3/tutorial/errors.html)
- [Data Classes](https://docs.python.org/3/library/dataclasses.html)
- [Typing](https://docs.python.org/3/library/typing.html)
- [Logging](https://docs.python.org/3/library/logging.html)
- [asyncio](https://docs.python.org/3/library/asyncio.html)
- [Virtual Environments and Packages](https://docs.python.org/3/tutorial/venv.html)
- [JSON](https://docs.python.org/3/library/json.html)
- [pickle security warning](https://docs.python.org/3/library/pickle.html)
- [contextlib](https://docs.python.org/3/library/contextlib.html)
- [functools](https://docs.python.org/3/library/functools.html)
- [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html)
- [Python Packaging User Guide: Writing `pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [PyTorch: `torch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
- [PyTorch: `torch.utils.data`](https://docs.pytorch.org/docs/stable/data.html)

## About this lesson

Adapted from the roadmap planning conversation, [Create Month Zero Learning List](https://chatgpt.com/share/6aa28c17-a29c-83ea-ae5b-9202ec5ba987), and extended with production pitfalls, a bug-finding exit test, and a hands-on exercise. Code examples were checked with Python 3.12, NumPy 2.4, and PyTorch 2.14 on CPU.
