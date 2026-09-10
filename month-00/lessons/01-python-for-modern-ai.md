# 0.1 — Python for Modern AI

The goal is not to become a Python language expert. It is to know enough Python that modern ML and LLM code never feels mysterious.

This lesson preserves the complete 0.1 material from the roadmap conversation: the required reading scope, detailed lessons and examples, essential patterns, exit tests, and topics to defer. It also expands the items that were named in the reading list but not demonstrated in the original walkthrough.

**Source conversation:** [Create Month Zero Learning List](https://chatgpt.com/share/6aa28c17-a29c-83ea-ae5b-9202ec5ba987)

**Recommended time:** 60–90 minutes unless the exit test reveals a gap.

## Complete coverage map

| Required area | Covered in |
|---|---|
| Data structures, comprehensions, loops, conditionals | 0.1.1 |
| Functions, arguments, lambdas, scope | 0.1.2 |
| Classes, methods, attributes | 0.1.3 |
| Inheritance and `super()` | 0.1.4 |
| Dataclasses and basic decorators | 0.1.5 |
| Type hints, `Optional`, unions, `TypedDict`, `Protocol` | 0.1.6 |
| Modules, packages, imports, entry points | 0.1.7 |
| Iterators, generators, generator expressions | 0.1.8 |
| Context managers and resource cleanup | 0.1.9 |
| Exceptions, `finally`, raising and custom errors | 0.1.10 |
| Environment variables and configuration | 0.1.11 |
| Production logging | 0.1.12 |
| Virtual environments, packages, pinning, `pyproject.toml` | 0.1.13 |
| Async, `await`, and the event loop | 0.1.14 |
| Threads versus processes | 0.1.15 |
| JSON and pickle serialization | 0.1.16 |

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
model.train(
    batch_size=16,
    learning_rate=1e-4,
    epochs=20,
)
```

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
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(768, 2)

    def forward(self, x):
        return self.layer(x)
```

PyTorch models store submodules as instance attributes and define computation in `forward`.

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

Recognize basic decorators; defer decorator factories, complex wrapping, and descriptor mechanics.

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

## 0.1.7 — Modules and packages

A module is a Python file. A package is a directory containing related modules.

```text
project/
├── train.py
├── models/
│   ├── __init__.py
│   ├── transformer.py
│   └── classifier.py
├── data/
│   └── dataset.py
└── training/
    └── trainer.py
```

An absolute import starts from the package root:

```python
from models.transformer import Transformer
```

A relative import starts from the current package:

```python
from .transformer import Transformer
from ..data.dataset import TextDataset
```

Prefer clear absolute imports in application code. Relative imports can be useful within a closely related package.

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
```

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

A `.env` file is a local convenience for defining environment variables during development. A library or launcher can load it into the process environment. Keep `.env` files containing secrets out of Git; commit an `.env.example` containing names and safe placeholders when useful.

Use typed configuration to validate settings once at startup:

```python
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    model_name: str
    device: str

def load_settings() -> Settings:
    return Settings(
        model_name=os.getenv("MODEL_NAME", "default-model"),
        device=os.getenv("DEVICE", "cpu"),
    )
```

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

## 0.1.13 — Virtual environments and packages

Virtual environments isolate one project's dependencies from another's.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch
python -m pip list
```

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
[project]
name = "ai-example"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["torch>=2.4"]
```

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

```python
import asyncio

async def main() -> None:
    first, second = await asyncio.gather(
        generate("first prompt"),
        generate("second prompt"),
    )
    print(first, second)

asyncio.run(main())
```

`async` does not make model computation faster. It is mainly useful for I/O-bound work, and blocking code inside a coroutine can still block the event loop.

## 0.1.15 — Multithreading versus multiprocessing

These are different concurrency models:

- **threads** run in one process and share memory; they are often useful for blocking I/O
- **processes** have separate Python interpreters and memory; they can run CPU-bound Python work in parallel but add serialization and coordination costs
- **async** usually uses one thread and cooperatively switches among I/O-bound tasks at `await` points

PyTorch and NumPy kernels may use native threads internally, and GPU computation follows its own execution model. Avoid adding concurrency until measurement shows a need. Choose based on the workload, library behavior, memory cost, and failure model.

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

Pickle can preserve many Python-specific objects:

```python
import pickle

encoded = pickle.dumps(config)
decoded = pickle.loads(encoded)
```

Never load pickle data from an untrusted source. Unpickling can execute code. Prefer JSON or another constrained format for data crossing a trust boundary. For model artifacts, follow the framework's safe-loading guidance and record the code and dependency versions needed to reproduce them.

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

- both exit tests feel straightforward
- the five essential patterns are immediately recognizable
- you can explain the difference between a collection, generator, module, class, context manager, coroutine, thread, and process
- you can create an isolated environment, load configuration safely, and add useful logs
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
