# 0.1 — Python for Modern AI

The goal is fluency with the Python patterns that appear repeatedly in ML and LLM code. This is a focused review, not a comprehensive Python course.

## 1. Collections and comprehensions

Know when to use lists, dictionaries, tuples, and sets, and read comprehensions comfortably.

```python
models = ["gpt", "bert", "dinov3"]
config = {"lr": 1e-4, "epochs": 10}
shape = (32, 512, 768)
classes = {"cat", "dog", "bird"}

squares = [x**2 for x in range(5)]
metrics = {name: 0.0 for name in ["loss", "accuracy"]}
positive = [x for x in squares if x > 0]
```

In AI code, dictionaries commonly hold configuration, batches, metrics, JSON requests, and model outputs. Tuples frequently represent shapes. Sets provide uniqueness and fast membership checks.

## 2. Functions and argument passing

Recognize required, default, positional, and keyword arguments. Understand how `*args` collects positional arguments and `**kwargs` collects keyword arguments.

```python
def train_model(model: str, epochs: int = 10, **kwargs) -> None:
    print(model, epochs, kwargs)

config = {"model": "bert", "epochs": 20, "learning_rate": 1e-4}
train_model(**config)
```

Configuration unpacking with `**config` is especially common in AI libraries.

## 3. Classes, inheritance, and dataclasses

Most model libraries use classes. Understand `__init__`, `self`, methods, inheritance, and `super()`.

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_size: int
    output_size: int

class Model:
    def __init__(self, config: ModelConfig):
        self.config = config

    def predict(self, value: float) -> float:
        return value
```

The PyTorch pattern below should look natural:

```python
class Classifier(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, x):
        return self.output(x)
```

## 4. Type hints

Type hints describe inputs, outputs, and object structure. They usually do not enforce types at runtime.

```python
def tokenize(text: str) -> list[int]:
    return [1, 2, 3]

def load_model(path: str | None = None) -> object:
    ...
```

Recognize primitive types, `list[...]`, `dict[...]`, unions with `|`, and older `Optional`/`Union` syntax. Be aware of `TypedDict` and `Protocol` when encountered.

## 5. Modules, packages, and entry points

A module is a Python file; a package groups modules in a directory. Understand absolute and relative imports and this execution guard:

```python
def main() -> None:
    print("training")

if __name__ == "__main__":
    main()
```

The guard runs `main()` when the file is executed directly, but not when another module imports it.

## 6. Iterators and generators

Generators yield values one at a time, which supports streaming and avoids loading an entire dataset into memory.

```python
def batches(records, batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]
```

Recognize `iter`, `next`, `yield`, and generator expressions.

## 7. Context managers

Context managers acquire and clean up resources automatically.

```python
with open("config.json", encoding="utf-8") as file:
    config_text = file.read()

with torch.no_grad():
    predictions = model(inputs)
```

Know the purpose of `with`; only conceptual awareness of `__enter__` and `__exit__` is required.

## 8. Exceptions

Catch specific errors and raise clear exceptions when inputs or configuration are invalid.

```python
try:
    model = load_model("model.pt")
except FileNotFoundError as error:
    logger.error("Model is missing: %s", error)

if batch_size <= 0:
    raise ValueError("batch_size must be positive")
```

Avoid bare `except` blocks that silently hide failures.

## 9. Environments and packaging

Know how to create a virtual environment, install packages, pin versions, and restore dependencies. Recognize both `requirements.txt` and `pyproject.toml`; deep packaging knowledge is unnecessary at this stage.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install package-name
```

## 10. Configuration, secrets, and serialization

Keep secrets and environment-specific configuration outside source code.

```python
import os

model_name = os.getenv("MODEL_NAME", "default-model")
api_key = os.environ["API_KEY"]
```

Know how JSON maps to Python collections. Use pickle only for trusted data because loading a pickle can execute code.

## 11. Logging

Use structured severity levels instead of relying on `print()` in production code.

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Training started")
```

Recognize `debug`, `info`, `warning`, `error`, and `exception`.

## 12. Async and concurrency concepts

Async helps programs make progress while waiting for network or other I/O. It does not make model computation faster.

```python
async def generate(prompt: str) -> str:
    response = await client.generate(prompt)
    return response
```

At a conceptual level, distinguish async I/O, threads, and processes:

- async: many waiting I/O operations in one event loop
- threads: useful for blocking I/O and libraries that release the interpreter lock
- processes: separate interpreters and memory, useful for CPU-bound parallel work

## Patterns to recognize instantly

1. `train(**config)` — configuration unpacking
2. `class Model(nn.Module)` plus `super().__init__()` — model inheritance
3. `def predict(text: str) -> list[float]` — typed function
4. `with torch.no_grad()` — temporary inference context
5. `await client.generate(prompt)` — asynchronous I/O

## Exit test

Explain every marked concept in this program without looking up Python syntax:

```python
from dataclasses import dataclass
import os

@dataclass
class ModelConfig:                         # dataclass and type hints
    model_name: str
    temperature: float = 0.7              # default value

class ModelRunner:                         # class
    def __init__(self, config: ModelConfig):
        self.config = config              # instance attribute

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

If the code feels straightforward, 0.1 is complete.

## Primary references

- [The Python Tutorial](https://docs.python.org/3/tutorial/)
- [Data Classes](https://docs.python.org/3/library/dataclasses.html)
- [Typing](https://docs.python.org/3/library/typing.html)
- [Logging](https://docs.python.org/3/library/logging.html)
- [asyncio](https://docs.python.org/3/library/asyncio.html)
- [Virtual Environments and Packages](https://docs.python.org/3/tutorial/venv.html)

