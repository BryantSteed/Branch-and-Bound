from typing import List, Tuple, Callable
import time, math, matplotlib.pyplot as plt
from pathlib import Path

def compute_runtimes(sizes: List[int], func: Callable[[int], bool]) -> List[float]:
    runtimes = []
    realized_sizes = []
    for size in sizes:
        start_time = time.perf_counter()
        succeeded = func(size)
        end_time = time.perf_counter()
        if succeeded:
            realized_sizes.append(size)
            runtimes.append((end_time - start_time) * 1000)  # Convert to milliseconds
    sizes[:] = realized_sizes  # Update sizes to only those that succeeded
    return runtimes

def compute_constants(sizes: List[int], runtimes: List[float], theoretical: Callable[[int], float]) -> List[float]:
    constants = []
    for size, runtime in zip(sizes, runtimes):
        theo_value = theoretical(size)
        constants.append(runtime / theo_value)
    return constants

def average_constant(constants: List[float]) -> float:
    return sum(constants) / len(constants) if constants else 0.0

def run_analysis(sizes: List[int], func: Callable[[int], bool]):
    """Runs performance analysis on the given function over specified input sizes.
    The function `func` should accept an integer size and return True if it completed successfully.
    It will mutate the sizes list to only include sizes for which the function succeeded. Similarly,
    the runtimes list will only include runtimes for successful sizes."""
    runtimes = compute_runtimes(sizes, func) # Mutates to only successful ones

    theoretical_funcs: List[Tuple[Callable[[int], float], str]] = get_theoretical_funcs()
    avg_constants = []
    for theoretical, label in theoretical_funcs:
        constants = compute_constants(sizes, runtimes, theoretical)
        avg_const = average_constant(constants)
        avg_constants.append(avg_const)
        plot_constants(sizes, constants, avg_const, label)
    
    plot_times_vs_sizes(sizes, runtimes, theoretical_funcs, avg_constants)

    save_path: Path = generate_save_path("runtimes_table")
    with open(save_path.with_suffix('.txt'), 'w') as f:
        for (runtime, size) in zip(runtimes, sizes):
            f.write(f"Size: {size}, Runtime: {runtime:.2f} ms\n")

def plot_times_vs_sizes(sizes: List[int], runtimes: List[float], theoretical_funcs: List[Tuple[Callable[[int], float], str]], avg_constants: List[float]):
    plt.clf()
    plt.plot(sizes, runtimes, 'o-', label='Measured Runtimes', color='black')

    for (theoretical, label), avg_const in zip(theoretical_funcs, avg_constants):
        theo_runtimes = [theoretical(size) * avg_const for size in sizes]
        plt.plot(sizes, theo_runtimes, '--', label=f'Theoretical {label}')

    plt.xlabel('Input Size')
    plt.ylabel('Runtime (ms)')
    plt.title('Runtimes vs Input Sizes')
    plt.legend()

    save_path = generate_save_path("runtimes_vs_sizes")
    plt.savefig(save_path)

def plot_constants(sizes: List[int], constants: List[float], avg_const: float, func_label: str):
    plt.clf()
    plt.bar(range(len(constants)), constants)
    plt.axhline(y=avg_const, color='r', linestyle='--', label=f'Average Constant: {avg_const}')
    plt.ylabel('Constant')
    plt.title(f'Constants for {func_label}')
    plt.legend()

    save_path = generate_save_path("constants_" + func_label)
    plt.savefig(save_path)

def generate_save_path(label):
    outdir = Path("empirical_stuff")
    outdir.mkdir(exist_ok=True)
    save_label = "".join(c if c.isalnum() else "_" for c in label)
    save_path = outdir / f"{save_label}.png"
    return save_path

def get_theoretical_funcs() -> List[Tuple[Callable[[int], float], str]]:
    funcs = [
        # (lambda n: 1, "O(1)"),
        # (lambda n: math.log(n), "O(log n)"),
        # (lambda n: n, "O(n)"),
        # (lambda n: n * math.log(n), "O(n log n)"),
        # (lambda n: pow(n, 2), "O(n^2)"),
        # (lambda n: pow(n, 3), "O(n^3)"),
        # (lambda n: pow(2, n), "O(2^n)"),
        (lambda n: pow(3.1, n), "O(3.1^n)"),
        (lambda n: pow(3.2, n), "O(3.2^n)"),
        (lambda n: pow(3.3, n), "O(3.3^n)"),
        (lambda n: pow(3, n) * pow(n, 3), "O(3^n * n^3)"),
        (lambda n: pow(4, n), "O(4^n)"),
        # (lambda n: math.factorial(n), "O(n!)"),
        # (lambda n: math.factorial(n) * pow(n, 2), "O(n! * n^2)"),
    ]
    return funcs