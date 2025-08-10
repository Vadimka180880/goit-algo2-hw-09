import random
import math
import argparse
from typing import Callable, List, Tuple, Optional

Vector = List[float]
Bounds = List[Tuple[float, float]]

# Цільова функція: Функція Сфери

def sphere_function(x: Vector) -> float:
    return sum(xi * xi for xi in x)

def random_point(bounds: Bounds) -> Vector:
    return [random.uniform(lo, hi) for lo, hi in bounds]


def clamp_to_bounds(x: Vector, bounds: Bounds) -> Vector:
    return [max(lo, min(v, hi)) for v, (lo, hi) in zip(x, bounds)]


def l2_dist(a: Vector, b: Vector) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def random_neighbor(x: Vector, bounds: Bounds, step_frac: float = 0.1) -> Vector:

    nxt = []
    for xi, (lo, hi) in zip(x, bounds):
        span = hi - lo
        step = step_frac * span
        nxt.append(xi + random.uniform(-step, step))
    return clamp_to_bounds(nxt, bounds)

# 1) Hill Climbing (координатний pattern search + мультістарт)

def hill_climbing(
    func: Callable[[Vector], float],
    bounds: Bounds,
    iterations: int = 20000,
    epsilon: float = 1e-10,
    step_frac: float = 1.0,
    restarts: int = 10,
    seed: Optional[int] = 42,
) -> Tuple[Vector, float]:

    if seed is not None:
        random.seed(seed)

    dim = len(bounds)
    spans = [hi - lo for (lo, hi) in bounds]

    def run_once() -> Tuple[Vector, float]:
        x = [random.uniform(lo, hi) for lo, hi in bounds]
        fx = func(x)
        steps = [step_frac * s for s in spans]
        it = 0

        while it < iterations:
            improved = False

            for i in range(dim):
                for sign in (+1.0, -1.0):
                    y = x[:]
                    y[i] = x[i] + sign * steps[i]
                    lo, hi = bounds[i]
                    y[i] = max(lo, min(y[i], hi))
                    fy = func(y)
                    it += 1
                    if fy + epsilon < fx:
                        x, fx = y, fy
                        improved = True
                        if it >= iterations:
                            break
                if it >= iterations:
                    break

            if not improved:
                steps = [s * 0.5 for s in steps]
                if max(steps) < 1e-6:
                    break

        return x, fx

    best_x, best_fx = None, float("inf")
    for _ in range(restarts):
        x, fx = run_once()
        if fx < best_fx:
            best_x, best_fx = x, fx

    return best_x, best_fx

# 2) Random Local Search (мультістарт + локальний дофайн-тюн)

def random_local_search(
    func: Callable[[Vector], float],
    bounds: Bounds,
    iterations: int = 5000,
    epsilon: float = 1e-8,
    restarts: int = 5,
    seed: Optional[int] = 123,
) -> Tuple[Vector, float]:

    if seed is not None:
        random.seed(seed)

    def run_once() -> Tuple[Vector, float]:
        x_best = random_point(bounds)
        f_best = func(x_best)

        for _ in range(iterations):
            cand = random_point(bounds)
            f_cand = func(cand)
            if f_cand + epsilon < f_best:
                x_best, f_best = cand, f_cand

        step = 0.05
        for _ in range(200):
            nb = random_neighbor(x_best, bounds, step_frac=step)
            f_nb = func(nb)
            if f_nb + epsilon < f_best:
                x_best, f_best = nb, f_nb
            step = max(step * 0.98, 1e-3)

        return x_best, f_best

    best_x, best_fx = None, float("inf")
    for _ in range(restarts):
        x, fx = run_once()
        if fx < best_fx:
            best_x, best_fx = x, fx

    return best_x, best_fx

# 3) Simulated Annealing (гаусівський крок + відстеження найкращого)

def simulated_annealing(
    func: Callable[[Vector], float],
    bounds: Bounds,
    iterations: int = 5000,
    temp: float = 10.0,
    cooling_rate: float = 0.99,
    epsilon: float = 1e-8,
    seed: Optional[int] = 7,
) -> Tuple[Vector, float]:

    if seed is not None:
        random.seed(seed)

    x = random_point(bounds)
    fx = func(x)
    best_x, best_fx = x[:], fx

    T0 = float(temp)
    T = T0
    spans = [hi - lo for (lo, hi) in bounds]

    for _ in range(iterations):
        if T < epsilon:
            break

        cand = []
        for xi, (lo, hi), span in zip(x, bounds, spans):
            sigma = 0.5 * (T / T0) * span  
            step = random.gauss(0.0, sigma)
            cand.append(xi + step)
        cand = clamp_to_bounds(cand, bounds)

        f_cand = func(cand)
        delta = f_cand - fx

        if delta < 0 or random.random() < math.exp(-delta / T):
            x, fx = cand, f_cand
            if fx < best_fx:
                best_x, best_fx = x[:], fx
        
        T *= cooling_rate

    return best_x, best_fx

def main():
    parser = argparse.ArgumentParser(description="Локальний пошук для мінімізації функції Сфери.")
    parser.add_argument("--dims", type=int, default=2, help="Кількість вимірів (n). За замовчуванням 2.")
    args = parser.parse_args()

    bounds: Bounds = [(-5, 5)] * args.dims

    print("Hill Climbing:")
    hc_solution, hc_value = hill_climbing(
        sphere_function, bounds,
        iterations=20000, epsilon=1e-10, step_frac=1.0, restarts=10, seed=42
    )
    print("Розв'язок:", hc_solution, "Значення:", hc_value)

    print("\nRandom Local Search:")
    rls_solution, rls_value = random_local_search(
        sphere_function, bounds,
        iterations=5000, epsilon=1e-8, restarts=5, seed=123
    )
    print("Розв'язок:", rls_solution, "Значення:", rls_value)

    print("\nSimulated Annealing:")
    sa_solution, sa_value = simulated_annealing(
        sphere_function, bounds,
        iterations=5000, temp=10.0, cooling_rate=0.99, epsilon=1e-8, seed=7
    )
    print("Розв'язок:", sa_solution, "Значення:", sa_value)


if __name__ == "__main__":
    main()
