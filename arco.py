#!/usr/bin/env python3

from collections import defaultdict
from itertools import product

import heapq
from dataclasses import dataclass

ARCO_FIELDS = ['lamb', 'xi', 'zeta', 'theta', 'epsilon', 'phi', 'gamma', 'omega']

@dataclass(frozen=True, order=True)
class Arcos:
    lamb: int = 0
    xi: int = 0
    zeta: int = 0
    theta: int = 0
    epsilon: int = 0
    phi: int = 0
    gamma: int = 0
    omega: int = 0


    def add(self, t: 'Arcos') -> 'Arcos':
        return Arcos(
            lamb=self.lamb + t.lamb,
            xi=self.xi + t.xi,
            zeta=self.zeta + t.zeta,
            theta=self.theta + t.theta,
            epsilon=self.epsilon + t.epsilon,
            phi=self.phi + t.phi,
            gamma=self.gamma + t.gamma,
            omega=self.omega + t.omega
        )


    def sub(self, t: 'Arcos') -> 'Arcos':
        return Arcos(
            lamb=self.lamb - t.lamb,
            xi=self.xi - t.xi,
            zeta=self.zeta - t.zeta,
            theta=self.theta - t.theta,
            epsilon=self.epsilon - t.epsilon,
            phi=self.phi - t.phi,
            gamma=self.gamma - t.gamma,
            omega=self.omega - t.omega
        )


    def error(self, t: 'Arcos') -> int:
        # mean squared error of this away from t
        diff = self.sub(t)
        return pow(diff.lamb, 2) + pow(diff.xi, 2) + pow(diff.zeta, 2) + pow(diff.theta, 2) + pow(diff.epsilon, 2) + pow(diff.phi, 2) + pow(diff.gamma, 2) + pow(diff.omega, 2)


    def val(self, field) -> int:
        assert field in ARCO_FIELDS
        return getattr(self, field)


    def repr_as_transform(self) -> str:
        negs = []
        pos = []
        for field in ARCO_FIELDS:
            if self.val(field) < 0:
                negs.append(field + ": " + str(self.val(field)))
            elif self.val(field) > 0:
                pos.append(field + ": " + str(self.val(field)))
        return "Transform{" + ", ".join(negs) + " into " + ", ".join(pos) + "}"


    def __repr__(self) -> str:
        fields = []
        for field in ARCO_FIELDS:
            v = self.val(field)
            if v != 0:
                fields.append(field + ": " + str(v))

        return "{"+', '.join(fields)+"}"


research = {
    'Research1A': Arcos(zeta=-1, omega=-1, lamb=2),
    'Research1B': Arcos(zeta=-1, omega=-1, phi=2),
    'Research2A': Arcos(lamb=-1, xi=-1, zeta=1, theta=1),
    'Research2B': Arcos(lamb=-1, xi=-1, epsilon=1, phi=1),
    'Research3A': Arcos(theta=-1, gamma=-1, epsilon=2),
    'Research3B': Arcos(theta=-1, gamma=-1, zeta=2),
    'Research4A': Arcos(epsilon=-1, phi=-1, zeta=1, theta=1),
    'Research4B': Arcos(epsilon=-1, phi=-1, omega=1, gamma=1),
    'TesseractA': Arcos(lamb=-1, xi=-1, zeta=-1, theta=1, epsilon=1, phi=1),
    'TesseractB': Arcos(lamb=-1, xi=-1, zeta=-1, phi=1, gamma=1, omega=1),
    'ProcessorA': Arcos(zeta=-1, theta=-1, epsilon=-1, phi=-1, gamma=-1, omega=-1, lamb=5, xi=1)
    'ProcessorB': Arcos(zeta=-1, theta=-1, epsilon=-1, phi=-1, gamma=-1, omega=-1, lamb=1, xi=5)
    'Research5A': Arcos(lamb=-1, zeta=-1, epsilon=-1, gamma=-1, xi=1, theta=1, phi=1, omega=1)
}


transforms = {
    'InversionA': Arcos(lamb=1, xi=1, epsilon=1, phi=1, zeta=-1, theta=-1, gamma=-1, omega=-1),
    'InversionB': Arcos(lamb=-1, xi=-1, epsilon=-1, phi=-1, zeta=1, theta=1, gamma=1, omega=1),
    'FoldA': Arcos(lamb=-1, omega=-1, xi=1, theta=1),
    'FoldB': Arcos(xi=-1, gamma=-1, lamb=1, zeta=1),
    'FoldC': Arcos(xi=-1, zeta=-1, theta=1, phi=1),
    'FoldD': Arcos(lamb=-1, theta=-1, zeta=1, epsilon=1),
    'FoldE': Arcos(theta=-1, epsilon=-1, phi=1, omega=1),
    'FoldF': Arcos(zeta=-1, phi=-1, epsilon=1, gamma=1),
    'FoldG': Arcos(phi=-1, gamma=-1, xi=1, omega=1),
    'FoldH': Arcos(epsilon=-1, omega=-1, lamb=1, gamma=1),
}


def solve(name, start, end):
    print("Solving: ", name)
    path = dijkstra2(start, end)
    if path:
        print_path(start, path)
    else:
        print("Could not find path :/")


def dijkstra2(start: Arcos, dest: Arcos, max_steps = 30000) -> list[str]:
    print("Searching from:", start, "to", dest)
    i = 0

    results = []

    seen: set[Arcos] = set()

    steps: dict[Arcos, int] = {}
    steps[start] = 0

    path: dict[Arcos, list[str]] = {}
    path[start] = []

    pq = [(0, start)]
    while len(pq) > 0:
        ce, current = heapq.heappop(pq)
        if current in seen:
            continue
        seen.add(current)

        for tn, t in transforms.items():
            n = current.add(t)

            if n not in seen:
                # first time we've seen this combination
                error = n.error(dest)
                #heapq.heappush(pq, (error, n))
                heapq.heappush(pq, (steps[current] + 1, n))
                steps[n] = steps[current] + 1
                path[n] = path[current] + [tn]

            if n == dest:
                print("Found one")
                print_path(start, path[n])

        i += 1
        if i > max_steps:
            return results
    return []


def normalize_path(start: Arcos, path: list[str]) -> Arcos:
    base = defaultdict(int)
    c = start
    for t in path:
        c = c.add(transforms[t])
        for field in ARCO_FIELDS:
            if c.val(field) < 0:
                base[field] = max(base[field], -c.val(field))
    return Arcos(**base)


def print_path(start: Arcos, path: list[str]) -> None:
    base = normalize_path(start, path)

    print("Start:", start, " (plus byproducts:", base, ")")
    c = start.add(base)
    for t in path:
        print("apply", t, "->", transforms[t].repr_as_transform())
        c = c.add(transforms[t])
        print("-> gives", c)
    print("Bingo")


def main() -> None:
    c = Arcos(phi=4, xi=1, theta= 1, epsilon=1, gamma=2, omega=1)
    print(c)
    print("Apply FoldG 2x")
    c = c.add(transforms['FoldG'])
    c = c.add(transforms['FoldG'])
    c = c.add(transforms['InversionB'])
    print(c)

    #solve("Research 1, left RNG * 2", Arcos(lamb=4), Arcos(zeta=2, omega=2))
    #solve("Research 1, right RNG * 2", Arcos(phi=4), Arcos(zeta=2, omega=2))
    #solve("Research 1, left+right RNG", Arcos(lamb=2, phi=2), Arcos(zeta=2, omega=2))
    #print()

    #solve("Research 2, left RNG * 2", Arcos(zeta= 2, theta= 2), Arcos(lamb= 2, xi= 2))
    #solve("Research 2, right RNG * 2", Arcos(epsilon= 2, phi= 2), Arcos(lamb= 2, xi= 2))
    #solve("Research 2, left+right RNG", Arcos(zeta= 1, theta= 1, epsilon= 1, phi= 1), Arcos(lamb= 2, xi= 2))
    #print()


if __name__ == '__main__':
    main()
