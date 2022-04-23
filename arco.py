#!/usr/bin/env python3

from typing import List

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
        return Arcos(**dict((f, v + t.val(f)) for f, v in self.vals()))

    def addOne(self, f) -> 'Arcos':
        unit = dict()
        unit[f] = 1
        return self.add(Arcos(**unit))

    def sub(self, t: 'Arcos') -> 'Arcos':
        return Arcos(**dict((f, v - t.val(f)) for f, v in self.vals()))

    def neg(self) -> 'Arcos':
        return self.mul(-1)

    def mul(self, n) -> 'Arcos':
        return Arcos(**dict((f, v * n) for f, v in self.vals()))

    def inputs(self) -> 'Arcos':
        return Arcos(**dict((f, v) for f, v in self.vals() if v < 0))

    def outputs(self) -> 'Arcos':
        return Arcos(**dict((f, v) for f, v in self.vals() if v > 0))

    def error(self, t: 'Arcos') -> int:
        # mean squared error of this away from t
        diff = self.sub(t)
        total = 0
        for f, v in diff.vals():
            total += pow(v, 2)
        return total


    def empty(self) -> bool:
        for f, v in self.vals():
            if v != 0:
                return False
        return True


    def val(self, field) -> int:
        assert field in ARCO_FIELDS
        return getattr(self, field)


    def vals(self):
        for field in ARCO_FIELDS:
            yield field, getattr(self, field)


    def nonEmptyVals(self):
        for field in ARCO_FIELDS:
            v = getattr(self, field)
            if v != 0:
                yield field, v


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
    'ProcessorA': Arcos(zeta=-1, theta=-1, epsilon=-1, phi=-1, gamma=-1, omega=-1, lamb=5, xi=1),
    'ProcessorB': Arcos(zeta=-1, theta=-1, epsilon=-1, phi=-1, gamma=-1, omega=-1, lamb=1, xi=5),
    'Research5A': Arcos(lamb=-1, zeta=-1, epsilon=-1, gamma=-1, xi=1, theta=1, phi=1, omega=1),
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
    return path


def dijkstra2(start: Arcos, dest: Arcos, max_steps = 30000) -> List[str]:
    print("Searching from:", start, "to", dest)
    i = 0

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
                return path[n]

        i += 1
        if i > max_steps:
            return []
    return []


def normalize_path(start: Arcos, path: List[str]) -> Arcos:
    base = defaultdict(int)
    c = start
    for t in path:
        c = c.add(transforms[t])
        for field in ARCO_FIELDS:
            if c.val(field) < 0:
                base[field] = max(base[field], -c.val(field))
    return Arcos(**base)


def print_path(start: Arcos, path: List[str]) -> None:
    base = normalize_path(start, path)

    print("Start:", start, " (plus byproducts:", base, ")")
    c = start.add(base)
    for t in path:
        print("apply", t, "->", transforms[t].repr_as_transform())
        c = c.add(transforms[t])
        print("-> gives", c)
    print("Bingo")


@dataclass
class Machine:
    name: str
    recipe: 'Arcos' = Arcos()
    running = False
    runCount = 0

    def __init__(self, name, recipe):
        self.name = name
        if recipe is None:
            recipe = Arcos()
        self.inp = recipe.inputs().neg()
        self.out = recipe.outputs()
        self.connections_in = defaultdict(list)
        self.connections_out = defaultdict(list)
        self.contents = Arcos()

    def check_connections(self):
        for f, v in self.inp.nonEmptyVals():
            if f not in self.connections_in:
                print("Missing connection to", self.name, "for", f)
            elif len(self.connections_in[f]) != v:
                print("Incorrect connection count to", self.name, "for", f, "(expected: ", v, "was:", len(self.connections_in[f]),")")

        for f, v in self.out.nonEmptyVals():
            if f not in self.connections_out:
                print("Missing connection from", self.name, "for", f)
            elif len(self.connections_out[f]) != v:
                print("Incorrect connection count from", self.name, "for", f)


    def connect(self, other, f):
        c = Connection(self, other, type=f)
        other.connections_in[f].append(c)
        self.connections_out[f].append(c)

    def fill(self, contents):
        self.contents = contents


    def check_can_run(self, n = 0):
        self.running = False
        if not self.inp.empty():
            if self.contents == self.inp:
                self.running = True
        if self.inp.empty() and (n % 10) == 1:
            self.running = True


    def step(self, n = 0):
        if self.running:
            print(self.name, "running")
            self.contents = self.contents.sub(self.inp)
            self.runCount += 1

            for f, outs in self.connections_out.items():
                for out in outs:
                    out.dest.contents = out.dest.contents.addOne(f)
        else:
            state = "waiting on input"
            if not self.inp.empty() and self.contents == self.inp:
                state = "ready to run"

            if self.runCount > 0:
                print(self.name, state, "ran:", self.runCount, "(holding: ", self.contents, ", missing: ", self.inp.sub(self.contents), ")")
            else:
                print(self.name, state, "(holding: ", self.contents, ", missing: ", self.inp.sub(self.contents), ")")
        self.running = False


@dataclass
class Connection:
    source: 'Machine'
    dest: 'Machine'
    type: str


def simulate(path):
    r = research['Research1B'].mul(2)
    machines = [
        Machine('Machine0', recipe=r.outputs()),
        Machine('Machine1', recipe=transforms['FoldB']),
        Machine('Machine2', recipe=transforms['FoldG']),
        Machine('Machine3', recipe=transforms['FoldG']),
        Machine('Machine4', recipe=transforms['FoldF']),
        Machine('Machine5', recipe=transforms['FoldD']),
        Machine('Machine6', recipe=transforms['InversionB']),
        Machine('Machine7', recipe=transforms['FoldH']),
        Machine('Machine8', recipe=r.inputs()),
    ]

    machines[0].connect(machines[2], 'phi')
    machines[0].connect(machines[3], 'phi')
    machines[0].connect(machines[4], 'phi')
    machines[0].connect(machines[6], 'phi')

    machines[1].connect(machines[5], 'lamb')
    machines[1].connect(machines[4], 'zeta')
    machines[1].fill(Arcos(gamma=1))

    machines[2].connect(machines[6], 'xi')
    machines[2].connect(machines[7], 'omega')
    machines[2].fill(Arcos(gamma=1))

    machines[3].connect(machines[1], 'xi')
    machines[3].connect(machines[8], 'omega')
    machines[3].fill(Arcos(gamma=1))

    machines[4].connect(machines[6], 'epsilon')
    machines[4].connect(machines[1], 'gamma')

    machines[5].connect(machines[8], 'zeta')
    machines[5].connect(machines[7], 'epsilon')

    machines[6].connect(machines[8], 'zeta')
    machines[6].connect(machines[5], 'theta')
    machines[6].connect(machines[2], 'gamma')
    machines[6].connect(machines[8], 'omega')
    machines[6].fill(Arcos(lamb=1))

    machines[7].connect(machines[6], 'lamb')
    machines[7].connect(machines[3], 'gamma')

    for m in machines:
        m.check_connections()


    for step in range(0, 40):
        print("STEP: ", step)
        for m in machines:
            m.check_can_run(step)
        for m in machines:
            m.step(step)
            step += 1





def main() -> None:
    r = research['Research1B']
    r = r.mul(2)
    path = solve("Research 1, right RNG * 2", r.outputs(), r.inputs().neg())
    simulate([r.outputs()] + [transforms[p] for p in path] + [r.inputs()])


if __name__ == '__main__':
    main()
