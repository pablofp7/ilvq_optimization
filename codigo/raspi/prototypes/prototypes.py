import numpy as np


class PrototypeBuffer:



    def __init__(self):
        self.n_features: int = -1
        self._size: int = 0
        self._prototypes: dict = dict()
        self._edge_age: dict = dict()
        self.classes: set = set()

    def __len__(self):
        return len(self._prototypes)

    def append(self, x: np.array, y) -> None:
        """
        Append a new prototype to the prototype buffer
        """
        if self._size == 0:  # initialize the shape of prototypes
            self.n_features = x.shape[0]

        # verify consistency of number of features
        if x.shape[0] != self.n_features:
            raise ValueError("Inconsistent number of features in x: {} previously observed {}."
                             .format(x.shape[0], self.n_features))

        self._size += 1
        self._prototypes[self._size] = {'x': x, 'y': y, 'm': 0, 'neighbors': list()}
        self.classes.add(y)

    @staticmethod
    def get_distance(a: np.array, b: np.array) -> float:
        """
        Get the distance between two vectors
        """
        distance = np.linalg.norm(a - b)
        return distance

    def find_nearest(self, x: np.array, n: int):
        """
        Get the n-closest prototypes
        """
        distances = [(index, self.get_distance(x, prototype['x'])) for index, prototype in self._prototypes.items()]
        for _ in range(n):
            s, d = min(distances, key=lambda z: z[1])
            distances.remove((s, d))
            yield s, d, self._prototypes[s]['y']

    def compute_threshold(self, prototype: int) -> float:
        """
        Computes the dynamical threshold value for a prototype
        """
        within, between = list(), list()

        for a, b in self._edge_age.keys():
            if prototype in (a, b) and self.prototypes[a]['y'] == self.prototypes[b]['y']:
                #within.append(self.get_distance(a, b))
                within.append(self.get_distance(self.prototypes[a]['x'], self.prototypes[b]['x']))
            elif prototype in (a, b) and self.prototypes[a]['y'] != self.prototypes[b]['y']:
                between.append(self.get_distance(self.prototypes[a]['x'], self.prototypes[b]['x']))
                #between.append(self.get_distance(a, b))
        between.sort(reverse=True)

        if not between and not within:  # isolated prototype
            # print('nan')
            return np.nan
        elif not between:
            return float(np.max(within))
        elif not within:
            t_between = between.pop()
            return t_between

        t_within = float(np.mean(within))
        while True:
            t_between = between.pop()
            if t_within < t_between or len(between) == 0:
                return t_between

    def compute_threshold_v2(self, prototype: int) -> float:
        """
        Computes the dynamical threshold value for a prototype
        """
        within, between = list(), list()

        for a, b in self._edge_age.keys():
            if prototype in (a, b) and self.prototypes[a]['y'] == self.prototypes[b]['y']:
                within.append(self.get_distance(a, b))
            elif prototype in (a, b) and self.prototypes[a]['y'] != self.prototypes[b]['y']:
                between.append(self.get_distance(a, b))
        between.sort(reverse=True)

        if not between and not within:  # isolated prototype
            return np.nan
        elif not between:
            return np.nan #float(np.max(within))
        elif not within:
            t_between = between.pop()
            return np.nan #t_between

        t_within = float(np.mean(within))
        while True:
            t_between = between.pop()
            if t_within < t_between:
                return t_between
            elif len(between) == 0:
                return np.nan


    def update_winner_m(self, s1: int) -> None:
        """
        Update the value m of a given prototype
        """
        self._prototypes[s1]['m'] += 1

    def update_positions(self, x: np.array, y, s1: int, alpha_winner: float, alpha_runner: float) -> None:
        """
        Updates the positions of the set of prototypes
        """
        if y == self._prototypes[s1]['y']:  # same label
            self._prototypes[s1]['x'] += alpha_winner * (x - self._prototypes[s1]['x'])  #
            for neighbor in self._prototypes[s1]['neighbors']:
                if self._prototypes[neighbor]['y'] != y:
                    self._prototypes[neighbor]['x'] -= alpha_runner * (x - self._prototypes[neighbor]['x'])

        else:  # distinct label
            self._prototypes[s1]['x'] -= alpha_winner * (x - self._prototypes[s1]['x'])
            for neighbor in self._prototypes[s1]['neighbors']:
                if self._prototypes[neighbor]['y'] == y:
                    self._prototypes[neighbor]['x'] += alpha_runner * (x - self._prototypes[neighbor]['x'])

    def create_edge(self, s1: int, s2: int) -> None:
        """
        Create a new edge between two given prototypes s1 and s2
        """
        self._edge_age[(s1, s2)] = 0  # creates the key (s1, s2) and sets its value age to zero
        self._prototypes[s1]['neighbors'].append(s2)  # adds the runner-up to the neighbors of the winner
        self._prototypes[s2]['neighbors'].append(s1)  # adds the winner to the neighbors of the runner-up

    def update_edge_age(self, s1: int) -> None:
        """
        Updates the age of all the incident edges of the winner
        """
        for edge in self._edge_age.keys():
            if s1 in edge:
                self._edge_age[edge] += 1

    def delete_old_edges(self, age_old: int) -> None:
        """
        Removes the edges that exceed the age limit
        """
        old_edges = {edge for edge, age in self._edge_age.items() if age > age_old}
        for s1, s2 in old_edges:
            del self._edge_age[(s1, s2)]  # removes the edge from the age dictionary
            self._prototypes[s1]['neighbors'].remove(s2)  # removes s2 as a neighbor of s1
            self._prototypes[s2]['neighbors'].remove(s1)  # removes s1 as a neighbor of s2

    def denoise(self) -> None:
        """
        Removes noisy prototypes
        """
        for index, prototype in [(index, prototype) for index, prototype in self._prototypes.items()
                                 if len(prototype['neighbors']) < 2]:
            if len(prototype['neighbors']) == 0:
                del self._prototypes[index]
            elif prototype['m'] < ((1 / 2 * len(self._prototypes)) * sum(v['m'] for v in self._prototypes.values())):
                neighbor = prototype['neighbors'].pop()
                del self._prototypes[index]
                self._prototypes[neighbor]['neighbors'].remove(index)
                self._edge_age = {edge: age for edge, age in self._edge_age.items() if index not in edge}

    @property
    def prototypes(self):
        return self._prototypes

    @property
    def edges(self):
        return self._edge_age.keys()
