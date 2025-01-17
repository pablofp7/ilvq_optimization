import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


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
            self._prototypes[s1]['neighbors'].remove(s2)
            self._prototypes[s2]['neighbors'].remove(s1)
    


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
    
    
    #################################################### NEW ####################################################
    
    def dbscan_prototypes(self, max_prototypes=100, target_range=(80, 90), eps_initial=0.000001) -> float:
        original_count = len(self._prototypes)
        if original_count <= max_prototypes:
            print("No need to run DBSCAN.")
            return
        
        original_prototypes = self._prototypes.copy()
        target_min, target_max = int(target_range[0] * original_count / 100), int(target_range[1] * original_count / 100)
        
        min_samples = 1
        iterations = 0
        previous_value = original_count
        ajuste_grueso = True
        
        if eps_initial is None:
            eps_initial = 0.000001

        eps = eps_initial
        last_eps = eps_initial
        lower_eps = eps_initial
        upper_eps = eps_initial

        while True:
            new_prototypes = {}
            next_prototype_id = 1

            for label in set(proto['y'] for proto in original_prototypes.values()):
                label_prototypes = np.array([proto['x'] for proto in original_prototypes.values() if proto['y'] == label])
                if label_prototypes.size == 0:
                    continue

                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(label_prototypes)

                for cluster_id in np.unique(labels):
                    cluster_indices = np.where(labels == cluster_id)[0]
                    cluster_protos = [label_prototypes[i] for i in cluster_indices]
                    centroid = np.mean(cluster_protos, axis=0)

                    sum_m = sum(original_prototypes[list(original_prototypes.keys())[i]]['m'] for i in cluster_indices)

                    new_prototypes[next_prototype_id] = {
                        'x': centroid,
                        'y': label,
                        'm': sum_m,
                        'neighbors': []
                    }
                    next_prototype_id += 1

            current_prototype_count = len(new_prototypes)
            # print(f"Prototypes after DBSCAN iteration {iterations}: {current_prototype_count}. Objective: {target_min} - {target_max}")
            
            if target_min <= current_prototype_count <= target_max:
                # print(f"Prototypes within target range after {iterations} iterations. {current_prototype_count} prototypes.")
                break
            
            if ajuste_grueso:
                if current_prototype_count > target_max:
                    if previous_value > target_max:
                        last_eps = eps
                        eps *= 10
                        # print(f"Ajuste grueso, eps = {eps}. Valores condi: Proto tras dbscan: {current_prototype_count}, target_max: {target_max}. Previous value: {previous_value}")
                    else:
                        ajuste_grueso = False
                        upper_eps = eps
                        lower_eps = last_eps

                    previous_value = current_prototype_count

                elif current_prototype_count < target_min:
                    if previous_value < target_min:
                        last_eps = eps
                        eps /= 10
                        # print(f"Ajuste grueso, eps = {eps}. Valores condi: Proto tras dbscan: {current_prototype_count}, target_min: {target_min}. Previous value: {previous_value}")
                    else:
                        ajuste_grueso = False
                        lower_eps = eps
                        upper_eps = last_eps
            
                    previous_value = current_prototype_count
            
            else:
                    
                if current_prototype_count > target_max:
                    upper_eps = eps
                elif current_prototype_count < target_min:
                    lower_eps = eps
                
                new_eps = (lower_eps + upper_eps) / 2
                if new_eps == eps:
                    ajuste_grueso = True
                eps = new_eps
                    
            
            if iterations > 100:
                raise ValueError("Maximum number of iterations (100) reached. There is a problem.")

            
            iterations += 1

        self.prototypes = new_prototypes
        return eps 
    
    
        
    def kmeans_prototypes(self, max_prototypes: int = 100, target_percentage: int = 90):
        original_count = len(self._prototypes)
        if original_count <= max_prototypes:
            print(f"No need to run K-Means. Original count: {original_count} is less than or equal to max prototypes: {max_prototypes}.")
            return

        target_prototypes = int(max_prototypes * (target_percentage / 100))
        # print(f"Running K-Means. Original count: {original_count}, max prototypes: {max_prototypes}, target prototypes: {target_prototypes}.")

        original_prototypes = self._prototypes.copy()
        new_prototypes = {}
        next_prototype_id = 1

        # Determine the number of clusters per label
        label_prototypes_count = {label: sum(1 for proto in original_prototypes.values() if proto['y'] == label)
                                for label in set(proto['y'] for proto in original_prototypes.values())}
        total_prototypes = sum(label_prototypes_count.values())

        # Calculate the target number of clusters for each label
        label_clusters = {label: max(1, int(count * target_prototypes / total_prototypes))
                        for label, count in label_prototypes_count.items()}

        # Apply K-Means clustering for each label
        for label, num_clusters in label_clusters.items():
            label_prototypes = np.array([proto['x'] for proto in original_prototypes.values() if proto['y'] == label])
            if label_prototypes.size == 0:
                continue

            kmeans = KMeans(
                n_clusters=min(num_clusters, len(label_prototypes)),
                init='k-means++',
                n_init='auto',
                random_state=42
            )
            labels = kmeans.fit_predict(label_prototypes)

            for cluster_id in range(kmeans.n_clusters):
                cluster_indices = np.where(labels == cluster_id)[0]
                cluster_protos = [label_prototypes[i] for i in cluster_indices]
                centroid = np.mean(cluster_protos, axis=0)
                sum_m = sum(original_prototypes[list(original_prototypes.keys())[i]]['m'] for i in cluster_indices)

                new_prototypes[next_prototype_id] = {
                    'x': centroid,
                    'y': label,
                    'm': sum_m,
                    'neighbors': []
                }
                next_prototype_id += 1

        # Update the prototype dictionary
        self.prototypes = new_prototypes
        # print(f"Updated prototypes to {len(new_prototypes)} clusters (goal was {target_prototypes}).")
    
        
    def rebuild_neighborhoods(self, num_neighbors=2):
        proto_ids = list(self._prototypes.keys())
        temp_edges = {}

        for pid1 in proto_ids:
            vector = self._prototypes[pid1]['x']
            nearest_generator = self.find_nearest(vector, num_neighbors)
            closest_neighbors = []

            for s, _, _ in nearest_generator:
                if s != pid1:
                    closest_neighbors.append(s)
                    if (pid1, s) not in temp_edges and (s, pid1) not in temp_edges:
                        self.create_edge(pid1, s)
                        temp_edges[(pid1, s)] = 0 

            self._prototypes[pid1]['neighbours'] = closest_neighbors

        self.edges = temp_edges
            

    @prototypes.setter
    def prototypes(self, prototypes):
        self._prototypes = prototypes

    @property
    def edges_whole(self):
        return self._edge_age

    @edges.setter
    def edges(self, edges):
        self._edge_age = edges