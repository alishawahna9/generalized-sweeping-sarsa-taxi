import heapq

import numpy as np


class GeneralizedSweepingSARSAAgent:
    """Tabular SARSA agent with optional prioritized sweeping and kernel updates."""

    def __init__(
        self,
        num_states=500,
        num_actions=6,
        alpha=0.5,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        use_planning=False,
        use_kernel=False,
        planning_steps=5,
        theta=1e-4,
        beta=1.0,
        c=1.5,
        max_kernel_neighbors=10,
        max_queue_size=5000,
        kernel_in_planning=False,
        kernel_scale=1.0,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.use_planning = use_planning
        self.use_kernel = use_kernel
        self.planning_steps = planning_steps
        self.theta = theta

        self.beta = beta
        self.c = c
        self.max_kernel_neighbors = max_kernel_neighbors
        self.max_queue_size = max_queue_size
        self.kernel_in_planning = kernel_in_planning
        self.kernel_scale = kernel_scale

        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

        self.model = {}
        self.predecessors = {}
        self.pq = []
        self.entry_count = 0

        self.kernel_neighbors = self._build_kernel_neighbors() if self.use_kernel else None

    def decode_state(self, state):
        """Decode Taxi-v4's integer state into row, col, passenger location, destination."""
        state = int(state)
        dest_idx = state % 4
        state //= 4
        pass_loc = state % 5
        state //= 5
        taxi_col = state % 5
        taxi_row = state // 5
        return taxi_row, taxi_col, pass_loc, dest_idx

    def encode_state(self, taxi_row, taxi_col, pass_loc, dest_idx):
        """Encode row, col, passenger location, destination into Taxi-v4's state id."""
        return ((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest_idx

    def choose_action(self, state):
        """Epsilon-greedy action selection with random tie-breaking."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.num_actions))

        action_values = self.q_table[int(state)]
        best_value = np.max(action_values)
        best_actions = np.flatnonzero(np.isclose(action_values, best_value))
        return int(np.random.choice(best_actions))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self, s, a, r, s_next, a_next, done):
        """Run one real SARSA update, then optional prioritized sweeping planning."""
        td_error = self._sarsa_update(s, a, r, s_next, a_next, done)

        if self.use_planning or self.use_kernel:
            self._store_transition(s, a, r, s_next, done)

        if self.use_kernel and self._should_generalize(s, a, s_next):
            self._kernel_update(s, a)

        if self.use_planning:
            self._push_priority(abs(td_error), s, a)
            self._run_planning()

        return abs(float(td_error))

    def _sarsa_update(self, s, a, r, s_next, a_next, done):
        q_current = self.q_table[s, a]
        q_next = 0.0 if done else self.q_table[s_next, a_next]
        td_error = r + self.gamma * q_next - q_current
        self.q_table[s, a] += self.alpha * td_error
        return td_error

    def _store_transition(self, s, a, r, s_next, done):
        self.model[(s, a)] = (r, s_next, done)
        self.predecessors.setdefault(s_next, set()).add((s, a))

    def _push_priority(self, priority, s, a):
        if priority <= self.theta or len(self.pq) >= self.max_queue_size:
            return

        heapq.heappush(self.pq, (-priority, self.entry_count, s, a))
        self.entry_count += 1

    def _run_planning(self):
        for _ in range(self.planning_steps):
            if not self.pq:
                break

            _, _, plan_s, plan_a = heapq.heappop(self.pq)
            transition = self.model.get((plan_s, plan_a))
            if transition is None:
                continue

            plan_r, plan_s_next, plan_done = transition
            plan_a_next = self.choose_action(plan_s_next)
            plan_td_error = self._sarsa_update(
                plan_s,
                plan_a,
                plan_r,
                plan_s_next,
                plan_a_next,
                plan_done,
            )

            if (
                self.use_kernel
                and self.kernel_in_planning
                and self._should_generalize(plan_s, plan_a, plan_s_next)
            ):
                self._kernel_update(plan_s, plan_a)

            pred_action_next = self.choose_action(plan_s)
            q_next_for_preds = self.q_table[plan_s, pred_action_next]

            for pred_s, pred_a in self.predecessors.get(plan_s, ()):
                pred_transition = self.model.get((pred_s, pred_a))
                if pred_transition is None:
                    continue

                pred_r, _, pred_done = pred_transition
                pred_q_current = self.q_table[pred_s, pred_a]
                pred_q_next = 0.0 if pred_done else q_next_for_preds
                pred_td_error = pred_r + self.gamma * pred_q_next - pred_q_current
                self._push_priority(abs(pred_td_error), pred_s, pred_a)

    def _build_kernel_neighbors(self):
        neighbors = []

        for state in range(self.num_states):
            taxi_row, taxi_col, pass_loc, dest_idx = self.decode_state(state)
            candidates = []

            for row in range(5):
                for col in range(5):
                    if row == taxi_row and col == taxi_col:
                        continue

                    distance = abs(taxi_row - row) + abs(taxi_col - col)
                    neighbor_state = self.encode_state(row, col, pass_loc, dest_idx)
                    weight = 1.0 / (1.0 + np.exp(self.beta * (distance - self.c)))
                    candidates.append((distance, neighbor_state, weight))

            candidates.sort(key=lambda item: (item[0], item[1]))
            if self.max_kernel_neighbors and self.max_kernel_neighbors > 0:
                candidates = candidates[: self.max_kernel_neighbors]

            state_ids = np.array([item[1] for item in candidates], dtype=np.int32)
            weights = np.array([item[2] for item in candidates], dtype=np.float64)
            neighbors.append((state_ids, weights))

        return neighbors

    def _kernel_update(self, s, a):
        state_ids, weights = self.kernel_neighbors[s]
        if state_ids.size == 0:
            return

        for neighbor_state, weight in zip(state_ids, weights):
            transition = self.model.get((int(neighbor_state), a))
            if transition is None:
                continue

            neighbor_r, neighbor_next, neighbor_done = transition
            if not self._should_generalize(int(neighbor_state), a, neighbor_next):
                continue

            neighbor_next_action = self.choose_action(neighbor_next)
            neighbor_q_next = (
                0.0
                if neighbor_done
                else self.q_table[neighbor_next, neighbor_next_action]
            )
            neighbor_td_error = (
                neighbor_r
                + self.gamma * neighbor_q_next
                - self.q_table[int(neighbor_state), a]
            )
            self.q_table[int(neighbor_state), a] += (
                self.alpha * self.kernel_scale * weight * neighbor_td_error
            )

    def _should_generalize(self, s, a, s_next):
        if a not in (0, 1, 2, 3):
            return False

        if s == s_next:
            return False

        return True
