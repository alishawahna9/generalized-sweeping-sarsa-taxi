import numpy as np
import heapq

class GeneralizedSweepingSARSAAgent:
    def __init__(self, num_states=500, num_actions=6, alpha=0.1, gamma=0.99, epsilon=0.1,
                 use_planning=False, use_kernel=False, planning_steps=10, theta=1e-4,
                 beta=4.0, c=0.5):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # דגלי אבלציה ותכנון
        self.use_planning = use_planning
        self.use_kernel = use_kernel
        self.planning_steps = planning_steps
        self.theta = theta
        
        # היפר-פרמטרים לפונקציית הגרעין
        self.beta = beta
        self.c = c
        
        # טבלת Q
        self.q_table = np.zeros((self.num_states, self.num_actions))
        
        # מודל וניהול תכנון
        self.model = {}
        self.predecessors = {}
        self.priority_queue = []

    def decode_state(self, state):
        """
        פענוח מצב גולמי (0-499) לפרמטרים הפיזיים של הסביבה:
        תשואה: (taxi_row, taxi_col, passenger_location, destination)
        """
        # גריד המונית הוא 5x5, ויש 5 מצבי נוסע ו-4 מצבי יעד: 5*5*5*4 = 500
        out = []
        out.append(state % 4)
        state = state // 4
        out.append(state % 5)
        state = state // 5
        out.append(state % 5)
        state = state // 5
        out.append(state)
        assert 0 <= state < 5
        # מחזיר: שורה (0-4), עמודה (0-4), מיקום נוסע (0-4), יעד (0-3)
        return int(out[3]), int(out[2]), int(out[1]), int(out[0])

    def _compute_kernel(self, s1, s2):
        """חישוב פונקציית הגרעין המתמטית בין שני מצבים"""
        r1, c1, p1, d1 = self.decode_state(s1)
        r2, c2, p2, d2 = self.decode_state(s2)
        
        # דרישה קריטית של המרצה: שני מצבים דומים אך ורק אם
        # מיקום הנוסע והיעד שלהם זהים לחלוטין!
        if p1 != p2 or d1 != d2:
            return 0.0
            
        # חישוב מרחק מנהטן על הגריד
        manhattan_dist = abs(r1 - r2) + abs(c1 - c2)
        
        # נוסחת הגרעין המוצעת על ידי המרצה (וריאציית סיגמואיד)
        kernel_val = 1.0 / (1.0 + np.exp(self.beta * (manhattan_dist - self.c)))
        return kernel_val

    def choose_action(self, state):
        """בחירת פעולה באסטרתגיית אפסילון-גרידי"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            max_value = np.max(self.q_table[state])
            actions_with_max_value = np.where(self.q_table[state] == max_value)[0]
            return np.random.choice(actions_with_max_value)

    def update(self, state, action, reward, next_state, next_action, done):
        """חוק העדכון המשולב עם חלחול ידע (Kernel Generalization) ותכנון"""
        q_current = self.q_table[state, action]
        q_next = 0 if done else self.q_table[next_state, next_action]
        td_error = reward + self.gamma * q_next - q_current
        
        # 1. עדכון המצב הנוכחי שחווינו בפועל
        self.q_table[state, action] += self.alpha * td_error
        
        # 2. מנגנון ההכללה (Kernel) - חלחול ידע למצבים דומים מסביב
        if self.use_kernel:
            # אופטימיזציה: רצים רק על מצבים שחולקים את אותו סטטוס נוסע ויעד
            r_curr, c_curr, p_curr, d_curr = self.decode_state(state)
            
            # לולאה חכמה שעוברת רק על 25 מצבי המיקום האפשריים של המונית עבור אותו תת-משימה
            for s_hat in range(self.num_states):
                if s_hat == state:
                    continue
                r_h, c_h, p_h, d_h = self.decode_state(s_hat)
                if p_h == p_curr and d_h == d_curr:
                    k_val = self._compute_kernel(state, s_hat)
                    if k_val > 1e-4:
                        # עדכון המצב הדומה לפי עוצמת הגרעין
                        self.q_table[s_hat, action] += self.alpha * k_val * td_error
        
        # 3. מנגנון תכנון (Prioritized Sweeping)
        if self.use_planning:
            self.model[(state, action)] = (reward, next_state, done)
            if next_state not in self.predecessors:
                self.predecessors[next_state] = set()
            self.predecessors[next_state].add((state, action))
            
            priority = abs(td_error)
            if priority > self.theta:
                heapq.heappush(self.priority_queue, (-priority, state, action))
                
            self._run_planning()

        return td_error

    def _run_planning(self):
        """לולאת תכנון מבוססת תור עדיפויות בסגנון SARSA"""
        steps = 0
        while self.priority_queue and steps < self.planning_steps:
            _, s, a = heapq.heappop(self.priority_queue)
            r, s_next, d = self.model[(s, a)]
            
            a_next = self.choose_action(s_next)
            q_curr_plan = self.q_table[s, a]
            q_next_plan = 0 if d else self.q_table[s_next, a_next]
            td_error_plan = r + self.gamma * q_next_plan - q_curr_plan
            
            self.q_table[s, a] += self.alpha * td_error_plan
            
            if s in self.predecessors:
                for (s_pred, a_pred) in self.predecessors[s]:
                    r_pred, _, d_pred = self.model[(s_pred, a_pred)]
                    a_next_pred = self.choose_action(s)
                    q_curr_pred = self.q_table[s_pred, a_pred]
                    q_next_pred = 0 if d_pred else self.q_table[s, a_next_pred]
                    
                    td_error_pred = r_pred + self.gamma * q_next_pred - q_curr_pred
                    priority_pred = abs(td_error_pred)
                    
                    if priority_pred > self.theta:
                        heapq.heappush(self.priority_queue, (-priority_pred, s_pred, a_pred))
            steps += 1