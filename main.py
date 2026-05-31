import gymnasium as gym
import numpy as np
from agent import GeneralizedSweepingSARSAAgent
from visualize import run_and_visualize_policy

def get_designed_reward(env_reward, terminated, info, reward_type="base"):
    """
    פונקציה המחשבת ומחליפה את הפרס המקורי של הסביבה לפי הבחירה שלנו
    """
    if reward_type == "base":
        # 1. הפרס המקורי של סביבת ה-Taxi
        return env_reward
        
    elif reward_type == "sparse":
        # 2. פרס דליל (Sparse): 0.5 על איסוף מוצלח, 0.5 על הורדה מוצלח, 0 בשאר הזמן
        # אנחנו מזהים הצלחות לפי הציון המקורי (20+ זו הורדה, ואיסוף מוצלח לא נותן עונש של 10-)
        if env_reward == 20: 
            return 0.5  # הורדה מוצלח
        
        # ב-Taxi-v4, איסוף מוצלח לא מחזיר פרס מיוחד, אבל אפשר לזהות אותו 
        # אם ביצענו פעולת איסוף (פעולה 4) ולא קיבלנו עונש לא חוקי של 10-
        # לצורך הפשטות של האופציה הראשונה שהמרצה נתן: +1 על dropoff מוצלח ו-0 בשאר הזמן
        if env_reward > 0:
            return 1.0 if env_reward == 20 else 0.0
        return 0.0
        
    elif reward_type == "reward_3":
        # 3. עיצוב אישי שלך (נממש בהמשך - למשל מבוסס מרחק)
        return env_reward
        
    elif reward_type == "reward_4":
        # 4. עיצוב אישי נוסף שלך (נממש בהמשך)
        return env_reward
        
    return env_reward

def train_agent(reward_type="base", num_episodes=2000):
    env = gym.make("Taxi-v4", is_rainy=False) 
    
    # אתחול הסוכן עם שני המנועים עובדים (Planning + Kernel)
    agent = GeneralizedSweepingSARSAAgent(
        alpha=0.1, 
        gamma=0.95, # הגדלנו קצת את גמא כדי שיעריך יותר את העתיד
        epsilon=0.1, 
        use_planning=True, 
        planning_steps=10,
        use_kernel=True,
        beta=4.0, 
        c=0.5
    )
    
    print(f"--- מתחילים אימון של {num_episodes} פרקים עם סוג פרס: {reward_type.upper()} ---")
    
    for episode in range(num_episodes):
        state, info = env.reset(seed=episode) 
        action = agent.choose_action(state)
        done = False
        total_env_reward = 0
        
        while not done:
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # החלפת הפרס המקורי בפרס המעוצב שלנו!
            custom_reward = get_designed_reward(env_reward, terminated, info, reward_type=reward_type)
            
            next_action = agent.choose_action(next_state)
            
            # עדכון הסוכן מתבצע עם ה-custom_reward
            agent.update(state, action, custom_reward, next_state, next_action, done)
            
            state = next_state
            action = next_action
            total_env_reward += env_reward
            
        # הדפסה פעם ב-200 פרקים
        if (episode + 1) % 200 == 0:
            print(f"פרק {episode + 1}/{num_episodes} | פרס מקורי מצטבר: {total_env_reward}")
            
    env.close()
    print("האימון הסתיים!")
    
    # הצגת הריצה הוויזואלית
    run_and_visualize_policy(agent, num_episodes=1)

if __name__ == "__main__":
    # נריץ כעת אימון ארוך ואיכותי יותר של 2000 פרקים על הפרס המקורי (Base)
    train_agent(reward_type="base", num_episodes=2000)