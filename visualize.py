import gymnasium as gym
import time
import numpy as np
import pygame  # מייבאים את פייגיים כדי לטפל באירועי החלון

def run_and_visualize_policy(agent, num_episodes=1):
    # יצירת הסביבה במצב תצוגה אנושי
    env = gym.make("Taxi-v4", is_rainy=False, render_mode="human")
    
    print(f"מציגים ריצה ויזואלית של המונית לאורך {num_episodes} פרקים...")
    
    for episode in range(num_episodes):
        state, info = env.reset(seed=100 + episode)
        done = False
        total_reward = 0
        steps = 0
        
        # איפוס זמני של אפסילון כדי להציג רק את מה שהיא למדה (Exploitation)
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0 
        
        # הגבלת צעדים ל-50 כדי למנוע לולאה אינסופית אם הסוכן מתבלבל
        while not done and steps < 50:
            # מניעת קיפאון של חלון ה-Pygame במערכת ההפעלה
            pygame.event.get()
            
            # בחירת פעולה
            action = agent.choose_action(state)
            
            # ביצוע הצעד
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # קצב צעדים נוח לצפייה
            time.sleep(0.3)
            
        agent.epsilon = original_epsilon
        print(f"פרק תצוגה {episode + 1} הסתיים! צעדים שבוצעו: {steps} | פרס מצטבר: {total_reward}")
        time.sleep(1)
        
    env.close()