from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random
import numpy as np

texts = ["I love this!", "This is terrible.", "I feel okay about it.", "Amazing experience!", "Not good at all.", "Neutral opinion."]
labels = [1, -1, 0, 1, -1, 0] 

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

def sentiments(user_message):
    X_user = vectorizer.transform([user_message])
    polarity = model.predict(X_user)[0] 
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

# Predefined responses
responses = {
    "positive": ["I'm so glad to hear that! 😊", "That's amazing! Tell me more!", "Wonderful! It’s great to see positive vibes!"],
    "negative": ["I'm sorry to hear that. Do you want to talk about it?", "That sounds tough. I'm here to listen.", "Oh no, I'm here to help if you need me!"],
    "neutral": ["Got it. Let me know how I can assist you further.", "Okay! Anything else you'd like to share?", "Thanks for sharing. What’s next on your mind?"]
}
def response(sentiment):
    return random.choice(responses[sentiment])

def chatbot():
    print("Hi, I'm your friendly chatbot! 😊 Let's chat. Type 'exit' to end the conversation.")
    
    while True:
        user = input("\nYou: ")
        if user.lower() == "exit":
            print("Chatbot: It was great talking to you! Take care. 😊")
            break
        
        sentiment = sentiments(user)
        respond = response(sentiment)
        print(f"Chatbot: {respond}")
if __name__ == "__main__":
    chatbot()