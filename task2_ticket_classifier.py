import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# STEP 1: Create Sample Customer Support Dataset
print("--- STEP 1: Creating Dataset ---")
data = {
    'ticket_text': [
        "Payment failed and money was debited from my bank account.",
        "Unable to login, password reset email link is broken.",
        "The application keeps crashing every time I open the cart page.",
        "I need a refund for my recent monthly subscription bill.",
        "Account is locked after typing incorrect credentials."
    ],
    'category': ['Billing', 'Account', 'Technical', 'Billing', 'Account'],
    'priority': ['High', 'Medium', 'High', 'Medium', 'Medium']
}
df = pd.DataFrame(data)

# STEP 2: Text Preprocessing Function
print("\n--- STEP 2: Cleaning Text Data ---")
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove numbers and special characters
    return text.strip()

df['clean_text'] = df['ticket_text'].apply(clean_text)

# STEP 3: Convert Text to Numerical Vectors (TF-IDF)
print("\n--- STEP 3: Vectorizing Text using TF-IDF ---")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])

# STEP 4: Train Classifiers
print("\n--- STEP 4: Training Classification Models ---")
# Category Model
model_category = LogisticRegression()
model_category.fit(X, df['category'])

# Priority Model
model_priority = LogisticRegression()
model_priority.fit(X, df['priority'])

# STEP 5: Test on New Tickets
print("\n--- STEP 5: Testing Incoming Support Tickets ---")
new_tickets = [
    "Money was taken from my card but my order is not placed.",
    "App suddenly closes when I try to pay."
]

for ticket in new_tickets:
    cleaned = clean_text(ticket)
    vec = vectorizer.transform([cleaned])
    
    pred_cat = model_category.predict(vec)[0]
    pred_prio = model_priority.predict(vec)[0]
    
    print(f"\nIncoming Ticket: '{ticket}'")
    print(f" -> Predicted Category: {pred_cat}")
    print(f" -> Predicted Priority: {pred_prio}")
