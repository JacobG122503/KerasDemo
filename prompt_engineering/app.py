def main():
    output = """=== Prompt Engineering Concepts ===

--- 1. Few-Shot Prompting ---
Initializing AI Model with System Prompt...
[System]: You are a helpful AI assistant that classifies reviews into Positive, Negative, or Neutral.

--- User Prompt ---
Please classify the sentiment of the following reviews:

Review: I loved this product, it works perfectly!
Sentiment: Positive

Review: This is the worst thing I have ever bought.
Sentiment: Negative

Review: It arrived on time.
Sentiment: Neutral

Review: The battery life is okay, but the screen is amazing!
Sentiment:
-------------------

Generating response...

--- AI Output ---
Positive
-----------------


--- 2. Prompt Chaining ---
--- Step 1: Requesting Outline ---
User: Give me a 3-bullet point outline on the benefits of exercise.
AI generating outline...

AI Output:
1. Improves cardiovascular health and stamina.
2. Boosts mental health and reduces stress.
3. Helps maintain a healthy weight and builds muscle.

--- Step 2: Expanding Outline into Summary ---
User: Based on the following outline, write a short, 2-sentence motivational summary:
1. Improves cardiovascular health and stamina.
2. Boosts mental health and reduces stress.
3. Helps maintain a healthy weight and builds muscle.

AI generating summary...

AI Output:
Exercise is a powerful way to enhance both your physical stamina and mental well-being while keeping your body strong. By staying active, you can build a healthier, happier version of yourself every single day!
"""
    print(output)

if __name__ == "__main__":
    main()
