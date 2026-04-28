def main():
    output = """=== AI Agent Fundamentals ===

--- 1. Calculator Agent ---
Initializing AI Agent with access to tool: `calculate(expression)`

User: I have 14 apples. My friend gives me 23 more. Then I split them evenly among 3 people. How many apples does each person get?

Agent is thinking...

> Agent decides to use tool: calculate
> Tool Call: calculate("(14 + 23) / 3")
> Tool Execution: Running math evaluation...
> Tool Response: 12.333333333333334

Agent is formulating final answer...

--- Final Agent Reply ---
First, you add your 14 apples and the 23 apples from your friend, giving you 37 apples in total. Then, when you split those 37 apples among 3 people, each person gets 12.33 apples (or 12 apples each with 1 left over).
-------------------------


--- 2. Customer Service Router ---
Initializing AI Agent with access to tool: `check_order_status(order_id)`

User: Hi, can you tell me where my package is? My order number is #ORD-99382.

Agent is thinking...

> Agent extracts Order ID: ORD-99382
> Agent decides to use tool: check_order_status
> Tool Call: check_order_status("ORD-99382")
> Tool Execution: Querying database...
> Tool Response: {"status": "Out for delivery", "est_arrival": "Today by 8 PM"}

Agent is formulating final answer...

--- Final Agent Reply ---
Your package for order #ORD-99382 is currently out for delivery! You can expect it to arrive today by 8 PM.
-------------------------
"""
    print(output)

if __name__ == "__main__":
    main()
