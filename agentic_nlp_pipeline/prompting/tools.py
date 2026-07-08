"""Tool template:
{
    "type": "function",
    "function": {
        "name": "my_tool",             # must match the function name
        "description": "...",          # the model reads this to decide when to use the tool
        "parameters": {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "..."  # the model reads this to know what to pass
                }
            },
            "required": ["param"]
        }
    }
}
"""
