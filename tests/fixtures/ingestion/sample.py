"""Sample module used to exercise the Python extractor."""
 
from __future__ import annotations
 
import os
 
VERSION = "1.0.0"
 
 
def greet(name: str) -> str:
    """Return a friendly greeting for *name*."""
    return f"Hello, {name}!"
 
 
class Greeter:
    """Greets people, keeping a running count of greetings."""
 
    def __init__(self) -> None:
        self.count = 0
 
    def greet(self, name: str) -> str:
        self.count += 1
        return greet(name)
 
 
print(os.getcwd())
 
