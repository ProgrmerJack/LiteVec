# AI Agent Memory Example

Give AI agents long-term memory using LiteVec's MCP server.

## Overview

AI agents (ChatGPT, Claude, custom LLMs) are stateless by default — they forget everything between conversations. LiteVec's MCP server gives them persistent, searchable memory.

## Architecture

```
User → AI Agent ←→ MCP Protocol ←→ LiteVec MCP Server ←→ Database
                                         ↕
                                   Vector Search
```

## Setup

### 1. Start the MCP Server

```bash
litevec-mcp --db agent_memory.lv
```

### 2. Configure Your Agent

For Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "litevec-mcp",
      "args": ["--db", "agent_memory.lv"]
    }
  }
}
```

### 3. Agent Uses Memory Tools

The agent can now:

1. **Store memories**: Insert conversation snippets as vectors
2. **Recall**: Search for relevant past conversations
3. **Update**: Overwrite outdated information

## Example Conversation Flow

```
User: "My dog's name is Biscuit and she's a golden retriever."

Agent thinks: I should remember this.
Agent calls: insert(collection="user_facts", id="dog_name", 
             vector=embed("User has a dog named Biscuit, golden retriever"),
             metadata={"fact": "dog", "name": "Biscuit", "breed": "golden retriever"})

--- Later ---

User: "What breed is my dog?"

Agent thinks: Let me check my memory.
Agent calls: search(collection="user_facts", 
             query=embed("user's dog breed"), k=3)
Result: {"id": "dog_name", "metadata": {"breed": "golden retriever"}}

Agent: "Your dog Biscuit is a golden retriever!"
```

## Programmatic Agent Memory (Python)

```python
import litevec
from sentence_transformers import SentenceTransformer

class AgentMemory:
    def __init__(self, db_path="agent_memory.lv", dimension=384):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.db = litevec.Database(db_path)
        try:
            self.memories = self.db.get_collection("memories")
        except:
            self.memories = self.db.create_collection("memories", dimension=dimension)
    
    def remember(self, memory_id: str, text: str, metadata: dict = None):
        """Store a memory."""
        embedding = self.model.encode(text).tolist()
        self.memories.insert(memory_id, embedding, metadata or {"text": text})
    
    def recall(self, query: str, k: int = 5) -> list:
        """Search for relevant memories."""
        embedding = self.model.encode(query).tolist()
        return self.memories.search(embedding, k=k)
    
    def forget(self, memory_id: str):
        """Delete a memory."""
        self.memories.delete(memory_id)

# Usage
memory = AgentMemory()
memory.remember("fact_1", "User prefers dark mode", {"category": "preference"})
memory.remember("fact_2", "User is a Python developer", {"category": "profile"})

relevant = memory.recall("What programming language does the user know?")
# Returns fact_2
```

## Best Practices

1. **Organize by collection**: Use separate collections for different memory types (facts, preferences, conversation history)
2. **Include text in metadata**: Always store the original text in metadata for retrieval
3. **Use meaningful IDs**: `user_preference_theme` is better than `mem_42`
4. **Periodic cleanup**: Delete old or irrelevant memories to keep search quality high
5. **Embedding model consistency**: Always use the same model for storage and retrieval
