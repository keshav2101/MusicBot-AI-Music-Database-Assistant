import sqlite3
import json
from datetime import datetime
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# ============================================
# PERSONA - The chatbot's personality
# ============================================
PERSONA = """
You are MusicBot, a friendly and intelligent music database assistant!
You love talking about music, artists, albums, and helping users explore the Chinook music database.
You can handle tough analytical questions and create beautiful visualizations to represent data graphically.
You are enthusiastic but clear in your responses.
You always try to remember what users have asked before and build on previous conversations.

IMPORTANT BEHAVIOR:
- When asked database-related questions (artists, albums, tracks, sales, customers), you should use your tools
- When asked general questions NOT related to the database (like "who is iron man", "what is the weather"),
  respond politely that you specialize in music database queries and gently ask them to ask music related questions
- DO NOT try to use database tools for general knowledge questions
- If you're unsure whether a question is database-related, ask for clarification
- Be conversational and helpful, don't just throw errors

For visualization requests, you should create charts and graphs.
For complex database questions, you should write and execute SQL queries.
"""

# ============================================
# MEMORY
# ============================================
class Memory:
    def __init__(self):
        self.conn = sqlite3.connect('chatbot_memory.db')
        self.setup_database()
   
    def setup_database(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_message TEXT,
                bot_response TEXT
            )
        ''')
        self.conn.commit()
   
    def save_conversation(self, user_msg, bot_msg):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (timestamp, user_message, bot_response)
            VALUES (?, ?, ?)
        ''', (datetime.now().isoformat(), user_msg, bot_msg))
        self.conn.commit()
   
    def get_recent_conversations(self, limit=5):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT user_message, bot_response FROM conversations
            ORDER BY id DESC LIMIT ?
        ''', (limit,))
        results = cursor.fetchall()
        return list(reversed(results))

# ============================================
# KNOWLEDGE
# ============================================
KNOWLEDGE_BASE = {
    "database_info": """
    The Chinook database contains information about:
    - Artists and their albums
    - Music tracks with duration and pricing
    - Customers and their purchases
    - Invoices and sales data
    - Playlists and genres
    """,
    "available_queries": [
        "Search for artists by name",
        "Find albums by an artist",
        "Get track information",
        "View customer purchase history",
        "Check playlist contents",
        "Execute complex SQL queries for tough questions",
        "Create bar charts to visualize data",
        "Create pie charts for category distributions",
        "Get comprehensive sales statistics"
    ]
}

# ============================================
# TOOLS
# ============================================
@tool
def search_artists(artist_name: str) -> str: #ye search kare ga artist ko
    """Search for artists by name in the database"""
    conn = sqlite3.connect('Chinook.db') #database connect kare ga
    cursor = conn.cursor()
    cursor.execute("SELECT ArtistId, Name FROM Artist WHERE Name LIKE ? LIMIT 10",
                   (f'%{artist_name}%',))
    results = cursor.fetchall()
    conn.close()
   
    if results:
        return f"Found {len(results)} artists: " + ", ".join([f"{r[1]} (ID: {r[0]})" for r in results])
    else:
        return "No artists found with that name"

@tool
def get_albums_by_artist(artist_id: int) -> str: #ye artist ko albums de ga
    """Get all albums by an artist using their ArtistId"""
    conn = sqlite3.connect('Chinook.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT Album.Title, Artist.Name
        FROM Album
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Artist.ArtistId = ?
    """, (artist_id,))
    results = cursor.fetchall()
    conn.close()
   
    if results:
        artist_name = results[0][1]
        albums = [r[0] for r in results]
        return f"{artist_name} has {len(albums)} albums: " + ", ".join(albums)
    else:
        return "No albums found for this artist"

@tool
def search_tracks(track_name: str) -> str: #ye tracks ko search kare ga
    """Search for music tracks by name"""
    conn = sqlite3.connect('Chinook.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT Track.Name, Artist.Name, Album.Title
        FROM Track
        JOIN Album ON Track.AlbumId = Album.AlbumId
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Track.Name LIKE ? LIMIT 10
    """, (f'%{track_name}%',))
    results = cursor.fetchall()
    conn.close()
   
    if results:
        tracks_info = [f"'{r[0]}' by {r[1]} from album '{r[2]}'" for r in results]
        return f"Found {len(results)} tracks:\n" + "\n".join(tracks_info)
    else:
        return "No tracks found with that name"

@tool
def get_customer_info(customer_id: int) -> str: #ye customer ko info de ga
    """Get customer information by their CustomerId"""
    conn = sqlite3.connect('Chinook.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT FirstName, LastName, Country, Email
        FROM Customer
        WHERE CustomerId = ?
    """, (customer_id,))
    result = cursor.fetchone()
    conn.close()
   
    if result:
        return f"Customer: {result[0]} {result[1]}, Country: {result[2]}, Email: {result[3]}"
    else:
        return "Customer not found"

@tool
def get_top_genres() -> str: #ye top genres ko de ga
    """Get the most popular music genres based on track count"""
    conn = sqlite3.connect('Chinook.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT Genre.Name, COUNT(*) as TrackCount
        FROM Genre
        JOIN Track ON Genre.GenreId = Track.GenreId
        GROUP BY Genre.Name
        ORDER BY TrackCount DESC
        LIMIT 5
    """)
    results = cursor.fetchall()
    conn.close()
   
    genres_info = [f"{r[0]}: {r[1]} tracks" for r in results]
    return "Top 5 genres:\n" + "\n".join(genres_info)

@tool
def execute_sql_query(sql_query: str) -> str: #ye custom sql query ko execute kare ga aoor dekhe ga ke koi cheeze delete na ho jaye
    """Execute a custom SQL query on the Chinook database for complex questions.
    Use this for tough questions that require complex joins, aggregations, or calculations.
    Example: 'SELECT Artist.Name, COUNT(*) as AlbumCount FROM Artist JOIN Album ON Artist.ArtistId = Album.ArtistId GROUP BY Artist.Name ORDER BY AlbumCount DESC LIMIT 10'
    """
    conn = sqlite3.connect('Chinook.db')
    cursor = conn.cursor()
   
    try:
        # keval select query allow kare ga
        if not sql_query.strip().upper().startswith('SELECT'):
            return "Error: Only SELECT queries are allowed for safety reasons"
       
        cursor.execute(sql_query)
        results = cursor.fetchall()
       
        if not results:
            return "Query executed successfully but returned no results"
       
        # results ko format kare ga
        if len(results) > 20:
            output = f"Found {len(results)} results. Showing first 20:\n"
            results = results[:20]
        else:
            output = f"Found {len(results)} results:\n"
       
        for row in results:
            output += str(row) + "\n"
       
        conn.close()
        return output
   
    except Exception as e:
        conn.close()
        return f"SQL Error: {str(e)}"

@tool
def create_bar_chart(sql_query: str, chart_title: str, x_label: str, y_label: str) -> str:
    """Create a bar chart from SQL query results and save it as an image.
    The SQL query should return exactly 2 columns: label (x-axis) and value (y-axis).
    Example: 'SELECT Genre.Name, COUNT(*) FROM Genre JOIN Track ON Genre.GenreId = Track.GenreId GROUP BY Genre.Name LIMIT 10'
    """
    conn = sqlite3.connect('Chinook.db')
    cursor = conn.cursor()
   
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
       
        if not results:
            return "No data to visualize"
       
        # Extract labels and values
        labels = [str(row[0]) for row in results]
        values = [float(row[1]) for row in results]
       
        # Create bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(labels, values, color='skyblue', edgecolor='navy')
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(chart_title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
       
        # Save chart
        filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
       
        return f"Bar chart created successfully! Saved as: {filename}\nShowing {len(results)} data points."
   
    except Exception as e:
        conn.close()
        return f"Chart creation error: {str(e)}"

@tool
def create_pie_chart(sql_query: str, chart_title: str) -> str:
    """Create a pie chart from SQL query results and save it as an image.
    The SQL query should return exactly 2 columns: category name and value.
    Example: 'SELECT Genre.Name, COUNT(*) FROM Genre JOIN Track ON Genre.GenreId = Track.GenreId GROUP BY Genre.Name LIMIT 8'
    """
    conn = sqlite3.connect('Chinook.db')
    cursor = conn.cursor()
   
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
       
        if not results:
            return "No data to visualize"
       
        # Extract labels and values
        labels = [str(row[0]) for row in results]
        values = [float(row[1]) for row in results]
       
        # Create pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
        plt.title(chart_title, fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
       
        # Save chart
        filename = f"pie_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
       
        return f"Pie chart created successfully! Saved as: {filename}\nShowing {len(results)} categories."
   
    except Exception as e:
        conn.close()
        return f"Chart creation error: {str(e)}"

@tool
def get_sales_statistics() -> str:
    """Get comprehensive sales statistics from the database - useful for tough analytical questions"""
    conn = sqlite3.connect('Chinook.db')
    cursor = conn.cursor()
   
    # Total sales
    cursor.execute("SELECT COUNT(*), SUM(Total) FROM Invoice")
    invoice_data = cursor.fetchone()
   
    # Top customers
    cursor.execute("""
        SELECT Customer.FirstName || ' ' || Customer.LastName, SUM(Invoice.Total) as TotalSpent
        FROM Customer
        JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId
        GROUP BY Customer.CustomerId
        ORDER BY TotalSpent DESC LIMIT 5
    """)
    top_customers = cursor.fetchall()
   
    # Sales by country
    cursor.execute("""
        SELECT BillingCountry, COUNT(*) as InvoiceCount, SUM(Total) as TotalSales
        FROM Invoice
        GROUP BY BillingCountry
        ORDER BY TotalSales DESC LIMIT 5
    """)
    top_countries = cursor.fetchall()
   
    conn.close()
   
    output = f"Sales Statistics:\n\n"
    output += f"Total Invoices: {invoice_data[0]}\n"
    output += f"Total Revenue: ${invoice_data[1]:.2f}\n\n"
    output += "Top 5 Customers:\n"
    for i, (name, total) in enumerate(top_customers, 1):
        output += f"{i}. {name}: ${total:.2f}\n"
    output += "\nTop 5 Countries by Sales:\n"
    for i, (country, count, total) in enumerate(top_countries, 1):
        output += f"{i}. {country}: {count} invoices, ${total:.2f}\n"
   
    return output

@tool
def visualize_artist_tracks(limit: int = 1000) -> str:
    """Create a bar chart showing artists with the most tracks/songs.
    This properly handles the Track-Album-Artist relationship.
    Use this when asked about artist track counts, song counts, or artist productivity.
   
    Parameters:
    - limit: Number of artists to show. Use 1000 for "all artists" or when user says "all", "every", or "maximum".
             Use 10-20 for "top N" requests. Default is 1000 to show all artists.
    """
    conn = sqlite3.connect('Chinook.db')
    cursor = conn.cursor()
   
    try:
        # pehle artist ke tracks ke count ko get kare ga
        cursor.execute("""
            SELECT COUNT(DISTINCT Artist.ArtistId)
            FROM Artist
            JOIN Album ON Artist.ArtistId = Album.ArtistId
            JOIN Track ON Album.AlbumId = Track.AlbumId
        """)
        total_artists = cursor.fetchone()[0]
       
        # limit ko adjust kare ga- default limit aoor orginal limit ko dekhe ga
        actual_limit = min(limit, total_artists)
       
        # Proper join: Track - Album - Artist
        query = """
            SELECT Artist.Name, COUNT(Track.TrackId) as TrackCount
            FROM Artist
            JOIN Album ON Artist.ArtistId = Album.ArtistId
            JOIN Track ON Album.AlbumId = Track.AlbumId
            GROUP BY Artist.Name
            ORDER BY TrackCount DESC
            LIMIT ?
        """
        cursor.execute(query, (actual_limit,))
        results = cursor.fetchall()
        conn.close()
       
        if not results:
            return "No data found"
       
        # Extract data
        artists = [str(row[0]) for row in results]
        track_counts = [int(row[1]) for row in results]
       
        # Adjust figure size based on number of artists
        if len(results) > 50:
            fig_width = 20
            fig_height = 12
            fontsize = 7
        elif len(results) > 20:
            fig_width = 16
            fig_height = 10
            fontsize = 8
        else:
            fig_width = 14
            fig_height = 7
            fontsize = 10
       
        # Create bar chart
        plt.figure(figsize=(fig_width, fig_height))
        plt.bar(artists, track_counts, color='steelblue', edgecolor='darkblue', linewidth=1.2)
        plt.xlabel('Artist', fontsize=13, fontweight='bold')
        plt.ylabel('Number of Tracks', fontsize=13, fontweight='bold')
       
        title = f'All {len(results)} Artists by Track Count' if len(results) == total_artists else f'Top {len(results)} Artists by Track Count'
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xticks(rotation=90, ha='right', fontsize=fontsize)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
       
        # Save chart
        filename = f"artist_tracks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
       
        # Create comprehensive summary
        summary = f"Bar chart created: {filename}\n\n"
        summary += f"Showing {len(results)} artists out of {total_artists} total artists with tracks.\n\n"
       
        # Show top 10 in summary
        summary += f"Top 10 artists by track count:\n"
        for i, (artist, count) in enumerate(results[:10], 1):
            summary += f"{i:2}. {artist}: {count} tracks\n"
       
        if len(results) > 10:
            summary += f"\n... and {len(results) - 10} more artists in the chart!"
       
        return summary
   
    except Exception as e:
        conn.close()
        return f"Error creating visualization: {str(e)}"

# STATE - Conversation state mannage kare ga
class State(TypedDict):
    messages: list

# LangGraph agent
class MusicBotAgent:
    def __init__(self):
        self.memory = Memory()
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.7
        )
       
        # yaha tools provider kar rahe hai
        self.tools = [
            search_artists,
            get_albums_by_artist,
            search_tracks,
            get_customer_info,
            get_top_genres,
            execute_sql_query,
            create_bar_chart,
            create_pie_chart,
            get_sales_statistics,
            visualize_artist_tracks
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
       
        # Build the graph
        self.graph = self.build_graph()
   
    def build_graph(self):
        workflow = StateGraph(State)
       
        # Add nodes
        workflow.add_node("agent", self.agent_node)  #ye agent ko add kare ge
        workflow.add_node("tools", ToolNode(self.tools))  #ye tools ko add kare ge
       
        # Set entry point
        workflow.set_entry_point("agent")  #agent se start hoga
       
        # Add edges
        workflow.add_conditional_edges(  #yaha decision point hai
            "agent",
            self.should_continue,  #ye check kare ga ki tools chahiye ya nahi
            {
                "continue": "tools",  #agar tools chahiye to tools node me jao
                "end": END  #agar nahi chahiye to khatam karo
            }
        )
        workflow.add_edge("tools", "agent")  #tools se wapas agent me aao
       
        return workflow.compile()  #graph ko compile karo
   
    def agent_node(self, state: State):  #yaha LLM sochta hai
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)  #llm ko messages bhejo
        return {"messages": messages + [response]}  #response ko messages me add karo
   
    def should_continue(self, state: State):  #ye decide kare ga aage kya karna hai
        last_message = state["messages"][-1]  #last message ko dekho
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:  #agar tools chahiye
            return "continue"  #to tools node me jao
        return "end"  #warna khatam karo
   
    def get_context(self):  #purani conversations ko load karo
        # Get recent conversation history
        recent = self.memory.get_recent_conversations(3)  #last 3 conversations nikalo
        if recent:
            context = "Recent conversation history:\n"
            for user_msg, bot_msg in recent:
                context += f"User: {user_msg}\nBot: {bot_msg}\n"
            return context
        return "No previous conversation history."
   
    def chat(self, user_input):
        # Build context from memory
        context = self.get_context()
       
        # Create system message with persona and knowledge
        system_msg = SystemMessage(content=f"""
{PERSONA}

Database Information:
{KNOWLEDGE_BASE['database_info']}

You can help users with:
{', '.join(KNOWLEDGE_BASE['available_queries'])}

IMPORTANT DATABASE RELATIONSHIPS:
- Tracks are linked to Artists through Albums (Track - Album - Artist)
- When asked about "tracks by artist" or "songs by artist", use the visualize_artist_tracks tool

VISUALIZATION GUIDELINES:
- If user asks for "all artists", "every artist", "all of them", or "maximum": use limit=1000 (shows all)
- If user asks for "top 10" or "top 20": use that specific limit
- Default behavior: show ALL artists unless user specifies a number
- The tool automatically adjusts chart size based on number of artists
- When calling visualize_artist_tracks, 'limit' must be an INTEGER (e.g., 10, 50). Do not pass dictionaries.

HANDLING NON-DATABASE QUESTIONS:
Examples of NON-database questions (respond conversationally, DON'T use tools):
- "Who is Iron Man?" - "I'm MusicBot, specialized in music database queries. Iron Man is a Marvel character, but I can help you find music by bands like Black Sabbath who made the Iron Man song! Would you like to explore that?"
- "What's the weather?" - "I don't have weather data, but I can help you find music! What would you like to know about artists, albums, or tracks?"
- "Tell me a joke" - "I'm better with music than jokes! But I can tell you that Iron Maiden has 213 tracks in the database - that's no joke! Want to see more?"

Examples of DATABASE questions (use tools):
- "Who is the artist with most songs?" - Use visualize_artist_tracks tool
- "Show me sales statistics" - Use get_sales_statistics tool
- "Find tracks by AC/DC" - Use search_artists then search_tracks

{context}
""")
       
        # Create user message
        user_msg = HumanMessage(content=user_input)
       
        # Run the agent
        initial_state = {
            "messages": [system_msg, user_msg],
            "context": context
        }
       
        result = self.graph.invoke(initial_state)
       
        # Get bot response
        bot_response = result["messages"][-1].content
       
        # Save to memory
        self.memory.save_conversation(user_input, bot_response)
       
        return bot_response
#ye raha main function
def main():
    print("=" * 50)
    print("Database Assistant Bot")
    print("=" * 50)
   
    # Create the agent
    agent = MusicBotAgent()
   
    print("\n MusicBot is ready! Type 'quit' to exit.\n")
   
    # Chat loop
    while True:
        user_input = input("You: ").strip()
       
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("MusicBot: Goodbye! Thanks for chatting!")
            break
       
        if not user_input:
            continue
       
        try:
            response = agent.chat(user_input)
            print(f"MusicBot: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
