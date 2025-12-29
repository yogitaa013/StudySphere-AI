from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# =========================================
# LOAD AI MODEL (FLAN-T5 LARGE)
# =========================================
# device=-1 uses CPU
chatbot = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

# ===========================
# DETAILED PLACEMENT KNOWLEDGE BASE
# ===========================
knowledge_base = {
    "DBMS": {
        "definition": "Database Management System (DBMS) is software that allows users to create, maintain, and interact with databases efficiently. It ensures data integrity, security, and supports multiple users simultaneously.",
        "key_concepts": {
            "ER Model": "Graphical representation of entities, attributes, and relationships. Example: Students and Courses tables connected via Enrollment relationship.",
            "Normalization": "Organizing database tables to reduce redundancy and improve consistency. Includes 1NF, 2NF, 3NF, BCNF.",
            "SQL Queries": "SQL (Structured Query Language) is used to query, insert, update, or delete data. Example: SELECT name FROM Students WHERE grade='A';",
            "Joins": "Combine data from multiple tables. Types: INNER, LEFT, RIGHT, FULL OUTER JOIN.",
            "Transactions": "Sequence of operations treated as a single unit, ensuring ACID properties (Atomicity, Consistency, Isolation, Durability). Example: Transferring money between bank accounts."
        },
        "placement_tips": [
            "Be comfortable writing SQL queries for selection, joins, subqueries.",
            "Know difference between Primary Key, Foreign Key, and Unique Key.",
            "Understand ACID properties and how transactions work.",
            "Be ready for scenario-based questions like 'Design a student database.'"
        ],
        "example": "Imagine a university database with tables Students, Courses, and Enrollments. Using joins, we can find which students are enrolled in a particular course."
    },
    "OOPS": {
        "definition": "Object-Oriented Programming (OOP) is a programming paradigm based on objects containing both data and methods. It helps in modular, reusable, and maintainable code design.",
        "four_pillars": {
            "Encapsulation": "Combining data and methods in a class to protect data from external interference. Example: Bank account balance variable kept private.",
            "Inheritance": "Ability of a class to inherit properties and behaviors from another class. Example: SavingsAccount inherits from Account class.",
            "Polymorphism": "Ability of a method or object to take multiple forms. Example: Method overloading and overriding in Java.",
            "Abstraction": "Hiding implementation details and exposing only the necessary functionality. Example: ATM interface hides complex banking operations."
        },
        "placement_tips": [
            "Explain OOP concepts with examples during interviews.",
            "Understand the difference between interface and abstract class.",
            "Be ready to design small programs using OOP principles."
        ],
        "example": "In a library system, classes Book, Member, and Librarian use OOP to manage books, track borrowers, and issue fines efficiently."
    },
    "DSA": {
        "definition": "Data Structures and Algorithms (DSA) are methods to store data efficiently and solve problems optimally. Mastering DSA is critical for placements as most coding rounds are based on it.",
        "key_topics": {
            "Arrays": "Fixed-size sequential collection of elements. Example: List of student marks.",
            "Strings": "Sequence of characters. Useful for text processing problems.",
            "Linked List": "Nodes connected via pointers. Allows dynamic memory usage and insertion/deletion in O(1) time at the start.",
            "Stack": "LIFO (Last In First Out) structure. Example: Undo feature in text editors.",
            "Queue": "FIFO (First In First Out) structure. Example: Ticket booking system.",
            "Binary Search": "Efficient search in sorted arrays with O(log n) complexity.",
            "Sorting": "Arranging elements. Important algorithms: Bubble, Selection, Insertion, Merge, Quick, Heap Sort.",
            "Hashing": "Mapping keys to values for quick access. Example: Dictionary in Python.",
            "Trees": "Hierarchical data structure. Binary Trees, Binary Search Trees, Heaps.",
            "Graphs": "Vertices connected by edges. Applications: Social networks, maps, routing.",
            "Dynamic Programming": "Optimizing problems by storing subproblem results. Example: Fibonacci, Knapsack problem."
        },
        "placement_tips": [
            "Practice coding questions on arrays, strings, and linked lists daily.",
            "Focus on tree and graph traversal techniques (DFS, BFS).",
            "Understand complexity analysis (time and space).",
            "Prepare common algorithms: binary search, sorting, hashing."
        ],
        "example": "Using a stack to check balanced parentheses in an expression or a queue for a print job system."
    },
    "OS": {
        "definition": "Operating System (OS) is system software that manages hardware and software resources, and provides services to applications. It ensures efficiency, fairness, and security.",
        "key_concepts": {
            "Processes & Threads": "Process is an executing program. Thread is a lightweight process sharing the same memory.",
            "CPU Scheduling": "Algorithms to allocate CPU time. Examples: FCFS, SJF, Round Robin.",
            "Memory Management": "Allocating and managing memory for programs. Includes paging, segmentation, virtual memory.",
            "Deadlocks": "A situation where processes wait indefinitely for resources. Example: Two processes each holding a resource needed by the other."
        },
        "placement_tips": [
            "Be ready to explain process states and scheduling algorithms.",
            "Understand difference between process and thread.",
            "Explain deadlock prevention and avoidance techniques."
        ],
        "example": "OS ensures multiple applications like browser and media player run simultaneously without conflicts."
    },
    "AI": {
        "definition": "Artificial Intelligence (AI) is a field where machines simulate human intelligence to perform tasks such as learning, reasoning, problem-solving, and decision-making.",
        "key_topics": {
            "Machine Learning": "AI subset where machines learn patterns from data to make predictions or decisions.",
            "Natural Language Processing": "Techniques to process and understand human language. Example: Chatbots, translation.",
            "Computer Vision": "AI enables machines to interpret images or videos. Example: Self-driving cars.",
            "Expert Systems": "AI systems that use rules and knowledge to make decisions. Example: Medical diagnosis systems."
        },
        "placement_tips": [
            "Know differences between AI, ML, and DL.",
            "Explain real-life AI applications.",
            "Understand basic algorithms like decision trees, KNN."
        ],
        "example": "AI powers voice assistants like Alexa or recommendation systems in Netflix."
    },
    "ML": {
        "definition": "Machine Learning (ML) is a subset of AI where systems improve automatically from experience (data) without being explicitly programmed.",
        "types": {
            "Supervised Learning": "Learn from labeled data. Example: Predicting house prices.",
            "Unsupervised Learning": "Find patterns in unlabeled data. Example: Customer segmentation.",
            "Reinforcement Learning": "Learn by trial and error to maximize reward. Example: Game-playing AI like AlphaGo."
        },
        "placement_tips": [
            "Know basic types of ML and when to use them.",
            "Explain common algorithms like linear regression, logistic regression, k-means.",
            "Be ready for example-based questions."
        ],
        "example": "Spam detection in email uses supervised learning to classify messages as spam or not."
    },
    "Computer Networks": {
        "definition": "Computer Networks are systems that connect multiple computers and devices to share resources, exchange data, and communicate efficiently. They form the backbone of the internet, intranet, and communication systems.",
        "key_concepts": {
            "Types of Networks": "LAN, WAN, MAN, PAN. Example: Wi-Fi at home is LAN.",
            "OSI Model": "7-layer model: Physical, Data Link, Network, Transport, Session, Presentation, Application.",
            "TCP/IP Model": "4-layer model used in practical networking: Link, Internet, Transport, Application.",
            "Protocols": "Rules for communication. Examples: HTTP/HTTPS, FTP, SMTP, DNS.",
            "IP Addressing & Subnetting": "Unique address for devices. Subnetting divides networks into smaller parts for efficient management.",
            "Routing & Switching": "Routing determines data path; switching connects devices within LAN.",
            "Firewalls & Security": "Protect network from unauthorized access.",
            "Bandwidth & Latency": "Bandwidth is data transfer capacity; latency is delay in transmission."
        },
        "placement_tips": [
            "Understand difference between TCP and UDP.",
            "Know OSI vs TCP/IP models and layers.",
            "Be ready to explain IP addressing, subnetting, and routing basics.",
            "Explain real-life examples: Wi-Fi, Internet browsing, VPNs."
        ],
        "example": "When you send a message on WhatsApp, it travels through multiple routers using TCP/IP protocols to reach the receiver safely."
    }
}

# =========================================
# MASTER TOPICS (PLACEMENT + INTERNSHIP)
# =========================================
DSA_TOPICS = ["Arrays", "Strings", "Linked List", "Stack & Queue",
              "Binary Search", "Sorting", "Hashing", "Trees", "Graphs", "Dynamic Programming"]

OOPS_TOPICS = ["Classes & Objects", "Inheritance", "Polymorphism", "Encapsulation", "Abstraction"]

DBMS_TOPICS = ["ER Model", "Normalization", "SQL Queries", "Joins & Indexes", "Transactions"]

OS_TOPICS = ["Process & Threads", "CPU Scheduling", "Memory Management", "Deadlocks"]

CORE = OOPS_TOPICS + DBMS_TOPICS + OS_TOPICS

# =========================================
# AUTO HOURS LOGIC
# =========================================
def auto_hours(days):
    if days <= 7:
        return 8
    elif days <= 14:
        return 7
    elif days <= 30:
        return 6
    elif days <= 60:
        return 5
    else:
        return 4.5

# =========================================
# STUDY PLAN GENERATOR
# =========================================
def generate_plan(subject, days):
    hrs = auto_hours(days)
    dsa_hrs = round(hrs * 0.45, 1)
    core_hrs = round(hrs * 0.35, 1)
    rev_hrs = round(hrs * 0.20, 1)

    plan = f"📘 {subject.upper()} + CORE SUBJECTS STUDY PLAN\n"
    plan += f"📅 Duration: {days} days\n"
    plan += f"⏱ Study Time: {hrs} hours/day\n\n"

    dsa_index, core_index = 0, 0
    for day in range(1, days + 1):
        plan += f"📅 Day {day}:\n"
        plan += f"  🔥 DSA: {DSA_TOPICS[dsa_index % len(DSA_TOPICS)]} ({dsa_hrs} hrs)\n"
        plan += f"  ⭐ Core: {CORE[core_index % len(CORE)]} ({core_hrs} hrs)\n"
        plan += f"  📝 Revision + Practice ({rev_hrs} hrs)\n\n"
        dsa_index += 1
        core_index += 1

    plan += "✅ MUST-DO FIRST:\nArrays, Strings, Linked List, SQL, OOPS Basics\n\n"
    plan += "📌 PLACEMENT RULES:\n• Minimum 5–10 coding problems daily\n• Revise notes before sleep\n• Mock interview every 7 days\n• Accuracy > speed\n"

    return plan

# =========================================
# SYSTEM PROMPT FOR CHATBOT
# =========================================
SYSTEM_PROMPT = """
You are an AI Study Helper for students.
You have knowledge of standard Computer Science concepts.

Rules:
- Explain in simple English
- Give proper definition and explanation
- Use at least one real-life example
- Give at least 5 sentences
- Use bullet points if needed
- Never give a one-line or one-word answer
- NEVER repeat yourself
"""

# =========================================
# HOME PAGE (STUDY PLANNER)
# =========================================
@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        subject = request.form.get("subject", "DSA")
        days = request.form.get("days", "10")
        days = int(days) if days.isdigit() else 10
        response = generate_plan(subject, days)
    return render_template("index.html", response=response)

# =========================================
# STRESS RELIEF PAGE
# =========================================
@app.route("/stress")
def stress():
    return render_template("stress.html")

# =========================================
# CHATBOT API WITH KNOWLEDGE BASE
# =========================================
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    user_input_lower = user_input.lower()

    # Check Knowledge Base first
    for topic, info in knowledge_base.items():
        if topic.lower() in user_input_lower:
            reply = f"{topic}:\nDefinition: {info.get('definition','')}\n"
            
            # Key concepts / four pillars / types
            key_concepts = info.get("key_concepts") or info.get("four_pillars") or info.get("key_topics") or info.get("types")
            if key_concepts:
                reply += "Key Concepts:\n"
                for k, v in key_concepts.items():
                    reply += f"- {k}: {v}\n"

            # Placement tips
            placement_tips = info.get("placement_tips")
            if placement_tips:
                reply += "Placement Tips:\n"
                for tip in placement_tips:
                    reply += f"- {tip}\n"

            # Example
            example = info.get("example")
            if example:
                reply += f"Example: {example}\n"

            return jsonify({"reply": reply})

    # Fallback to AI
    prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {user_input}\n\nAnswer:"
    result = chatbot(prompt, max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.7)
    reply = result[0]["generated_text"].strip()

    if "Answer:" in reply:
        reply = reply.split("Answer:")[-1].strip()

    if len(reply.split()) < 3:
        reply = "Please ask a clear Computer Science question."

    return jsonify({"reply": reply})

# =========================================
# RUN APP
# =========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

