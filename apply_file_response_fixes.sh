#!/bin/bash

echo "🔄 Converting Chatty AI to use file-based jokes and fun facts..."

# Create backup
cp chatty_ai.py chatty_ai.py.backup_before_file_responses

# 1. Add new file constants near the top (after line 56)
echo "1. Adding new file constants..."
sed -i '/^JOKES_FILE = "jokes.txt"/a\
FUN_FACTS_FILE = "fun_facts.txt"\
BORED_RESPONSES_GENERIC_FILE = "bored_responses_generic.txt"\
WAITING_RESPONSES_GENERIC_FILE = "waiting_responses_generic.txt"' chatty_ai.py

# 2. Create the new response files with default content
echo "2. Creating default response files..."

# Create fun_facts.txt
cat > fun_facts.txt << 'EOF'
Did you know that octopuses have three hearts and blue blood?
Honey never spoils - archaeologists have found edible honey in ancient Egyptian tombs!
A group of flamingos is called a 'flamboyance'.
The human brain uses about 20% of the body's total energy.
Bananas are berries, but strawberries aren't!
There are more possible games of chess than atoms in the observable universe.
Wombat poop is cube-shaped to prevent it from rolling away.
A cloud can weigh more than a million pounds.
Sharks have been around longer than trees.
The shortest war in history lasted only 38-45 minutes.
Your stomach gets an entirely new lining every 3-4 days.
A single cloud can contain over a billion water droplets.
EOF

# Create bored_responses_generic.txt  
cat > bored_responses_generic.txt << 'EOF'
I'm getting a bit bored waiting here
Still hanging around here waiting
I'm patiently waiting for commands
I am feeling restless waiting here
Still here waiting to help
Getting a little restless over here
Waiting around for someone to talk to
Still here if anyone needs assistance
EOF

# Create waiting_responses_generic.txt
cat > waiting_responses_generic.txt << 'EOF'
I am still around if you need me
Still here waiting to assist
Patiently waiting for your next request
I'm here whenever you need help
Standing by for assistance
Ready to help when you are
Waiting here to be of service
Available for any questions you might have
EOF

# Update existing bored_responses.txt to include {name} placeholder
cat > bored_responses.txt << 'EOF'
Hey {name}, I'm getting a bit bored waiting here
{name}, still hanging around here waiting for you, dude
I'm patiently waiting for your commands, {name}
Hey {name}, I am feeling restless waiting here
Still here waiting to help you, {name}
{name}, getting a little restless over here
Waiting around for you to talk to me, {name}
Still here if you need anything, {name}
EOF

# Update existing waiting_responses.txt to include {name} placeholder
cat > waiting_responses.txt << 'EOF'
Hey {name}, I am still here if you need anything
{name}, still here waiting to assist you
Patiently waiting for your next request, {name}
I'm here whenever you need help, {name}
Standing by for you, {name}
Ready to help when you are, {name}
{name}, waiting here to be of service
Available for any questions you might have, {name}
EOF

# Update greeting_responses.txt to include {name} placeholder
cat > greeting_responses.txt << 'EOF'
Hello {name}! It is nice to see you again. How may I help you?
Hey {name}! Good to see you, buddy! What's up?
Hi there {name}! Great to see you again. What can I do for you?
Welcome back {name}! How are you doing today?
{name}! Nice to see your face again. What brings you here?
Hello {name}! Always a pleasure. What would you like to know?
Hey {name}! Ready to help with whatever you need.
Hi {name}! Good to have you back. How can I assist you?
EOF

echo "✅ Response files created with personalized content!"

# 3. Find and show line numbers for manual method replacements
echo ""
echo "📍 MANUAL METHOD REPLACEMENTS NEEDED:"
echo "===================================="

echo ""
echo "METHOD 1: Replace get_llm_joke method"
joke_line=$(grep -n "def get_llm_joke" chatty_ai.py | cut -d: -f1)
echo "   Found at line: $joke_line"
echo "   Replace entire method with get_file_joke from artifact"

echo ""
echo "METHOD 2: Replace get_llm_fun_fact method"  
fact_line=$(grep -n "def get_llm_fun_fact" chatty_ai.py | cut -d: -f1)
echo "   Found at line: $fact_line"
echo "   Replace entire method with get_file_fun_fact from artifact"

echo ""
echo "METHOD 3: Update load_response_files method"
load_line=$(grep -n "def load_response_files" chatty_ai.py | cut -d: -f1)
echo "   Found at line: $load_line"
echo "   Add the new file loading sections from artifact"

echo ""
echo "METHOD 4: Update __init__ method"
init_line=$(grep -n "self.bored_responses = \[\]" chatty_ai.py | cut -d: -f1)
echo "   Found bored_responses at line: $init_line"
echo "   Add new instance variables after this line:"
echo "       self.fun_facts = []"
echo "       self.bored_responses_generic = []"  
echo "       self.waiting_responses_generic = []"

echo ""
echo "METHOD 5: Update check_for_bored_response method"
bored_line=$(grep -n "def check_for_bored_response" chatty_ai.py | cut -d: -f1)
echo "   Found at line: $bored_line" 
echo "   Replace entire method with new version from artifact"

echo ""
echo "🎯 NEXT STEPS:"
echo "1. ✅ Response files have been created automatically"
echo "2. 📝 Manually replace the 5 methods using the code from the artifact above"
echo "3. 🔄 Update method calls: get_llm_joke → get_file_joke, get_llm_fun_fact → get_file_fun_fact"
echo "4. 🧪 Test the system to ensure file-based responses work correctly"

echo ""
echo "📁 FILES CREATED:"
echo "- fun_facts.txt (8 default facts)"
echo "- bored_responses_generic.txt (8 generic responses)"
echo "- waiting_responses_generic.txt (8 generic responses)"
echo "- Updated bored_responses.txt with {name} placeholders"
echo "- Updated waiting_responses.txt with {name} placeholders" 
echo "- Updated greeting_responses.txt with {name} placeholders"

echo ""
echo "🎭 PERSONALIZATION FEATURES:"
echo "- Known persons: Use personalized responses with their name"
echo "- Unknown persons/strangers: Use generic responses without names"
echo "- All greeting responses now include the detected person's name"
echo "- File-based jokes and facts for consistent quality"