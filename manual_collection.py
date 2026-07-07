"""
Manual Test Case Collection Tool
Quick way to add real patient questions you find online
"""

import json
from datetime import datetime
from pathlib import Path

class ManualQuestionCollector:
    """
    Interactive tool to quickly add patient questions you find
    """
    
    def __init__(self):
        self.questions = []
        self.output_dir = Path("test_data")
        self.output_dir.mkdir(exist_ok=True)
    
    def add_question(self):
        """Add a single question interactively"""
        print("\n" + "="*60)
        print("ADD NEW QUESTION")
        print("="*60)
        
        question = input("\nPaste the question: ").strip()
        if not question:
            return False
        
        print("\nCategory options:")
        print("1. symptom_identification")
        print("2. treatment_options")
        print("3. when_to_see_doctor")
        print("4. prevention")
        print("5. medication")
        print("6. lifestyle")
        print("7. general")
        
        cat_choice = input("\nSelect category (1-7): ").strip()
        categories = {
            '1': 'symptom_identification',
            '2': 'treatment_options',
            '3': 'when_to_see_doctor',
            '4': 'prevention',
            '5': 'medication',
            '6': 'lifestyle',
            '7': 'general'
        }
        category = categories.get(cat_choice, 'general')
        
        source = input("Source (e.g., HealthBoards, WebMD, Reddit): ").strip() or "Manual"
        url = input("URL (optional, press Enter to skip): ").strip() or ""
        
        self.questions.append({
            'id': f"manual_{len(self.questions) + 1:03d}",
            'title': question,
            'body': '',
            'source': source,
            'url': url,
            'category': category,
            'added_date': datetime.now().isoformat()
        })
        
        print(f"\n✓ Added question #{len(self.questions)}")
        return True
    
    def batch_add_from_text(self, text_input: str):
        """Add multiple questions from pasted text (one per line)"""
        lines = [line.strip() for line in text_input.split('\n') if line.strip()]
        
        for line in lines:
            self.questions.append({
                'id': f"manual_{len(self.questions) + 1:03d}",
                'title': line,
                'body': '',
                'source': 'Manual',
                'url': '',
                'category': self._auto_categorize(line),
                'added_date': datetime.now().isoformat()
            })
        
        print(f"✓ Added {len(lines)} questions")
    
    def _auto_categorize(self, text: str) -> str:
        """Automatically categorize based on keywords"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['is this', 'what is', 'do i have', 'symptoms']):
            return 'symptom_identification'
        elif any(word in text_lower for word in ['how to', 'treat', 'cure', 'remedy', 'help']):
            return 'treatment_options'
        elif any(word in text_lower for word in ['should i see', 'doctor', 'emergency', 'urgent']):
            return 'when_to_see_doctor'
        elif any(word in text_lower for word in ['prevent', 'avoid', 'stop']):
            return 'prevention'
        elif any(word in text_lower for word in ['medication', 'drug', 'prescription']):
            return 'medication'
        elif any(word in text_lower for word in ['diet', 'food', 'fiber', 'eat']):
            return 'lifestyle'
        else:
            return 'general'
    
    def load_existing(self, filename: str = "manual_test_cases.json"):
        """Load previously saved questions"""
        filepath = self.output_dir / filename
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.questions = data.get('questions', [])
                print(f"✓ Loaded {len(self.questions)} existing questions")
        else:
            print("No existing questions found")
    
    def save(self, filename: str = "manual_test_cases.json"):
        """Save questions to file"""
        filepath = self.output_dir / filename
        
        data = {
            'scraped_date': datetime.now().isoformat(),
            'total_questions': len(self.questions),
            'collection_method': 'manual',
            'questions': self.questions
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved {len(self.questions)} questions to {filepath}")
        
        # Show summary
        categories = {}
        for q in self.questions:
            cat = q['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nQuestion Categories:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")
    
    def show_all(self):
        """Display all collected questions"""
        if not self.questions:
            print("\nNo questions collected yet")
            return
        
        print(f"\n{'='*60}")
        print(f"COLLECTED QUESTIONS ({len(self.questions)} total)")
        print(f"{'='*60}")
        
        for i, q in enumerate(self.questions, 1):
            print(f"\n{i}. [{q['category']}]")
            print(f"   {q['title']}")
            if q['source'] != 'Manual':
                print(f"   Source: {q['source']}")
    
    def delete_question(self, index: int):
        """Delete a question by index"""
        if 0 <= index < len(self.questions):
            deleted = self.questions.pop(index)
            print(f"✓ Deleted: {deleted['title'][:50]}...")
        else:
            print("Invalid index")

def interactive_mode():
    """Interactive collection mode"""
    collector = ManualQuestionCollector()
    
    # Load existing
    collector.load_existing()
    
    print("\n" + "="*60)
    print("MANUAL TEST CASE COLLECTOR")
    print("="*60)
    print("\nCommands:")
    print("  add    - Add a single question")
    print("  batch  - Paste multiple questions (one per line)")
    print("  show   - Display all questions")
    print("  delete - Remove a question")
    print("  save   - Save questions")
    print("  quit   - Save and exit")
    
    while True:
        cmd = input("\nCommand: ").strip().lower()
        
        if cmd == 'add':
            collector.add_question()
            
        elif cmd == 'batch':
            print("\nPaste questions (one per line, then empty line to finish):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            
            if lines:
                collector.batch_add_from_text('\n'.join(lines))
        
        elif cmd == 'show':
            collector.show_all()
        
        elif cmd == 'delete':
            collector.show_all()
            idx = int(input("\nEnter question number to delete: ")) - 1
            collector.delete_question(idx)
        
        elif cmd == 'save':
            collector.save()
        
        elif cmd == 'quit':
            collector.save()
            print("\nGoodbye!")
            break
        
        else:
            print("Unknown command. Try: add, batch, show, delete, save, quit")

# Pre-populated starter questions you can use immediately
STARTER_QUESTIONS = [
    ("I've had blood on toilet paper for a week, is this normal?", "symptom_identification"),
    ("Can stress cause hemorrhoids to flare up?", "general"),
    ("I'm constipated and nothing works - tried everything", "treatment_options"),
    ("Is it OK to push hard when constipated?", "prevention"),
    ("How long should I wait before seeing a doctor for hemorrhoids?", "when_to_see_doctor"),
    ("Can I exercise with hemorrhoids?", "lifestyle"),
    ("Does drinking coffee make constipation worse?", "lifestyle"),
    ("I feel a lump near my anus, should I panic?", "symptom_identification"),
    ("What's the difference between internal and external hemorrhoids?", "symptom_identification"),
    ("Can hemorrhoids cause narrow stools?", "symptom_identification"),
    ("I've been taking MiraLAX daily for months, is that safe?", "medication"),
    ("My hemorrhoids itch like crazy at night, what can I do?", "treatment_options"),
    ("Can you get hemorrhoids from sitting on cold surfaces?", "prevention"),
    ("I'm pregnant and constipated, what's safe to take?", "medication"),
    ("How much fiber is too much fiber?", "lifestyle"),
    ("Can hemorrhoids go away on their own?", "general"),
    ("I have alternating constipation and diarrhea", "symptom_identification"),
    ("Is bleeding after hemorrhoid banding normal?", "treatment_options"),
    ("Can spicy food cause hemorrhoids?", "prevention"),
    ("I feel like I need to poop but nothing comes out", "symptom_identification"),
]

def create_starter_file():
    """Create a starter file with common questions"""
    collector = ManualQuestionCollector()
    
    for question, category in STARTER_QUESTIONS:
        collector.questions.append({
            'id': f"manual_{len(collector.questions) + 1:03d}",
            'title': question,
            'body': '',
            'source': 'Curated',
            'url': '',
            'category': category,
            'added_date': datetime.now().isoformat()
        })
    
    collector.save()
    print("\n✓ Created starter file with 20 common patient questions!")
    print("Run 'python manual_collection.py' to add more")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'starter':
        create_starter_file()
    else:
        interactive_mode()