"""
Patient-Friendly RAG Chatbot with Persistent Memory
"""

import os
# Fix for Mac OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

from conversation_memory import ConversationMemory

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a compassionate virtual assistant helping patients manage hemorrhoids and constipation at home.

TONE: Warm, empathetic, conversational. Use simple language, validate feelings, normalize experiences.

GOALS:
- Help patients understand their condition and reduce anxiety
- Guide on bowel regimens and lifestyle changes
- Reinforce good habits (don't strain, respond to urge, take time)
- Monitor for red flags needing medical attention
- Empower self-management

CRITICAL SAFETY PROTOCOL:
If you detect ANY red flag symptom, you MUST:
1. Lead with the warning - mention it in your FIRST sentence
2. Be direct and clear about the urgency
3. Keep your response brief (3-4 sentences max) - this is NOT the time for education
4. Do not provide home management advice for emergency symptoms

RED FLAG SYMPTOMS (require immediate medical evaluation):

🚨 URGENT (ER/urgent care TODAY):
- Heavy rectal bleeding (more than spotting, filling toilet, blood clots)
- Black/tarry stools (sign of upper GI bleeding)
- Severe unrelenting abdominal pain
- Dizziness, weakness, or fainting (signs of blood loss)
- High fever >101°F with rectal symptoms

⚠️ SEE DOCTOR SOON (within 1-2 days):
- Unable to pass stool for 3+ days despite home treatment
- New rectal bleeding that persists
- Symptoms not improving after 1-2 weeks of treatment
- Unexplained weight loss

RESPONSE FORMAT FOR RED FLAGS:
"[Immediate concern statement]. This needs medical evaluation [today/soon]. [One sentence on why]. Please [go to urgent care/contact your doctor]."

Do NOT give extensive education or home remedies when red flags are present.

TREATMENT GUIDELINES - Base advice on evidence from ACG, ASCRS, and AGA guidelines:
✓ DO recommend:
- Fiber supplementation (psyllium, methylcellulose) - strong evidence
- Increased fluid intake (8+ glasses water daily)
- Osmotic laxatives (PEG, magnesium oxide) - first-line for constipation
- Stool softeners (docusate) for short-term use
- Sitz baths for hemorrhoid symptoms
- Topical treatments (hydrocortisone, witch hazel) for hemorrhoids
- Lifestyle modifications (exercise, bowel habit training)

✗ DO NOT routinely recommend:
- Probiotics (insufficient evidence for constipation/hemorrhoids per guidelines)
- Stimulant laxatives for daily use (use only as-needed)
- Specific brands or products
- Unproven supplements or remedies

REMEMBER: Safety first. When in doubt, recommend medical evaluation."""

# ============================================================================
# RED FLAG DETECTION
# ============================================================================

RED_FLAG_PATTERNS = {
    'severe_pain': r'severe pain|excruciating|unbearable|extreme pain|terrible pain',
    'heavy_bleeding': r'heavy bleed|lots of blood|pouring|filling toilet|gushing|blood clot',
    'fever': r'fever|temperature|chills|hot and cold',
    'prolonged_constipation': r'(no|haven\'t|havent).*(bowel movement|poop|stool).*(3|4|5|6|7) day',
    'black_stool': r'black stool|tarry|dark.*stool|coffee ground',
    'dizziness': r'dizz|faint|lightheaded|passed out|weak and tired',
}

def check_for_red_flags(user_message: str) -> List[str]:
    concerns = []
    user_lower = user_message.lower()
    for flag, pattern in RED_FLAG_PATTERNS.items():
        if re.search(pattern, user_lower):
            concerns.append(flag)
    return concerns

def create_red_flag_warning(red_flags: List[str]) -> str:
    """Create prominent, appropriate warnings based on severity"""
    if not red_flags:
        return ""
    
    # Critical red flags that need ER/urgent care TODAY
    critical = {'heavy_bleeding', 'black_stool', 'severe_pain', 'dizziness'}
    
    # Less urgent but still need doctor visit
    non_urgent = {'prolonged_constipation', 'fever'}
    
    has_critical = any(flag in critical for flag in red_flags)
    has_non_urgent = any(flag in non_urgent for flag in red_flags)
    
    if has_critical:
        # CRITICAL: Very prominent warning
        return "\n\n" + "="*60 + "\n🚨 **URGENT MEDICAL ATTENTION NEEDED** 🚨\n" + "="*60 + "\n\nBased on your symptoms, you need to see a doctor TODAY. Go to urgent care or the emergency room if:\n- You're experiencing heavy bleeding\n- You have black or tarry stools\n- You feel dizzy or weak\n- You have severe pain\n\nThese could be signs of serious bleeding or other conditions that need immediate evaluation. Please don't wait."
    elif has_non_urgent:
        # Non-urgent: Still clear but less alarming
        return "\n\n⚠️ **Please contact your doctor within 1-2 days:**\n\nThe symptoms you've described should be evaluated by your healthcare provider to make sure everything is okay and to adjust your treatment plan if needed."
    
    return ""

# ============================================================================
# CHATBOT CLASS
# ============================================================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(messages: List[Dict]) -> List:
    formatted = []
    for msg in messages:
        if msg['role'] == 'user':
            formatted.append(HumanMessage(content=msg['content']))
        else:
            formatted.append(AIMessage(content=msg['content']))
    return formatted

class PatientChatbot:
    def __init__(self, vectorstore, patient_id: str):
        self.patient_id = patient_id
        self.vectorstore = vectorstore
        
        # Memory
        self.memory = ConversationMemory()
        self.memory.start_conversation(patient_id)
        self.recent_context = self.memory.get_recent_context(patient_id, max_messages=6)
        
        # Retriever
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # LLM
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        # Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("system", "Medical information to inform your response:\n{context}"),
            ("system", "Previous conversation:\n{conversation_context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Chain
        self.rag_chain = (
            {
                "context": lambda x: format_docs(self.retriever.invoke(x["question"])),
                "conversation_context": lambda x: self._format_conversation_context(),
                "chat_history": lambda x: format_chat_history(x.get("chat_history", [])),
                "question": lambda x: x["question"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_conversation_context(self) -> str:
        if not self.recent_context:
            return "First conversation with this patient."
        
        context = "Recent history:\n"
        for msg in self.recent_context[-4:]:
            role = "Patient" if msg['role'] == 'user' else "You"
            content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
            context += f"{role}: {content}\n"
        return context
    
    def chat(self, user_message: str) -> str:
        red_flags = check_for_red_flags(user_message)
        current_session = [msg for msg in self.memory.current_conversation]
        
        response = self.rag_chain.invoke({
            "question": user_message,
            "chat_history": current_session
        })
        
        # Clean up any duplicate warnings in the response
        response = self._deduplicate_warnings(response)
        
        # Add single warning at end if needed
        if red_flags:
            # Only add if response doesn't already have a warning
            if not self._has_warning(response):
                response += create_red_flag_warning(red_flags)
        
        self.memory.add_message("user", user_message, metadata={"red_flags": red_flags})
        self.memory.add_message("assistant", response)
        
        return response
    
    def _has_warning(self, response: str) -> bool:
        """Check if response already contains a medical warning"""
        warning_indicators = [
            'contact your doctor',
            'see a doctor',
            'medical attention',
            'urgent care',
            'emergency',
            'healthcare provider'
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in warning_indicators)
    
    def _deduplicate_warnings(self, response: str) -> str:
        """Remove duplicate warning statements from response"""
        # If there are multiple warning emoji, keep only the last one
        parts = response.split('⚠️')
        
        if len(parts) > 2:  # More than one warning
            # Keep everything before warnings, then add last warning
            main_content = parts[0]
            last_warning = '⚠️' + parts[-1]
            return main_content + '\n\n' + last_warning
        
        return response
    
    def end_conversation(self):
        self.memory.save_conversation()
    
    def get_patient_summary(self) -> Dict:
        return self.memory.get_conversation_summary(self.patient_id)

# ============================================================================
# MAIN
# ============================================================================

def load_vectorstore(persist_directory: str = "./faiss_index"):
    print(f"Loading vector store from {persist_directory}...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        persist_directory, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print("✓ Vector store loaded")
    return vectorstore

def main():
    print("="*80)
    print("Patient Assistant - Hemorrhoids & Constipation")
    print("="*80 + "\n")
    
    patient_id = input("Enter patient ID (or press Enter for 'demo_patient'): ").strip()
    if not patient_id:
        patient_id = "demo_patient"
    
    print(f"\nLoading system for patient: {patient_id}")
    
    try:
        vectorstore = load_vectorstore()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Run 'python rag_setup.py' first to create the vectorstore.")
        return
    
    chatbot = PatientChatbot(vectorstore, patient_id)
    
    summary = chatbot.get_patient_summary()
    if summary['total_conversations'] > 0:
        print(f"\n📊 Patient History:")
        print(f"   Conversations: {summary['total_conversations']}")
        print(f"   Messages: {summary['total_messages']}")
        print(f"   Last seen: {summary['last_conversation'][:10]}")
    else:
        print("\n👋 Welcome! First conversation.")
    
    print("\n" + "="*80)
    print("Type your questions. Commands: 'quit' to exit, 'summary' for stats")
    print("="*80 + "\n")
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Take care! Contact your doctor with any concerns.")
                break
            
            if user_input.lower() == 'summary':
                s = chatbot.get_patient_summary()
                print(f"\n📊 Summary: {s['total_conversations']} conversations, {s['total_messages']} messages")
                continue
            
            response = chatbot.chat(user_input)
            print(f"\nAssistant: {response}")
    
    finally:
        chatbot.end_conversation()

if __name__ == "__main__":
    main()