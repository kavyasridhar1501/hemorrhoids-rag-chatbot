"""
Persistent Conversation Memory System
Stores and retrieves conversation history across sessions
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

class ConversationMemory:
    """Manages persistent conversation history for patients."""
    
    def __init__(self, storage_dir: str = "conversations"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.current_conversation = []
        self.patient_id = None
    
    def start_conversation(self, patient_id: str) -> Dict:
        """Start a new conversation for a patient"""
        self.patient_id = patient_id
        self.current_conversation = []
        
        history = self.get_patient_history(patient_id)
        
        return {
            'patient_id': patient_id,
            'conversation_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'started_at': datetime.now().isoformat(),
            'previous_conversations': len(history)
        }
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the current conversation"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.current_conversation.append(message)
    
    def save_conversation(self):
        """Save current conversation to disk"""
        if not self.patient_id:
            raise ValueError("No active conversation")
        
        if not self.current_conversation:
            return
        
        patient_dir = self.storage_dir / self.patient_id
        patient_dir.mkdir(exist_ok=True)
        
        conversation_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = patient_dir / f"conversation_{conversation_id}.json"
        
        data = {
            'patient_id': self.patient_id,
            'conversation_id': conversation_id,
            'started_at': self.current_conversation[0]['timestamp'],
            'ended_at': datetime.now().isoformat(),
            'message_count': len(self.current_conversation),
            'messages': self.current_conversation
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Conversation saved: {filename}")
    
    def get_patient_history(self, patient_id: str) -> List[Dict]:
        """Get all conversations for a patient"""
        patient_dir = self.storage_dir / patient_id
        
        if not patient_dir.exists():
            return []
        
        conversations = []
        for file in sorted(patient_dir.glob("conversation_*.json")):
            with open(file, 'r', encoding='utf-8') as f:
                conversations.append(json.load(f))
        
        return conversations
    
    def get_recent_context(self, patient_id: str, max_messages: int = 10) -> List[Dict]:
        """Get recent conversation context for a patient"""
        history = self.get_patient_history(patient_id)
        
        if not history:
            return []
        
        all_messages = []
        for conversation in reversed(history):
            all_messages.extend(conversation['messages'])
            if len(all_messages) >= max_messages:
                break
        
        return all_messages[-max_messages:]
    
    def get_conversation_summary(self, patient_id: str) -> Dict:
        """Get summary statistics for a patient's conversations"""
        history = self.get_patient_history(patient_id)
        
        if not history:
            return {
                'total_conversations': 0,
                'total_messages': 0,
                'first_conversation': None,
                'last_conversation': None
            }
        
        total_messages = sum(conv['message_count'] for conv in history)
        
        red_flags = []
        for conversation in history:
            for message in conversation['messages']:
                if message.get('metadata', {}).get('red_flags'):
                    red_flags.extend(message['metadata']['red_flags'])
        
        return {
            'total_conversations': len(history),
            'total_messages': total_messages,
            'first_conversation': history[0]['started_at'],
            'last_conversation': history[-1]['started_at'],
            'red_flags_detected': len(red_flags),
            'unique_red_flags': list(set(red_flags))
        }
    
    def clear_current_conversation(self):
        """Clear the current conversation without saving"""
        self.current_conversation = []
        self.patient_id = None