import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { Message } from './Message';
import { api } from '../api';
import type { Source } from '../api';

interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    sources?: Source[];
}

interface ChatAreaProps {
    currentCollection: string;
}

export const ChatArea: React.FC<ChatAreaProps> = ({ currentCollection }) => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const activeCollectionRef = useRef(currentCollection);
    const requestIdRef = useRef(0);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        // Reset chat state whenever the collection changes to keep a single thread per collection
        activeCollectionRef.current = currentCollection;
        setMessages([]);
        setInput('');
        setIsLoading(false);
        requestIdRef.current += 1; // invalidate any in-flight responses
    }, [currentCollection]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || !currentCollection) return;

        const collectionAtSend = currentCollection;
        const requestId = requestIdRef.current + 1;
        requestIdRef.current = requestId;
        const userMessage = input;
        setInput('');
        setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
        setIsLoading(true);

        try {
            const response = await api.chat(currentCollection, userMessage);
            if (collectionAtSend !== activeCollectionRef.current || requestId !== requestIdRef.current) {
                return;
            }
            setMessages((prev) => [
                ...prev,
                {
                    role: 'assistant',
                    content: response.answer,
                    sources: response.sources,
                },
            ]);
        } catch (error) {
            console.error('Chat failed:', error);
            setMessages((prev) => [
                ...prev,
                { role: 'assistant', content: 'Sorry, I encountered an error processing your request.' },
            ]);
        } finally {
            if (requestId === requestIdRef.current) {
                setIsLoading(false);
            }
        }
    };

    if (!currentCollection) {
        return (
            <div className="chat-area empty-state">
                <div className="empty-content">
                    <h1>Welcome to DocuSage AI</h1>
                    <p>Select or create a collection to start chatting with your documents.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="chat-area">
            <div className="messages-scroll">
                {messages.map((msg, idx) => (
                    <Message key={idx} {...msg} />
                ))}
                {isLoading && (
                    <div className="message-container assistant loading">
                        <div className="avatar"><Loader2 className="spin" size={24} /></div>
                        <div className="message-content">Thinking...</div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="input-area">
                <form onSubmit={handleSubmit} className="input-form">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask a question about your documents..."
                        disabled={isLoading}
                    />
                    <button type="submit" disabled={isLoading || !input.trim()}>
                        <Send size={20} />
                    </button>
                </form>
            </div>
        </div>
    );
};
