import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Bot, User, ChevronDown, ChevronRight, FileText } from 'lucide-react';
import type { Source } from '../api';
import { motion, AnimatePresence } from 'framer-motion';

interface MessageProps {
    role: 'user' | 'assistant';
    content: string;
    sources?: Source[];
}

export const Message: React.FC<MessageProps> = ({ role, content, sources }) => {
    const [showSources, setShowSources] = useState(false);

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`message-container ${role === 'assistant' ? 'assistant' : 'user'}`}
        >
            <div className="avatar">
                {role === 'assistant' ? <Bot size={24} /> : <User size={24} />}
            </div>
            <div className="message-content">
                <div className="markdown-body">
                    <ReactMarkdown>{content}</ReactMarkdown>
                </div>

                {sources && sources.length > 0 && (
                    <div className="sources-section">
                        <button
                            className="sources-toggle"
                            onClick={() => setShowSources(!showSources)}
                        >
                            {showSources ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                            {sources.length} Sources
                        </button>

                        <AnimatePresence>
                            {showSources && (
                                <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: 'auto', opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    className="sources-list"
                                >
                                    {sources.map((source, idx) => (
                                        <div key={idx} className="source-item">
                                            <div className="source-header">
                                                <FileText size={14} />
                                                <span>{source.metadata.source.split('/').pop()} - Page {source.metadata.page}</span>
                                            </div>
                                            <p className="source-text">{source.page_content.slice(0, 200)}...</p>
                                        </div>
                                    ))}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                )}
            </div>
        </motion.div>
    );
};
