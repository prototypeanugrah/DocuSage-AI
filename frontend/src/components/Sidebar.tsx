import React, { useState, useRef } from 'react';
import { Upload, FileText, Database, Loader2 } from 'lucide-react';
import { api } from '../api';

interface SidebarProps {
    currentCollection: string;
    onCollectionChange: (collection: string) => void;
    collections: string[];
    refreshCollections: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
    currentCollection,
    onCollectionChange,
    collections,
    refreshCollections,
}) => {
    const [isUploading, setIsUploading] = useState(false);
    const [newCollectionName, setNewCollectionName] = useState('');
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const files = event.target.files;
        if (!files || files.length === 0) return;
        if (!newCollectionName && !currentCollection) {
            alert("Please enter a collection name or select an existing one.");
            return;
        }

        const collectionToUse = newCollectionName || currentCollection;

        setIsUploading(true);
        try {
            await api.indexDocuments(collectionToUse, Array.from(files));
            refreshCollections();
            onCollectionChange(collectionToUse);
            setNewCollectionName('');
            if (fileInputRef.current) fileInputRef.current.value = '';
            alert('Documents processed successfully!');
        } catch (error) {
            console.error('Upload failed:', error);
            alert('Failed to process documents.');
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="sidebar">
            <div className="sidebar-header">
                <h2><Database className="icon" /> DocuSage AI</h2>
            </div>

            <div className="sidebar-section">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <h3>Collections</h3>
                    <button onClick={refreshCollections} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }} title="Refresh List">
                        <Loader2 size={12} />
                    </button>
                </div>
                <div className="collection-list">
                    {collections.length === 0 ? (
                        <div style={{ padding: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.85rem', fontStyle: 'italic' }}>
                            No collections found.
                        </div>
                    ) : (
                        collections.map((col) => (
                            <button
                                key={col}
                                className={`collection-item ${currentCollection === col ? 'active' : ''}`}
                                onClick={() => onCollectionChange(col)}
                            >
                                <FileText className="icon-sm" />
                                {col}
                            </button>
                        ))
                    )}
                </div>
            </div>

            <div className="sidebar-section mt-auto">
                <h3>Collection Management</h3>
                <div className="input-group">
                    <input
                        type="text"
                        placeholder="Collection Name"
                        value={newCollectionName}
                        onChange={(e) => setNewCollectionName(e.target.value)}
                        className="input-field"
                    />
                    <button
                        className="btn-secondary"
                        onClick={() => {
                            if (newCollectionName) {
                                console.log('Loading collection:', newCollectionName);
                                onCollectionChange(newCollectionName);
                                setNewCollectionName('');
                            }
                        }}
                        disabled={!newCollectionName}
                        title="Switch to this collection"
                    >
                        Load
                    </button>
                </div>

                <input
                    type="file"
                    multiple
                    accept=".pdf"
                    ref={fileInputRef}
                    style={{ display: 'none' }}
                    onChange={handleFileUpload}
                />
                <button
                    className="btn-primary full-width mt-2"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                >
                    {isUploading ? (
                        <><Loader2 className="icon-sm spin" /> Processing...</>
                    ) : (
                        <><Upload className="icon-sm" /> Upload & Process</>
                    )}
                </button>
            </div>
        </div>
    );
};
