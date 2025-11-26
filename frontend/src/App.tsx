import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { api } from './api';
import './App.css'; // We'll use App.css for global styles instead of index.css for simplicity

function App() {
  const [collections, setCollections] = useState<string[]>([]);
  const [currentCollection, setCurrentCollection] = useState<string>('');

  console.log('Current collection:', currentCollection);

  const refreshCollections = async () => {
    try {
      const data = await api.getCollections();
      setCollections(data.collections);

      const hasCurrent = currentCollection && data.collections.includes(currentCollection);
      if (hasCurrent) return;

      const fallback = data.collections[0] ?? '';
      setCurrentCollection(fallback);
    } catch (error) {
      console.error('Failed to fetch collections:', error);
    }
  };

  useEffect(() => {
    refreshCollections();
  }, []);

  return (
    <div className="app-container">
      <Sidebar
        currentCollection={currentCollection}
        onCollectionChange={setCurrentCollection}
        collections={collections}
        refreshCollections={refreshCollections}
      />
      <ChatArea currentCollection={currentCollection} />
    </div>
  );
}

export default App;
