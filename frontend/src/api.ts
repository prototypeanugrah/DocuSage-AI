import axios from 'axios';

const API_URL = 'http://localhost:8000';

export interface ChatResponse {
  answer: string;
  sources: Source[];
  focus_source: string | null;
}

export interface Source {
  page_content: string;
  metadata: {
    source: string;
    page: number;
    [key: string]: any;
  };
}

export const api = {
  getCollections: async () => {
    const response = await axios.get<{ collections: string[] }>(`${API_URL}/collections`);
    return response.data;
  },

  indexDocuments: async (collectionName: string, files: File[]) => {
    const formData = new FormData();
    formData.append('collection_name', collectionName);
    files.forEach((file) => {
      formData.append('files', file);
    });

    const response = await axios.post(`${API_URL}/index`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  chat: async (collectionName: string, question: string) => {
    const response = await axios.post<ChatResponse>(`${API_URL}/chat`, {
      collection_name: collectionName,
      question,
    });
    return response.data;
  },
};
