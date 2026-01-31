import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../lib/authStore';
import apiClient from '../lib/api';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Layout, Sparkles, Brain, Clock, Activity, Send, X, LogOut, Settings, MessageSquare, Database
} from 'lucide-react';
import {
    ResponsiveContainer, AreaChart, Area, BarChart, Bar,
    XAxis, YAxis, CartesianGrid, Tooltip
} from 'recharts';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    confidence?: number;
    timestamp?: string;
    intent?: string;
    data?: any[];
    sql?: string;
    execution_time?: number;
    insights?: string[];
}

interface AISettings {
    language: 'en';
    theme: 'dark';
    confidenceThreshold: number;
    showConfidenceScores: boolean;
    proactiveSuggestions: boolean;
    autoExport: boolean;
    showSQL: boolean;
    showExecutionTime: boolean;
    advancedMode: boolean;
}

const CHART_COLORS = ['#818cf8', '#c084fc', '#f472b6', '#fbbf24', '#34d399', '#f87171'];

export default function Chat() {
    const { user, logout } = useAuthStore();
    const navigate = useNavigate();

    // State
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [showSettings, setShowSettings] = useState(false);
    const [selectedChartType, setSelectedChartType] = useState<'area' | 'bar'>('area');
    const [conversations, setConversations] = useState<any[]>([]);
    const [selectedConvId, setSelectedConvId] = useState<string | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const [settings, setSettings] = useState<AISettings>({
        language: 'en',
        theme: 'dark',
        confidenceThreshold: 70,
        showConfidenceScores: true,
        proactiveSuggestions: true,
        autoExport: false,
        showSQL: true,
        showExecutionTime: true,
        advancedMode: true
    });



    // Auto-scroll to bottom
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Initial Fetch
    useEffect(() => {
        fetchConversations();
    }, []);

    const fetchConversations = async () => {
        try {
            const response = await apiClient.get('/conversations/');
            const data = Array.isArray(response.data) ? response.data : [];
            setConversations(data);

            // Auto-load last if none selected
            if (data.length > 0 && !selectedConvId) {
                loadConversation(data[0].id);
            }
        } catch (error) {
            console.error('Failed to fetch conversations:', error);
            setConversations([]);
        }
    };

    const loadConversation = async (id: string) => {
        setLoading(true);
        setSelectedConvId(id);
        try {
            const response = await apiClient.get(`/conversations/${id}/messages/`);
            const data = Array.isArray(response.data) ? response.data : [];
            setMessages(data.map((m: any) => ({
                role: m.role,
                content: m.content,
                timestamp: m.created_at,
                intent: m.intent,
                sql: m.sql_query,
                execution_time: m.processing_time
            })));
        } catch (error) {
            console.error('Failed to load messages:', error);
            setMessages([]);
        } finally {
            setLoading(false);
        }
    };

    const startNewSession = () => {
        setSelectedConvId(null);
        setMessages([]);
    };

    const sendMessage = async () => {
        if (!input.trim()) return;

        const userMessage: Message = {
            role: 'user',
            content: input,
            timestamp: new Date().toISOString()
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        try {
            const response = await apiClient.post(
                '/chat/',
                {
                    message: input,
                    conversation_id: selectedConvId,
                    language: settings.language,
                    advanced_mode: settings.advancedMode
                }
            );

            const aiMessage: Message = {
                role: 'assistant',
                content: response.data.response,
                confidence: response.data.confidence || 92,
                timestamp: new Date().toISOString(),
                intent: response.data.intent,
                data: response.data.data,
                sql: response.data.sql,
                execution_time: response.data.execution_time,
                insights: response.data.insights || []
            };

            setMessages(prev => [...prev, aiMessage]);

            // Handle conversation state updates (ID and Title)
            if (!selectedConvId) {
                setSelectedConvId(response.data.conversation_id);
                fetchConversations();
            } else if (response.data.title) {
                // Update title in sidebar if it changed
                setConversations(prev => prev.map(c =>
                    c.id === response.data.conversation_id ? { ...c, title: response.data.title } : c
                ));
            }
        } catch (error) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: (error as any).response?.data?.response || (error as any).response?.data?.error || (error as any).message || 'I encountered a connection error. Please verify your network and try again.',
                timestamp: new Date().toISOString()
            }]);
        } finally {
            setLoading(false);
        }
    };

    const renderChart = (data: any[]) => {
        if (!data || data.length === 0) return null;
        const keys = Object.keys(data[0]);
        const xKey = keys[0];
        const yKeys = keys.slice(1);

        return (
            <div className="h-64 w-full mt-4 bg-white/5 rounded-xl p-4 border border-white/10">
                <ResponsiveContainer width="100%" height="100%">
                    {selectedChartType === 'area' ? (
                        <AreaChart data={data}>
                            <defs>
                                <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                            <XAxis dataKey={xKey} stroke="#9ca3af" fontSize={11} tickLine={false} axisLine={false} />
                            <YAxis stroke="#9ca3af" fontSize={11} tickLine={false} axisLine={false} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
                                itemStyle={{ color: '#e5e7eb', fontSize: '12px' }}
                            />
                            {yKeys.map((key, i) => (
                                <Area key={key} type="monotone" dataKey={key} stroke={CHART_COLORS[i % CHART_COLORS.length]} fill="url(#colorGradient)" strokeWidth={2} />
                            ))}
                        </AreaChart>
                    ) : (
                        <BarChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                            <XAxis dataKey={xKey} stroke="#9ca3af" fontSize={11} tickLine={false} axisLine={false} />
                            <YAxis stroke="#9ca3af" fontSize={11} tickLine={false} axisLine={false} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
                                cursor={{ fill: '#ffffff05' }}
                            />
                            {yKeys.map((key, i) => (
                                <Bar key={key} dataKey={key} fill={CHART_COLORS[i % CHART_COLORS.length]} radius={[4, 4, 0, 0]} />
                            ))}
                        </BarChart>
                    )}
                </ResponsiveContainer>
                <div className="flex gap-2 mt-2 justify-end">
                    <button onClick={() => setSelectedChartType('area')} className={`text-[10px] px-2 py-1 rounded ${selectedChartType === 'area' ? 'bg-indigo-500 text-white' : 'bg-white/5 text-gray-400'}`}>Area</button>
                    <button onClick={() => setSelectedChartType('bar')} className={`text-[10px] px-2 py-1 rounded ${selectedChartType === 'bar' ? 'bg-indigo-500 text-white' : 'bg-white/5 text-gray-400'}`}>Bar</button>
                </div>
            </div>
        );
    };

    return (
        <div className="flex h-screen bg-[#0f1117] text-gray-100 font-sans overflow-hidden selection:bg-indigo-500/30">

            {/* Sidebar Navigation */}
            <aside className="w-64 bg-[#161b28] border-r border-white/5 flex flex-col shrink-0 z-20">
                <div className="p-6 flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                        <Brain className="w-5 h-5 text-white" />
                    </div>
                    <div className="flex flex-col">
                        <span className="font-bold text-lg tracking-tight text-white leading-tight">Sephlighty</span>
                        <span className="text-[10px] text-indigo-400 font-bold uppercase tracking-widest">Business Intelligence</span>
                    </div>
                </div>

                <nav className="flex-1 px-4 space-y-1 mt-4">
                    {[
                        { id: 'chat', label: 'Compnaion Chat', icon: Sparkles, active: true },
                        { id: 'dashboard', label: 'Analytics Board', icon: Layout, onClick: () => navigate('/dashboard') },
                        { id: 'sql', label: 'Data Explorer', icon: Database, onClick: () => navigate('/sql') },
                    ].map((item, idx) => (
                        <button
                            key={idx}
                            onClick={item.onClick}
                            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${item.active
                                ? 'bg-indigo-600/10 text-indigo-400'
                                : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                                }`}
                        >
                            <item.icon size={18} />
                            {item.label}
                        </button>
                    ))}

                    <div className="pt-8 pb-2 px-3 text-xs font-semibold text-gray-500 uppercase tracking-wider flex justify-between items-center">
                        <span>Workspace</span>
                        <button onClick={startNewSession} className="text-indigo-400 hover:text-indigo-300 text-[10px] font-bold">NEW</button>
                    </div>
                    <div className="space-y-1 max-h-[40vh] overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-gray-800">
                        {conversations.length === 0 ? (
                            <div className="px-3 py-4 text-center">
                                <p className="text-[10px] text-gray-600 italic">No history yet</p>
                            </div>
                        ) : (
                            conversations.map((conv) => (
                                <button
                                    key={conv.id}
                                    onClick={() => loadConversation(conv.id)}
                                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium transition-all text-left truncate ${selectedConvId === conv.id
                                        ? 'bg-indigo-600/10 text-indigo-400 border border-indigo-500/20'
                                        : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                                        }`}
                                >
                                    <MessageSquare size={14} className="shrink-0" />
                                    <span className="truncate">{conv.title}</span>
                                </button>
                            ))
                        )}
                    </div>
                </nav>

                <div className="p-4 border-t border-white/5">
                    <div className="flex items-center gap-3 p-2 rounded-lg hover:bg-white/5 transition-colors cursor-pointer" onClick={() => setShowSettings(true)}>
                        <div className="w-8 h-8 rounded-full bg-indigo-500/20 flex items-center justify-center text-indigo-400 font-bold border border-indigo-500/20">
                            {user?.first_name?.[0]}
                        </div>
                        <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-white truncate">{user?.first_name} {user?.last_name}</p>
                            <p className="text-xs text-gray-500 truncate">{user?.email}</p>
                        </div>
                        <Settings size={16} className="text-gray-500" />
                    </div>
                </div>
            </aside>

            {/* Main Content Area */}
            <main className="flex-1 flex flex-col relative min-w-0">
                {/* Header */}
                <header className="h-16 border-b border-white/5 bg-[#0f1117]/80 backdrop-blur-md flex items-center justify-between px-6 sticky top-0 z-10">
                    <div className="flex items-center gap-3">
                        <h2 className="font-semibold text-white">New Session</h2>
                        <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-green-500/10 text-green-400 border border-green-500/20">
                            OPERATIONAL
                        </span>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="text-xs text-gray-400 flex items-center gap-2">
                            <Clock size={14} />
                            <span>Latency: 14ms</span>
                        </div>
                    </div>
                </header>

                {/* Chat Stream */}
                <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 scrollbar-thin scrollbar-thumb-gray-800 scrollbar-track-transparent">
                    {messages.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center text-center max-w-2xl mx-auto">
                            <div className="w-20 h-20 bg-indigo-500/10 rounded-2xl flex items-center justify-center mb-8 border border-indigo-500/20 shadow-glow-indigo">
                                <Brain className="w-10 h-10 text-indigo-400" />
                            </div>
                            <h2 className="text-3xl font-bold text-white mb-4">How can I help you analyze today?</h2>
                            <p className="text-gray-400 mb-10 max-w-lg leading-relaxed">
                                I have access to your entire business database. Ask me about sales trends, inventory anomalies, or customer insights.
                            </p>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
                                {[
                                    "Analyze revenue trends for this quarter",
                                    "Identify top 5 selling products",
                                    "Show me customer churn risk",
                                    "Compare sales vs expenses"
                                ].map((prompt, i) => (
                                    <button
                                        key={i}
                                        onClick={() => setInput(prompt)}
                                        className="p-4 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 hover:border-indigo-500/30 text-left text-sm text-gray-300 transition-all group"
                                    >
                                        <span className="group-hover:text-white">{prompt}</span>
                                    </button>
                                ))}
                            </div>
                        </div>
                    ) : (
                        messages.map((msg, idx) => (
                            <motion.div
                                key={idx}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div className={`max-w-[80%] rounded-2xl p-5 ${msg.role === 'user'
                                    ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/10'
                                    : 'bg-[#1e2330] border border-white/5 text-gray-100 shadow-xl'
                                    }`}>
                                    {msg.role === 'assistant' && (
                                        <div className="flex items-center gap-2 mb-3 pb-3 border-b border-white/5">
                                            <Brain size={16} className="text-indigo-400" />
                                            <span className="text-xs font-bold text-indigo-300 uppercase tracking-wide">AI Assistant</span>
                                            {msg.confidence && (
                                                <span className="ml-auto text-[10px] bg-white/5 px-2 py-0.5 rounded text-gray-400">
                                                    {msg.confidence}% Conf
                                                </span>
                                            )}
                                        </div>
                                    )}

                                    <div className="prose prose-invert prose-sm max-w-none leading-relaxed whitespace-pre-wrap">
                                        {msg.content}
                                    </div>

                                    {msg.data && msg.data.length > 0 && (
                                        <div className="mt-4 pt-4 border-t border-white/5">
                                            {renderChart(msg.data)}
                                        </div>
                                    )}

                                    {msg.insights && msg.insights.length > 0 && (
                                        <div className="mt-4 space-y-2">
                                            {msg.insights.map((insight, i) => (
                                                <div key={i} className="flex gap-2 text-xs text-emerald-400/90 bg-emerald-500/5 p-2 rounded-lg border border-emerald-500/10">
                                                    <Sparkles size={14} className="shrink-0 mt-0.5" />
                                                    {insight}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </motion.div>
                        ))
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="p-4 md:p-6 bg-[#0f1117] border-t border-white/5">
                    <div className="max-w-4xl mx-auto relative rounded-xl shadow-2xl bg-[#1e2330] border border-indigo-500/20 transition-all focus-within:border-indigo-500/50 focus-within:ring-1 focus-within:ring-indigo-500/20">
                        <textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    sendMessage();
                                }
                            }}
                            placeholder="Ask a question about your business data..."
                            className="w-full bg-transparent border-none text-white placeholder-gray-500 px-4 py-4 min-h-[60px] max-h-[200px] resize-none focus:ring-0 text-base"
                            rows={1}
                        />
                        <div className="absolute bottom-2 right-2 flex items-center gap-2">
                            <button
                                onClick={sendMessage}
                                disabled={!input.trim() || loading}
                                className="p-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg shadow-indigo-600/20"
                            >
                                {loading ? <Activity size={18} className="animate-spin" /> : <Send size={18} />}
                            </button>
                        </div>
                    </div>
                    <p className="text-center text-[10px] text-gray-600 mt-3">
                        AI can make mistakes. Verify important data.
                    </p>
                </div>
            </main>

            {/* Settings Modal (Simplified) */}
            <AnimatePresence>
                {showSettings && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={() => setShowSettings(false)}
                        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
                    >
                        <motion.div
                            initial={{ scale: 0.95 }}
                            animate={{ scale: 1 }}
                            exit={{ scale: 0.95 }}
                            onClick={e => e.stopPropagation()}
                            className="bg-[#1e2330] border border-white/10 w-full max-w-lg rounded-2xl p-6 shadow-2xl"
                        >
                            <div className="flex items-center justify-between mb-6">
                                <h3 className="text-xl font-bold text-white">Settings</h3>
                                <button onClick={() => setShowSettings(false)} className="p-2 hover:bg-white/5 rounded-full text-gray-400 hover:text-white transition-colors">
                                    <X size={20} />
                                </button>
                            </div>

                            <div className="space-y-6">
                                <div className="space-y-4">
                                    <h4 className="text-sm font-semibold text-gray-400 uppercase tracking-wider text-xs">AI CONFIGURATION</h4>

                                    <div className="flex items-center justify-between">
                                        <label className="text-sm text-gray-200">Creativity Level</label>
                                        <input
                                            type="range"
                                            min="50" max="100"
                                            value={settings.confidenceThreshold}
                                            onChange={(e) => setSettings({ ...settings, confidenceThreshold: parseInt(e.target.value) })}
                                            className="accent-indigo-500"
                                        />
                                    </div>

                                    <div className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/5">
                                        <div className="flex items-center gap-3">
                                            <Database size={18} className="text-gray-400" />
                                            <span className="text-sm text-gray-200">Show SQL Queries</span>
                                        </div>
                                        <div
                                            onClick={() => setSettings({ ...settings, showSQL: !settings.showSQL })}
                                            className={`w-10 h-6 rounded-full p-1 cursor-pointer transition-colors ${settings.showSQL ? 'bg-indigo-600' : 'bg-gray-700'}`}
                                        >
                                            <div className={`w-4 h-4 rounded-full bg-white shadow-sm transition-transform ${settings.showSQL ? 'translate-x-4' : 'translate-x-0'}`} />
                                        </div>
                                    </div>
                                </div>

                                <div className="pt-6 border-t border-white/5 flex gap-3">
                                    <button onClick={() => { logout(); navigate('/login'); }} className="flex-1 py-3 rounded-xl bg-red-500/10 text-red-400 hover:bg-red-500/20 font-medium transition-colors flex items-center justify-center gap-2">
                                        <LogOut size={16} />
                                        Log Out
                                    </button>
                                    <button onClick={() => setShowSettings(false)} className="flex-1 py-3 rounded-xl bg-white/10 text-white hover:bg-white/15 font-medium transition-colors">
                                        Done
                                    </button>
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
