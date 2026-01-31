import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../lib/authStore';
import {
    Database, Play, Save, Download, RefreshCw, Layout, Sparkles,
    History, Table as TableIcon, Filter, Settings, X, LogOut
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function DataExplorer() {
    const { user, logout } = useAuthStore();
    const navigate = useNavigate();
    const [query, setQuery] = useState('SELECT * FROM orders LIMIT 10;');
    const [results, setResults] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const [showSettings, setShowSettings] = useState(false);

    // Mock execution
    const executeQuery = () => {
        setLoading(true);
        setTimeout(() => {
            setResults([
                { id: 1, date: '2024-01-20', customer: 'Acme Corp', amount: 1200.00, status: 'Completed' },
                { id: 2, date: '2024-01-21', customer: 'Globex Inc', amount: 850.50, status: 'Pending' },
                { id: 3, date: '2024-01-22', customer: 'Soylent Corp', amount: 2300.00, status: 'Completed' },
                { id: 4, date: '2024-01-23', customer: 'Initech', amount: 560.00, status: 'Failed' },
                { id: 5, date: '2024-01-24', customer: 'Umbrella Corp', amount: 4500.00, status: 'Completed' },
            ]);
            setLoading(false);
        }, 800);
    };

    return (
        <div className="flex h-screen bg-[#0f1117] text-gray-100 font-sans overflow-hidden selection:bg-indigo-500/30">
            {/* Sidebar Navigation */}
            <aside className="w-64 bg-[#161b28] border-r border-white/5 flex flex-col shrink-0 z-20">
                <div className="p-6 flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                        <Database className="w-5 h-5 text-white" />
                    </div>
                    <span className="font-bold text-lg tracking-tight text-white">Data Explorer</span>
                </div>

                <nav className="flex-1 px-4 space-y-1 mt-4">
                    {[
                        { id: 'chat', label: 'Compnaion Chat', icon: Sparkles, onClick: () => navigate('/chat') },
                        { id: 'dashboard', label: 'Analytics Board', icon: Layout, onClick: () => navigate('/dashboard') },
                        { id: 'sql', label: 'Data Explorer', icon: Database, active: true },
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

            {/* Main Content */}
            <main className="flex-1 flex flex-col min-w-0 bg-[#0f1117]">
                <header className="h-16 border-b border-white/5 bg-[#0f1117]/80 backdrop-blur-md flex items-center justify-between px-6 sticky top-0 z-10">
                    <div className="flex items-center gap-3">
                        <h2 className="font-semibold text-white">SQL Console</h2>
                        <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                            READ ONLY
                        </span>
                    </div>
                    <div className="flex gap-2">
                        <button className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-xs font-medium text-gray-300 transition-colors border border-white/5">
                            <History size={14} />
                            History
                        </button>
                        <button className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-xs font-medium text-gray-300 transition-colors border border-white/5">
                            <Save size={14} />
                            Saved Queries
                        </button>
                    </div>
                </header>

                <div className="flex-1 overflow-hidden flex flex-col p-6 gap-6">
                    {/* Query Editor */}
                    <div className="flex-shrink-0 bg-[#1e2330] border border-white/5 rounded-xl shadow-xl overflow-hidden flex flex-col" style={{ height: '300px' }}>
                        <div className="flex items-center justify-between px-4 py-3 bg-white/5 border-b border-white/5">
                            <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">Query Editor</span>
                            <div className="flex gap-2">
                                <button className="p-1.5 hover:bg-white/10 rounded text-gray-400 hover:text-white transition-colors">
                                    <Download size={14} />
                                </button>
                                <button
                                    onClick={executeQuery}
                                    disabled={loading}
                                    className="flex items-center gap-2 px-3 py-1 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition-colors"
                                >
                                    {loading ? <RefreshCw size={12} className="animate-spin" /> : <Play size={12} />}
                                    Run Query
                                </button>
                            </div>
                        </div>
                        <div className="flex-1 relative">
                            <textarea
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                className="w-full h-full bg-[#1e2330] text-gray-100 font-mono text-sm p-4 resize-none focus:ring-0 border-none outline-none"
                                spellCheck={false}
                            />
                        </div>
                    </div>

                    {/* Results Table */}
                    <div className="flex-1 bg-[#1e2330] border border-white/5 rounded-xl shadow-xl overflow-hidden flex flex-col">
                        <div className="flex items-center justify-between px-4 py-3 bg-white/5 border-b border-white/5">
                            <div className="flex items-center gap-2">
                                <TableIcon size={14} className="text-indigo-400" />
                                <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">Results {results.length > 0 && `(${results.length} rows)`}</span>
                            </div>
                            <div className="flex gap-2">
                                <button className="flex items-center gap-1.5 px-2 py-1 rounded hover:bg-white/5 text-xs text-gray-400 hover:text-white transition-colors">
                                    <Filter size={12} />
                                    Filter
                                </button>
                            </div>
                        </div>
                        <div className="flex-1 overflow-auto">
                            {results.length > 0 ? (
                                <table className="w-full text-left text-sm text-gray-400">
                                    <thead className="bg-white/5 text-gray-200 font-medium sticky top-0">
                                        <tr>
                                            {Object.keys(results[0]).map((key) => (
                                                <th key={key} className="px-6 py-3 border-b border-white/5">{key}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-white/5">
                                        {results.map((row, i) => (
                                            <tr key={i} className="hover:bg-white/5 transition-colors">
                                                {Object.values(row).map((val: any, j) => (
                                                    <td key={j} className="px-6 py-3">{val}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center text-gray-500 gap-3">
                                    <Database size={32} className="opacity-20" />
                                    <p className="text-sm">Run a query to view results</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>

            {/* Settings Modal - Reused logic */}
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
                            <div className="pt-6 border-t border-white/5 flex gap-3">
                                <button onClick={() => { logout(); navigate('/login'); }} className="flex-1 py-3 rounded-xl bg-red-500/10 text-red-400 hover:bg-red-500/20 font-medium transition-colors flex items-center justify-center gap-2">
                                    <LogOut size={16} />
                                    Log Out
                                </button>
                                <button onClick={() => setShowSettings(false)} className="flex-1 py-3 rounded-xl bg-white/10 text-white hover:bg-white/15 font-medium transition-colors">
                                    Done
                                </button>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
