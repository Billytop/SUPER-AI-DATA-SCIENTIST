import { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, X, Bot, User, FileDown, ExternalLink } from 'lucide-react';
import { AreaChart, Area, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import api from '../lib/api';
import { cn } from '../lib/utils';

export function AIChat() {
    const [isOpen, setIsOpen] = useState(false);
    const [query, setQuery] = useState('');
    const [messages, setMessages] = useState<{
        role: 'user' | 'ai',
        content: string,
        sql?: string,
        visual?: 'text' | 'chart',
        data?: any[]
    }[]>([
        { role: 'ai', content: "Hello! I'm Sephlighty AI. Ready to analyze your business data." }
    ]);
    const [isLoading, setIsLoading] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!query.trim()) return;

        const userMsg = query;
        setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        setQuery('');
        setIsLoading(true);

        try {
            const { data } = await api.post('/ask_ai/', { query: userMsg });
            setMessages(prev => [...prev, {
                role: 'ai',
                content: data.answer,
                sql: data.sql,
                visual: data.visual,
                data: data.data
            }]);
        } catch (error) {
            setMessages(prev => [...prev, { role: 'ai', content: "I encountered an error connecting to my brain. Please try again." }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <>
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 20, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 20, scale: 0.95 }}
                        className="fixed bottom-24 right-6 w-[450px] h-[650px] glass-panel rounded-2xl shadow-2xl flex flex-col z-50 overflow-hidden border border-white/10"
                    >
                        {/* Header */}
                        <div className="p-4 border-b border-white/10 flex items-center justify-between bg-gradient-to-r from-blue-900/30 to-purple-900/30 backdrop-blur-md">
                            <div className="flex items-center gap-2">
                                <div className="p-1.5 bg-blue-500/20 rounded-lg">
                                    <Sparkles className="w-5 h-5 text-blue-400 animate-pulse" />
                                </div>
                                <div>
                                    <span className="font-bold text-white tracking-wide">Sephlighty<span className="text-blue-400">AI</span></span>
                                    <p className="text-[10px] text-blue-300 font-medium uppercase tracking-wider">Business Intelligence v2.0</p>
                                </div>
                            </div>
                            <button onClick={() => setIsOpen(false)} className="text-gray-400 hover:text-white transition-colors p-1 hover:bg-white/10 rounded-lg">
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Messages */}
                        <div className="flex-1 overflow-y-auto p-4 space-y-6" ref={scrollRef}>
                            {messages.map((msg, idx) => (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    key={idx}
                                    className={cn("flex gap-3", msg.role === 'user' ? "flex-row-reverse" : "flex-row")}
                                >
                                    <div className={cn(
                                        "w-8 h-8 rounded-full flex items-center justify-center shrink-0 border border-white/10",
                                        msg.role === 'ai' ? "bg-blue-600/20 text-blue-400" : "bg-purple-600/20 text-purple-400"
                                    )}>
                                        {msg.role === 'ai' ? <Bot className="w-4 h-4" /> : <User className="w-4 h-4" />}
                                    </div>

                                    <div className={cn(
                                        "flex flex-col max-w-[85%]",
                                        msg.role === 'user' ? "items-end" : "items-start"
                                    )}>
                                        <div className={cn(
                                            "p-4 rounded-2xl text-[14px] leading-relaxed shadow-lg backdrop-blur-sm",
                                            msg.role === 'user'
                                                ? "bg-gradient-to-br from-blue-600 to-blue-700 text-white rounded-tr-sm border border-blue-400/20"
                                                : "bg-white/5 text-gray-200 rounded-tl-sm border border-white/10"
                                        )}>
                                            <div className="prose prose-invert prose-p:my-1 prose-headings:my-2 prose-strong:text-blue-300 max-w-none">
                                                {msg.content.split('\n').filter(line => !line.includes('[DOWNLOAD_ACTION]')).map((line, i) => (
                                                    <p key={i}>{line}</p>
                                                ))}
                                            </div>

                                            {/* Download Button Action */}
                                            {msg.role === 'ai' && msg.content.includes('[DOWNLOAD_ACTION]') && (() => {
                                                const url = msg.content.split('[DOWNLOAD_ACTION]: ')[1]?.split('\n')[0].trim();
                                                const isPdf = url?.toLowerCase().endsWith('.pdf');
                                                const fileType = isPdf ? 'PDF DOCUMENT' : 'EXCEL REPORT';

                                                return (
                                                    <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-xl flex flex-col gap-3">
                                                        <div className="flex items-center gap-2 text-blue-300 text-[12px] font-medium">
                                                            <FileDown className="w-4 h-4" />
                                                            <span>{isPdf ? 'PDF Document' : 'Excel Report'} Ready for Download</span>
                                                        </div>
                                                        <a
                                                            href={url}
                                                            target="_blank"
                                                            rel="noopener noreferrer"
                                                            className="flex items-center justify-center gap-2 py-2.5 px-4 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-all font-bold text-[13px] shadow-lg shadow-blue-500/20 active:scale-[0.98]"
                                                        >
                                                            DOWNLOAD {fileType}
                                                            <ExternalLink className="w-3.5 h-3.5" />
                                                        </a>
                                                    </div>
                                                );
                                            })()}

                                            {msg.visual === 'chart' && msg.data && (
                                                <div className="mt-4 h-40 w-full bg-black/40 rounded-xl border border-white/5 p-2">
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <AreaChart data={msg.data}>
                                                            <defs>
                                                                <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                                                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                                                </linearGradient>
                                                            </defs>
                                                            <XAxis dataKey="name" hide />
                                                            <YAxis hide />
                                                            <Tooltip
                                                                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                                                                itemStyle={{ color: '#e2e8f0' }}
                                                            />
                                                            <Area type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} fill="url(#colorVal)" />
                                                        </AreaChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            )}
                                        </div>
                                        {msg.sql && (
                                            <div className="mt-2 text-[10px] font-mono text-gray-500 bg-black/20 px-2 py-1 rounded border border-white/5 max-w-full overflow-hidden text-ellipsis whitespace-nowrap">
                                                SQL: {msg.sql}
                                            </div>
                                        )}
                                    </div>
                                </motion.div>
                            ))}
                            {isLoading && (
                                <motion.div
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    className="flex gap-3"
                                >
                                    <div className="w-8 h-8 rounded-full bg-blue-600/20 text-blue-400 flex items-center justify-center border border-white/10">
                                        <Bot className="w-4 h-4" />
                                    </div>
                                    <div className="bg-white/5 border border-white/10 p-4 rounded-2xl rounded-tl-sm flex items-center gap-1.5 h-12">
                                        <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                        <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                                        <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                                    </div>
                                </motion.div>
                            )}
                        </div>

                        {/* Input */}
                        <form onSubmit={handleSubmit} className="p-4 border-t border-white/10 bg-black/20 backdrop-blur-md">
                            <div className="relative group bg-black/40 border border-white/10 rounded-xl transition-all shadow-inner focus-within:ring-1 focus-within:ring-blue-500/50 focus-within:border-blue-500/50">
                                <textarea
                                    value={query}
                                    onChange={(e) => {
                                        setQuery(e.target.value);
                                        e.target.style.height = 'auto';
                                        e.target.style.height = e.target.scrollHeight + 'px';
                                    }}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter' && !e.shiftKey) {
                                            e.preventDefault();
                                            handleSubmit(e);
                                        }
                                    }}
                                    placeholder="Type your question or paste data..."
                                    className="w-full bg-transparent border-none rounded-xl pl-4 pr-12 py-3.5 text-sm text-white placeholder-gray-500 focus:ring-0 resize-none overflow-hidden min-h-[50px] max-h-[200px]"
                                    disabled={isLoading}
                                    rows={1}
                                    style={{ height: '52px' }}
                                />
                                <button
                                    type="submit"
                                    disabled={isLoading || !query.trim()}
                                    className="absolute right-2 bottom-2 p-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg shadow-blue-500/20"
                                >
                                    <Send className="w-4 h-4" />
                                </button>
                            </div>
                            <div className="mt-2 flex justify-center gap-4 text-[10px] text-gray-500">
                                <span>✨ Powered by SephlightyBrain v3</span>
                            </div>
                        </form>
                    </motion.div>
                )}
            </AnimatePresence>

            <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setIsOpen(!isOpen)}
                className="fixed bottom-6 right-6 h-14 w-14 bg-blue-600 hover:bg-blue-500 rounded-full shadow-lg shadow-blue-600/30 flex items-center justify-center text-white z-50 transition-all border border-white/20 animate-pulse-glow"
            >
                {isOpen ? <X className="w-6 h-6" /> : <Sparkles className="w-6 h-6" />}
            </motion.button>
        </>
    );
}
