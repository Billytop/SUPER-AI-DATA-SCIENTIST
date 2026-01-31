import { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '../lib/authStore';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Mail, Lock, ArrowRight, Shield, Globe, Cpu, Command, Sparkles, Orbit, Fingerprint, Activity } from 'lucide-react';

export default function Login() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [isScanning, setIsScanning] = useState(true);

    const { login } = useAuthStore();
    const navigate = useNavigate();

    useEffect(() => {
        const timer = setTimeout(() => setIsScanning(false), 2000);
        return () => clearTimeout(timer);
    }, []);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            await login(email, password);
            navigate('/chat');
        } catch (err: any) {
            console.error(err);
            const serverError = err.response?.data?.detail || err.response?.data?.message || err.response?.data?.error;
            const networkError = err.message || 'Unknown error occurred';
            setError(serverError || networkError || 'Handshake failed. Protocol mismatch.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="h-screen w-full bg-[#02040a] relative flex items-center justify-center overflow-hidden noise-bg font-sans">
            <div className="scanline"></div>

            {/* Immersive Neural Background */}
            <div className="absolute inset-0 pointer-events-none data-grid opacity-20"></div>

            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="absolute inset-0 pointer-events-none"
            >
                <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] bg-indigo-500/10 rounded-full blur-[180px] animate-pulse"></div>
                <div className="absolute bottom-[-20%] right-[-10%] w-[60%] h-[60%] bg-purple-500/5 rounded-full blur-[180px] animate-pulse" style={{ animationDelay: '2s' }}></div>
            </motion.div>

            <div className="w-full max-w-[1280px] px-8 relative z-20">
                <AnimatePresence>
                    {isScanning ? (
                        <motion.div
                            key="scanner"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0, scale: 1.05 }}
                            className="absolute inset-0 flex flex-col items-center justify-center z-50 bg-[#02040a]/80 backdrop-blur-3xl"
                        >
                            <div className="relative">
                                <motion.div
                                    animate={{ rotate: 360 }}
                                    transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                                    className="w-32 h-32 border-2 border-white/5 border-t-white/40 rounded-full"
                                ></motion.div>
                                <Fingerprint className="absolute inset-0 m-auto w-12 h-12 text-white/40 animate-pulse" />
                            </div>
                            <motion.p
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 0.5 }}
                                className="mt-8 text-[11px] font-black uppercase tracking-[0.5em] text-white"
                            >
                                Initializing Secure Handshake...
                            </motion.p>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="main"
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.8, ease: "circOut" }}
                            className="grid lg:grid-cols-[1.2fr_1fr] glass-panel bg-black/40 rounded-[40px] overflow-hidden shadow-2xl border border-white/5"
                        >

                            {/* Left Panel: Neural Intelligence Narrative */}
                            <div className="hidden lg:flex flex-col p-16 border-r border-white/5 relative bg-gradient-to-b from-white/[0.02] to-transparent">
                                <motion.div
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 0.2 }}
                                    className="flex items-center gap-4 mb-24"
                                >
                                    <div className="w-14 h-14 bg-white rounded-[20px] flex items-center justify-center shadow-glow">
                                        <Brain className="w-8 h-8 text-black" />
                                    </div>
                                    <div className="space-y-0.5">
                                        <span className="text-2xl font-black tracking-tighter text-white">SEPHLIGHTY</span>
                                        <p className="text-[10px] font-bold text-white/30 uppercase tracking-[0.3em]">Neural Core v4.0</p>
                                    </div>
                                </motion.div>

                                <div className="space-y-12">
                                    <motion.div
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: 0.4 }}
                                        className="space-y-6"
                                    >
                                        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-[10px] font-black tracking-[0.2em] text-white/40 uppercase">
                                            <Activity className="w-3 h-3 text-indigo-400" />
                                            Global Mesh: Synchronized
                                        </div>
                                        <h2 className="text-7xl font-black tracking-tighter text-white leading-[0.85]">
                                            Cognitive <br />
                                            <span className="text-white/20">Data Logic.</span>
                                        </h2>
                                    </motion.div>

                                    <motion.p
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 0.4 }}
                                        transition={{ delay: 0.6 }}
                                        className="text-xl text-white font-medium max-w-sm leading-relaxed"
                                    >
                                        Leverage 672 trained neural gates for autonomous decision logic and real-time surveillance.
                                    </motion.p>

                                    <motion.div
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        transition={{ delay: 0.8 }}
                                        className="pt-12 grid grid-cols-2 gap-12 border-t border-white/5"
                                    >
                                        <div className="space-y-2">
                                            <div className="text-4xl font-black text-white tracking-tighter">0.12<span className="text-white/20 italic">ms</span></div>
                                            <div className="text-[11px] text-white/20 uppercase tracking-[0.2em] font-bold">Inference Rate</div>
                                        </div>
                                        <div className="space-y-2">
                                            <div className="text-4xl font-black text-white tracking-tighter">99.99<span className="text-white/20 italic">%</span></div>
                                            <div className="text-[11px] text-white/20 uppercase tracking-[0.2em] font-bold">Core Precision</div>
                                        </div>
                                    </motion.div>
                                </div>

                                <div className="mt-auto pt-20 flex items-center gap-6">
                                    <div className="flex -space-x-3">
                                        {[1, 2, 3].map(i => (
                                            <div key={i} className="w-12 h-12 rounded-2xl border-2 border-black bg-white/5 backdrop-blur-xl flex items-center justify-center">
                                                <Cpu className="w-5 h-5 text-white/20" />
                                            </div>
                                        ))}
                                    </div>
                                    <p className="text-[11px] text-white/20 font-bold uppercase tracking-widest leading-loose">
                                        Tier 4 Identity <br /> Verified by Mesh
                                    </p>
                                </div>
                            </div>

                            {/* Right Panel: High-Security Input */}
                            <div className="p-12 lg:p-24 flex flex-col justify-center bg-[#02040a]/60">
                                <div className="max-w-md mx-auto w-full space-y-12">
                                    <div className="space-y-3">
                                        <h3 className="text-5xl font-black text-white tracking-tighter">Deploy Access.</h3>
                                        <p className="text-white/30 font-bold italic tracking-tight">Handshake protocol required for terminal entry.</p>
                                    </div>

                                    {error && (
                                        <motion.div
                                            initial={{ opacity: 0, x: -10 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            className="p-5 bg-red-500/10 border border-red-500/20 rounded-2xl text-red-400 text-sm font-bold flex items-center gap-4"
                                        >
                                            <Shield className="w-5 h-5 shrink-0" />
                                            {error}
                                        </motion.div>
                                    )}

                                    <form onSubmit={handleSubmit} className="space-y-8">
                                        <div className="space-y-3">
                                            <label className="text-[11px] font-black text-white/20 uppercase tracking-[0.3em] ml-2">Operator Identity</label>
                                            <div className="relative group">
                                                <Mail className="absolute left-6 top-1/2 -translate-y-1/2 w-5 h-5 text-white/10 group-focus-within:text-indigo-400 transition-colors" />
                                                <input
                                                    type="email"
                                                    value={email}
                                                    onChange={(e) => setEmail(e.target.value)}
                                                    className="w-full bg-white/[0.02] border border-white/5 rounded-3xl pl-16 pr-6 py-6 text-white placeholder:text-white/5 focus:outline-none focus:ring-1 focus:ring-white/20 focus:bg-white/[0.04] transition-all font-bold text-lg"
                                                    placeholder="operator@neural.mesh"
                                                    required
                                                />
                                            </div>
                                        </div>

                                        <div className="space-y-3">
                                            <div className="flex items-center justify-between ml-2">
                                                <label className="text-[11px] font-black text-white/20 uppercase tracking-[0.3em]">Access Sequence</label>
                                                <a href="#" className="text-[10px] text-white/20 hover:text-white uppercase tracking-widest font-black transition-all">Emergency Override</a>
                                            </div>
                                            <div className="relative group">
                                                <Lock className="absolute left-6 top-1/2 -translate-y-1/2 w-5 h-5 text-white/10 group-focus-within:text-indigo-400 transition-colors" />
                                                <input
                                                    type="password"
                                                    value={password}
                                                    onChange={(e) => setPassword(e.target.value)}
                                                    className="w-full bg-white/[0.02] border border-white/5 rounded-3xl pl-16 pr-6 py-6 text-white placeholder:text-white/5 focus:outline-none focus:ring-1 focus:ring-white/20 focus:bg-white/[0.04] transition-all font-bold text-lg"
                                                    placeholder="••••••••••••"
                                                    required
                                                />
                                            </div>
                                        </div>

                                        <div className="flex items-center gap-4 py-2">
                                            <div className="relative flex items-center cursor-pointer group">
                                                <input type="checkbox" id="remember" className="peer sr-only" />
                                                <div className="w-6 h-6 border-2 border-white/10 rounded-lg bg-white/5 transition-all peer-checked:bg-white peer-checked:border-white group-hover:border-white/30"></div>
                                                <Shield className="absolute w-3 h-3 text-black opacity-0 peer-checked:opacity-100 transition-opacity left-1.5" />
                                            </div>
                                            <label htmlFor="remember" className="text-xs text-white/20 font-black uppercase tracking-widest cursor-pointer select-none">Maintain Session Persistence</label>
                                        </div>

                                        <button
                                            type="submit"
                                            disabled={loading}
                                            className="w-full mt-10 bg-white text-black font-black py-7 rounded-[28px] hover:bg-gray-100 active:scale-[0.96] transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_30px_60px_-15px_rgba(255,255,255,0.2)] flex items-center justify-center gap-6 text-xl group overflow-hidden"
                                        >
                                            {loading ? (
                                                <div className="w-8 h-8 border-4 border-black/20 border-t-black rounded-full animate-spin"></div>
                                            ) : (
                                                <>
                                                    <span className="tracking-tighter">INITIALIZE SESSION</span>
                                                    <ArrowRight className="w-6 h-6 transform group-hover:translate-x-1.5 transition-transform" />
                                                </>
                                            )}
                                        </button>
                                    </form>

                                    <div className="mt-16 pt-12 border-t border-white/5 text-center">
                                        <p className="text-white/20 text-[11px] font-black uppercase tracking-[0.3em]">
                                            New operator joining?{' '}
                                            <Link to="/register" className="text-white hover:text-indigo-400 transition-all border-b border-white/20 pb-0.5 ml-2 hover:border-indigo-400/50">
                                                Register Node
                                            </Link>
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Global Footer Security Stats */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1 }}
                    className="mt-16 flex flex-wrap justify-center gap-12 text-[10px] text-white/5 uppercase tracking-[0.5em] font-black"
                >
                    <span className="flex items-center gap-3"><Orbit className="w-4 h-4" /> Mesh Surveillance Active</span>
                    <span className="flex items-center gap-3"><Cpu className="w-4 h-4" /> Neural Gates Scaled</span>
                    <span className="flex items-center gap-3"><Globe className="w-4 h-4" /> East African Region</span>
                    <span className="flex items-center gap-3"><Command className="w-4 h-4" /> Kernel v4.2.0-Ent</span>
                </motion.div>
            </div>
        </div>
    );
}
